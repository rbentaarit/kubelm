"""QLoRA SFT entry point for kubelm-edge.

Runs on a rented GPU box (A100 / H100 / large consumer GPU). NOT
tested on the maintainer's local macOS M1 — `unsloth` doesn't
build cleanly on Apple Silicon.

Usage:
    uv run python training/sft.py \
        --config training/configs/kubelm-edge-v0.yaml \
        --out runs/kubelm-edge-v0-attempt-1/

The script is deliberately thin: it loads a YAML config, loads the
trajectory JSONL files listed in the config, applies the filter,
formats the messages via Qwen's chat template, hands the dataset to
Unsloth's `SFTTrainer`, saves the adapter and (optionally) a merged
copy, and writes a per-step training log.

Design notes:
- The config is the reproducibility surface. Don't hardcode
  hyperparameters here; the YAML is authoritative.
- Filter logic is plain Python rather than `datasets.filter` because
  the filter conditions are nested (`provenance.review_status`,
  `quality.conclusion_rubric_passed`) and `datasets.filter` would
  need a lambda anyway.
- Chat-template application happens INSIDE the SFTTrainer via
  Unsloth's `dataset_text_field` + a formatting function. We don't
  pre-tokenize because Unsloth's mask-on-completion patching needs
  to see structured `messages`.

If Unsloth becomes a blocker, the same script structure works with
vanilla `trl.SFTTrainer` + `bitsandbytes`; replace the
`FastLanguageModel.from_pretrained` import with
`AutoModelForCausalLM.from_pretrained` + a manual
`prepare_model_for_kbit_training`. The dataset and trainer config
are the same.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _get_nested(d: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = d
    for part in dotted_key.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part, default)
    return cur


def _record_passes_filter(record: dict[str, Any], filt: dict[str, Any]) -> bool:
    for key, allowed in filt.items():
        value = _get_nested(record, key)
        if isinstance(allowed, list):
            if value not in allowed:
                return False
        elif value != allowed:
            return False
    return True


def _load_trajectory_dataset(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Load every JSONL source listed in the config, apply the filter, return records."""
    src_paths = [REPO_ROOT / p for p in cfg["dataset"]["sources"]]
    filt = cfg["dataset"].get("filter", {})

    records: list[dict[str, Any]] = []
    for src in src_paths:
        if not src.exists():
            print(f"  WARN: dataset source missing: {src}", file=sys.stderr)
            continue
        for line in src.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if _record_passes_filter(rec, filt):
                records.append(rec)

    seed = cfg["dataset"].get("shuffle_seed")
    if seed is not None:
        import random

        random.Random(int(seed)).shuffle(records)

    print(f"  loaded {len(records)} records after filter", file=sys.stderr)
    return records


def _to_sft_example(record: dict[str, Any]) -> dict[str, Any]:
    """Strip everything except `messages` — that's the only thing the trainer sees."""
    return {"messages": record["messages"]}


def _smoke_test_masking(trainer: Any, tokenizer: Any) -> None:
    """Pull one batch, verify non-assistant tokens are masked (label == -100).

    For trajectory SFT, only assistant turns should contribute to the loss.
    If `train_on_responses_only` was silently ignored or applied to the
    wrong region, every token gets a real label (training wastes gradient
    on memorizing tool-result JSON) or no tokens do (training collapses).
    This check catches both before the rental clock starts.

    Our trajectory shape: tool-result JSON dominates each sample
    (K8sGPT MCP dumps are 1-5K tokens × 4-8 messages = bulk of tokens),
    so the expected healthy mask ratio is ~90-99% — the small remainder
    is the assistant's tool_call block + final conclusion. The decoded
    sample of unmasked tokens is the real signal of correctness; the
    percentage alone can mislead.
    """
    print("=== smoke-test: assistant-only loss masking ===", file=sys.stderr)
    loader = trainer.get_train_dataloader()
    batch = next(iter(loader))
    if "labels" not in batch:
        print("ERROR: batch has no 'labels' key; cannot verify masking.", file=sys.stderr)
        raise SystemExit(2)

    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]
    total = int(labels.numel())
    masked = int((labels == -100).sum().item())
    unmasked = total - masked
    pct = 100.0 * masked / total if total else 0.0
    print(f"  total tokens in sample[0]: {total}", file=sys.stderr)
    print(f"  masked (-100):             {masked} ({pct:.1f}%)", file=sys.stderr)
    print(f"  unmasked (loss-bearing):   {unmasked}", file=sys.stderr)

    # ALWAYS print the decoded unmasked region BEFORE evaluating
    # pass/fail. The percentage is a coarse signal; the decoded text is
    # what tells us whether the right region is loss-bearing.
    keep = [
        int(tok)
        for tok, lbl in zip(input_ids.tolist(), labels.tolist(), strict=True)
        if lbl != -100
    ]
    if keep:
        decoded = tokenizer.decode(keep[:400], skip_special_tokens=False)
        print(
            f"  first ~400 unmasked tokens decode to:\n    {decoded!r}",
            file=sys.stderr,
        )
    else:
        print("  no unmasked tokens — mask is eating the entire sequence", file=sys.stderr)

    # Thresholds:
    # - pct < 50%: mask isn't taking effect. Whatever the decoded region
    #   says, training would still leak gradient onto user/tool tokens.
    # - unmasked == 0: mask ate everything. Training would have nothing
    #   to learn from.
    # - 50% <= pct < 100%: PASS. Read the decoded sample to confirm.
    #   For our corpus, 90-99% is normal. Anything below ~70% suggests
    #   user/tool tokens are leaking into loss.
    if pct < 50.0:
        print(
            "FAIL: <50% of tokens masked — train_on_responses_only is NOT "
            "taking effect, or it's masking the wrong region. The decoded "
            "sample above shows what IS being trained on. Investigate "
            "before any paid run.",
            file=sys.stderr,
        )
        raise SystemExit(3)
    if unmasked == 0:
        print(
            "FAIL: every token masked — train_on_responses_only matched "
            "no assistant boundary at all. Check that the chat template "
            "renders Qwen 2.5's `<|im_start|>assistant\\n` exactly.",
            file=sys.stderr,
        )
        raise SystemExit(3)

    print(
        f"PASS: {pct:.1f}% masked, {unmasked} loss-bearing tokens.",
        file=sys.stderr,
    )
    print(
        "  Eyeball the decoded sample above — it should be the assistant's "
        "tool_call JSON and/or the final conclusion, NOT tool-result text "
        "or user goals. If the decoded region is the wrong content, abort.",
        file=sys.stderr,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the config and dataset, print stats, but don't import torch or train.",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Load everything (model, tokenizer, dataset, trainer), pull one "
            "batch through the dataloader, verify assistant_only_loss masking "
            "actually masks non-assistant tokens, then exit before training. "
            "Use this on the GPU box BEFORE a paid run to confirm the labels "
            "look right — silent mis-masking is the single biggest money-burn "
            "risk for QLoRA SFT on trajectory data."
        ),
    )
    args = p.parse_args()

    cfg = _load_config(args.config)
    print(f"loaded config: {args.config}", file=sys.stderr)
    print(
        f"  base_model: {cfg['base_model']} (rev {cfg.get('base_model_revision', '<none>')})",
        file=sys.stderr,
    )
    print(f"  dataset sources: {len(cfg['dataset']['sources'])}", file=sys.stderr)

    records = _load_trajectory_dataset(cfg)
    if not records:
        print("ERROR: no records loaded; check dataset sources and filter.", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("dry-run: dataset loaded successfully; not invoking the trainer.", file=sys.stderr)
        return 0

    # Heavy imports deferred so --dry-run works without a CUDA install.
    # Unsloth must be imported BEFORE trl / transformers / peft so its
    # monkey-patching of those libraries' internals takes effect — see
    # the "Unsloth should be imported before [trl, transformers, peft]"
    # warning that fires otherwise.
    from unsloth import FastLanguageModel  # noqa: I001 — order is load-bearing
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    print(f"loading base model {cfg['base_model']}...", file=sys.stderr)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        dtype=None,  # auto
    )

    # Unsloth's get_peft_model sets task_type="CAUSAL_LM" internally
    # (FastLanguageModel only supports causal LMs). Passing task_type
    # from outside collides with the internal kwarg and raises
    # `TypeError: dict() got multiple values for keyword argument 'task_type'`.
    # The YAML still carries task_type for documentation; we just don't
    # forward it to the wrapper.
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias=cfg["lora"]["bias"],
    )

    # Pre-format records into a `text` column using the model's bundled
    # chat template. Unsloth's compiled SFTTrainer wrapper does not
    # auto-handle conversational (`messages`) datasets the way vanilla
    # trl 0.24 does — it raises `Unsloth: You must specify a
    # formatting_func`. Materializing the chat-templated text up front
    # is the documented Unsloth path and keeps the dataset reproducible
    # (the rendered string is captured in the run output, not deferred
    # to a closure).
    def _render(example: dict[str, Any]) -> dict[str, Any]:
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    train_dataset = Dataset.from_list([_to_sft_example(r) for r in records]).map(
        _render, remove_columns=["messages"]
    )

    train_cfg = cfg["training"]
    # Note: we DO NOT set `assistant_only_loss` on SFTConfig. Unsloth's
    # compiled trainer ignores that field and has its own masking utility
    # (`train_on_responses_only`), which we apply after construction
    # below. Setting the kwarg here would be a silent no-op at best and
    # a TypeError at worst depending on the version drift between
    # Unsloth's wrapper and trl 0.24's SFTConfig.
    sft_config = SFTConfig(
        output_dir=str(args.out),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        optim=train_cfg["optim"],
        bf16=train_cfg["bf16"],
        fp16=train_cfg["fp16"],
        max_grad_norm=train_cfg["max_grad_norm"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=train_cfg["save_total_limit"],
        seed=train_cfg["seed"],
        report_to=cfg.get("report_to") or [],
        max_seq_length=cfg["max_seq_length"],
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=sft_config,
    )

    # Replace the trainer's data collator with one that masks every
    # token outside an assistant turn. This is Unsloth's equivalent of
    # trl's `assistant_only_loss=True`. The instruction/response
    # delimiters are Qwen 2.5's ChatML boundary tokens — any tool/user
    # turn starts with `<|im_start|>user\n`, every assistant turn with
    # `<|im_start|>assistant\n`. Tool responses are rendered inside
    # user-tagged blocks by Qwen's template, so they're correctly
    # masked alongside actual user turns.
    from unsloth.chat_templates import train_on_responses_only

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    if args.smoke_test:
        _smoke_test_masking(trainer, tokenizer)
        return 0

    trainer.train()

    # Save adapter (small) and merged weights (large, optional)
    adapter_dir = args.out / cfg["output"]["adapter_dir"]
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    merged_dir = args.out / cfg["output"]["merged_dir"]
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    # Dump the resolved config alongside the artifacts so the run is
    # auditable from its output dir alone.
    (args.out / "resolved_config.yaml").write_text(yaml.safe_dump(cfg))
    print(f"adapter saved to {adapter_dir}", file=sys.stderr)
    print(f"merged model saved to {merged_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
