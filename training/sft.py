"""QLoRA SFT entry point for kubelm-standard.

Runs on a rented GPU box (A100 / H100 / large consumer GPU). NOT
tested on the maintainer's local macOS M1 — `unsloth` doesn't
build cleanly on Apple Silicon.

Usage:
    uv run python training/sft.py \
        --config training/configs/kubelm-standard-v0.yaml \
        --out runs/kubelm-standard-v0-attempt-1/

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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the config and dataset, print stats, but don't import torch or train.",
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
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    print(f"loading base model {cfg['base_model']}...", file=sys.stderr)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        dtype=None,  # auto
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias=cfg["lora"]["bias"],
        task_type=cfg["lora"]["task_type"],
    )

    tokenizer.chat_template = (
        tokenizer.chat_template  # rely on the model's bundled template (Qwen ships one)
    )

    train_dataset = Dataset.from_list([_to_sft_example(r) for r in records])

    train_cfg = cfg["training"]
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
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=sft_config,
    )

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
