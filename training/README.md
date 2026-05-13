# `training/` — Phase 5 fine-tuning scaffolding

Code and configuration for fine-tuning `kubelm-standard` on the
Phase 4 trajectory corpus. The scripts here are designed to run on
a rented GPU box (RunPod / Modal / Lambda); nothing in this
directory expects to run on the maintainer's local M1.

## File layout

```
training/
├── README.md               this file
├── configs/
│   └── kubelm-standard-v0.yaml    SFT config: base model, dataset paths, hyperparams
├── sft.py                  QLoRA SFT entry point (Unsloth)
└── eval_checkpoint.py      run the kubelm bench against a local checkpoint
```

## What we're training

**`kubelm-standard` v0** — a 3B specialist for K8sGPT MCP tool-use.

Base model selection rationale (see PROJECT.md decisions log
2026-05-13): `Qwen/Qwen2.5-3B-Instruct`. The 2026-05-12 Shape B
benchmark showed `qwen2.5:7b` at 24/30 conclusion rubric and 14
grounding failures — competitive with `gpt-4o` on grounding (12)
at 4.7 GB. The 3B target is the next size down in the same family
with the same instruction-tuning regime. The training hypothesis
becomes: "can `Qwen2.5-3B-Instruct` + kubelm SFT match
`Qwen2.5-7B-Instruct` on this MCP surface?" Same base family
controls for general-capability variance; the only thing changing
is specialization.

Llama 3.2 3B was the alternative but its 2026-05-12 baseline was
catastrophic (1/30 complete, 6/30 rubric, 0/30 ref_pass). The
capability gap to close is larger than what SFT on 365 trajectories
is likely to bridge. Phi-3.5 mini is untested on this surface; can
revisit for v0.2.

## Quality bar for release

Per ROADMAP Phase 5:

- Tool-name hallucination rate ≤ base model
- Argument hallucination rate ≤ base model
- Conclusion rubric pass rate ≥ base model
- Grounding failure rate ≤ base model (caveat: v1 grounding metric
  is brittle; treat the number as directional until grounding-v2)

If kubelm-standard v0 doesn't measurably beat the
`Qwen2.5-3B-Instruct` base on at least the first three, don't
release — iterate the data.

## Cost model

Per ROADMAP: under $10 per training run on a single A100. 365
trajectories × ~2-3 epochs × QLoRA (low memory footprint) fits
comfortably in 4-6 hours of A100 time. RunPod community-cloud A100
is ~$0.79/hr at time of writing; Modal A100 is ~$1.60/hr.

## How to run a cycle

### 1. Prep dataset on the GPU box

```bash
# Clone the repo (or pull the latest)
git clone https://github.com/rbentaarit/kubelm.git
cd kubelm

# Install training deps
uv sync --group train
```

(The `train` dep group adds `unsloth`, `transformers`, `trl`,
`datasets`, `torch` — none of which install on macOS Apple Silicon
through `uv` cleanly, hence the GPU-box-only flow.)

### 2. SFT

```bash
uv run python training/sft.py \
    --config training/configs/kubelm-standard-v0.yaml \
    --out runs/kubelm-standard-v0-attempt-1/
```

Outputs:
  - `runs/<name>/adapter/` — the LoRA adapter (~50 MB)
  - `runs/<name>/merged/` — the merged FP16 weights (~6 GB)
  - `runs/<name>/training_log.jsonl` — per-step loss + lr + grad norm

### 3. Evaluate

```bash
# Boot the model on an inference server first (vLLM or llama.cpp)
# so the kubelm eval harness can hit it like any other OpenAI-compatible backend.

uv run python training/eval_checkpoint.py \
    --checkpoint runs/kubelm-standard-v0-attempt-1/merged/ \
    --bench-models eval/scenarios/benchmarks/shape-b.yaml \
    --scenario-set eval/scenarios/specs/ \
    --out eval/results/checkpoints/kubelm-standard-v0-attempt-1/
```

This runs the same 30-scenario bench used for the Phase 3 baseline,
so the result is directly comparable to the published rows.

### 4. Quantize (optional, for CPU release)

```bash
# Requires llama.cpp tooling on PATH
python -m llama_cpp.convert_hf_to_gguf runs/<name>/merged/ \
    --outfile runs/<name>/kubelm-standard.gguf
python -m llama_cpp.quantize \
    runs/<name>/kubelm-standard.gguf \
    runs/<name>/kubelm-standard.Q4_K_M.gguf \
    Q4_K_M
```

### 5. Release

If the bench numbers clear the quality bar above, push to Hugging Face:

  - LoRA adapter: `rbentaarit/kubelm-standard-v0-lora`
  - Merged FP16: `rbentaarit/kubelm-standard-v0`
  - GGUF Q4_K_M: `rbentaarit/kubelm-standard-v0-GGUF`

The release upload is a maintainer action; not scripted.

## Followups tracked in ROADMAP Phase 5

- Hyperparameter sweep (5–10 runs at varying lr / lora rank / epochs)
- Best-checkpoint-by-eval selection
- Model card on Hugging Face
- Blog post on the fine-tune

## Why Unsloth specifically

- 2× faster than vanilla TRL SFT on small models with QLoRA — relevant
  for the cost budget.
- Native GGUF quantization output, so step 4 above can be skipped if
  Unsloth's quantize-to-GGUF flag is set during training.
- Maintained, widely used, no exotic dependencies beyond PyTorch.

If Unsloth becomes a problem, vanilla TRL with `bitsandbytes` is the
drop-in alternative; sft.py's structure is portable.
