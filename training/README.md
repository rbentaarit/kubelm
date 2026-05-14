# `training/` — Phase 5 fine-tuning scaffolding

Code and configuration for fine-tuning `kubelm-edge` on the Phase 4
trajectory corpus. The scripts here are designed to run on a rented
GPU box (RunPod / Modal / Lambda); nothing in this directory expects
to run on the maintainer's local M1.

## File layout

```
training/
├── README.md                       this file
├── runpod_setup.sh                 one-shot GPU-box bootstrap: clone, install, HF login, smoke-test
├── runpod_finalize.sh              post-train quantize + MFS-aware F16 -> Q4_K_M
├── configs/
│   └── kubelm-edge-v0.yaml         SFT config: base model, dataset paths, hyperparams
├── sft.py                          QLoRA SFT entry point (Unsloth)
└── eval_checkpoint.py              run the kubelm bench against a local checkpoint (llama-cpp path)
```

## What we're training

**`kubelm-edge` v0** — a 1.5B specialist for K8sGPT MCP tool-use,
sized for standalone / dev-cluster deployment.

Base model selection rationale (see PROJECT.md decisions log
2026-05-13): `Qwen/Qwen2.5-1.5B-Instruct`. Three findings drove
this choice:

1. **Qwen 2.5 7B is the empirical target.** The 2026-05-12 Shape B
   benchmark showed `qwen2.5:7b` at 24/30 rubric, 14 grounding
   failures — competitive with `gpt-4o` (12 grounding failures) at
   4.7 GB. That's the row a kubelm fine-tune is trying to match
   at smaller scale.

2. **Qwen 2.5 1.5B has a real baseline.** A standalone 2026-05-13
   bench (`eval/results/summaries/shape-b-2026-05-13-qwen-1.5b.json`)
   measured the 1.5B model out of the box: 8/30 complete, 10/30
   rubric, 3/30 ref_pass, 0 name hallucinations, 2 arg
   hallucinations. The model is not catastrophically broken
   (unlike `llama3.2:3b`, which terminated 1/30 times); it has a
   real foothold for SFT to build on.

3. **The HF survey for K8s-specialized small models came up empty
   for our surface.** Candidates at <2B params are smaller bases
   (0.5B or 1.1B), trained for different surfaces (kubectl Q&A,
   K8s command generation), and have minimal adoption. Our
   365-trajectory K8sGPT-MCP-specific corpus is more on-target
   than what any of those models were trained on. The Qwen
   family-control argument wins.

Llama 3.2 3B / 1B was the alternative but the 2026-05-12 baseline
was catastrophic (1/30 complete, 6/30 rubric, 0/30 ref_pass). The
capability gap to close from there is larger than 365 trajectories
of SFT can plausibly bridge. Phi-3.5 mini is untested on this
surface and is too large for the edge tier anyway. Revisit both
for v0.2.

## Deployment footprint (kubelm-edge tier)

The point of going to 1.5B is fitting on the same node as the
K8sGPT analyzer in a standalone or dev cluster, without dedicated
AI infrastructure.

**Disk** (llama.cpp serving):
- Q4_K_M GGUF: ~1.0 GB
- Q5_K_M (higher quality): ~1.2 GB
- Q8_0 (near-lossless): ~1.7 GB

**RAM at runtime** (weights + KV cache + compute buffer + server overhead):

| context | weights | KV cache (FP16) | total working set |
|---|---|---|---|
| 8K (default) | ~1.0 GB | ~230 MB | **~1.5–1.7 GB** |
| 4K | ~1.0 GB | ~115 MB | ~1.3–1.5 GB |
| 2K | ~1.0 GB | ~58 MB | ~1.2–1.4 GB |

KV cache per token for Qwen 2.5 1.5B: 28 layers × 2 KV heads × 128
head_dim × 2 (K+V) × 2 bytes (FP16) ≈ 28 KB.

**K8s Pod resource recommendations:**

```yaml
resources:
  requests:
    cpu: "1"        # 1 core minimum; investigation latency suffers below this
    memory: "1.5Gi" # fits 4K context working set
  limits:
    cpu: "2"        # burst to 2 cores during inference
    memory: "2Gi"   # headroom for 8K context when a tool result is large
```

Matches the ROADMAP `kubelm-edge` tier definition ("2-core CPU,
2GB RAM").

**Per-step latency targets**:
- 4-core ARM/x86 (M-series, modern Xeon): ~30-50 tokens/sec →
  ~10-20s per assistant turn
- 2-core: ~15-25 tokens/sec → ~20-40s per turn
- Typical investigation 2-5 turns: ~30s–3min per scenario at the
  edge tier

K8sGPT MCP tool results can be large (10 KB+ JSON dumps from
`list-resources`). At 4K context with a chatty scenario you can
truncate. 8K context (the config default) handles every scenario
in the v0 corpus without truncation. Deploy at 8K unless RAM is
tight.

## Quality bar for release

Per ROADMAP Phase 5:

- Tool-name hallucination rate ≤ base model
- Argument hallucination rate ≤ base model
- Conclusion rubric pass rate ≥ base model
- Grounding failure rate ≤ base model (caveat: v1 grounding metric
  is brittle; treat the number as directional until grounding-v2)

Concrete targets given the qwen2.5:1.5b baseline of (rubric 10/30,
complete 8/30, ref_pass 3/30, ground_fail 16):

- **Minimum bar (release):** rubric ≥ 12, complete ≥ 12,
  ref_pass ≥ 6, name_halluc 0, arg_halluc ≤ 2 (i.e., a clear
  improvement on every column the base wasn't already saturated
  on)
- **Stretch target:** rubric ≥ 17, complete ≥ 20, ref_pass ≥ 12
  — bringing the 1.5B into the same neighborhood as gpt-5.4's
  rubric performance, at a 4-5× smaller footprint
- **Optimistic target:** match qwen2.5:7b (rubric 24, complete
  30, ref_pass 29) — "specialization fully recovered the
  capability lost going 7B→1.5B in the same family"

If kubelm-edge v0 doesn't clear the minimum bar, don't release.
Iterate the data.

## Cost model

A 1.5B QLoRA SFT on ~319 trajectories × 3 epochs fits in **2-4
hours of A100 time**:

- RunPod community-cloud A100: $0.79/hr → **$1.50–$3.00/run**
- Modal A100: $1.60/hr → $3.00–$6.50/run

Roughly 2× cheaper than the 3B-target plan from the previous
config iteration, because the 1.5B trains faster per step and
has a smaller activation footprint.

## How to run a cycle

### 1. Bootstrap the GPU box (one command, two token prompts)

From a fresh RunPod PyTorch box's web terminal:

```bash
curl -sL https://raw.githubusercontent.com/rbentaarit/kubelm/main/training/runpod_setup.sh | bash
```

The script prompts (silently) for a GitHub PAT and a Hugging Face
read token, then runs end-to-end with no further input: clones the
repo, installs uv, pins torch to the template's exact version
(skips ~3 GB of duplicate nvidia-cu12* downloads), creates the venv
inheriting the template's CUDA stack, layers the matching Unsloth
CUDA extra, logs into HF, pre-caches `Qwen/Qwen2.5-1.5B-Instruct`,
and runs the assistant-only-loss smoke-test. Total wall-time on a
warm RunPod community-cloud A100: 5-10 minutes.

The script ends with the exact training command to copy-paste,
including a timestamped output dir.

(The `train` dep group adds `unsloth`, `transformers`, `trl`,
`datasets`, `torch`. None of these install on macOS Apple Silicon
through `uv` cleanly, hence the GPU-box-only flow.)

### 2. SFT

```bash
uv run python training/sft.py \
    --config training/configs/kubelm-edge-v0.yaml \
    --out runs/kubelm-edge-v0-attempt-1/
```

Outputs:
  - `runs/<name>/adapter/` — the LoRA adapter (~30 MB on a 1.5B base)
  - `runs/<name>/merged/` — the merged FP16 weights (~3 GB)
  - `runs/<name>/training_log.jsonl` — per-step loss + lr + grad norm

### 3. Evaluate

```bash
# Boot the model on an inference server (vLLM or llama.cpp).

uv run python training/eval_checkpoint.py \
    --backend-url http://localhost:8000/v1 \
    --model-name kubelm-edge-v0 \
    --scenarios-dir eval/scenarios/specs \
    --out eval/results/checkpoints/kubelm-edge-v0-attempt-1/
```

This runs the same 30-scenario bench used for the
2026-05-12/2026-05-13 baselines, so the result is directly
comparable to the published rows.

### 4. Quantize (required for edge release)

```bash
# Requires llama.cpp tooling on PATH
python -m llama_cpp.convert_hf_to_gguf runs/<name>/merged/ \
    --outfile runs/<name>/kubelm-edge.gguf
python -m llama_cpp.quantize \
    runs/<name>/kubelm-edge.gguf \
    runs/<name>/kubelm-edge.Q4_K_M.gguf \
    Q4_K_M
```

Re-run the eval against the Q4_K_M (via `--boot-llama-cpp --gguf
...` in eval_checkpoint.py) to verify quantization didn't tank the
metrics. If the Q4_K_M version is within ~10% of the FP16 numbers,
ship it.

### 5. Release

If the bench numbers clear the quality bar above, push to Hugging Face:

  - LoRA adapter: `rbentaarit/kubelm-edge-v0-lora`
  - Merged FP16: `rbentaarit/kubelm-edge-v0`
  - GGUF Q4_K_M: `rbentaarit/kubelm-edge-v0-GGUF`

The release upload is a maintainer action; not scripted.

## Followups tracked in ROADMAP Phase 5

- Hyperparameter sweep (5–10 runs at varying lr / lora rank / epochs)
- Best-checkpoint-by-eval selection
- Model card on Hugging Face
- Blog post on the fine-tune

After kubelm-edge ships, ROADMAP Phase 7 expands to `kubelm-standard`
(3B) and `kubelm-pro` (7B) — same dataset, same recipe, different
base models. Homogeneity keeps maintenance manageable.

## Why Unsloth specifically

- 2× faster than vanilla TRL SFT on small models with QLoRA — relevant
  for the cost budget.
- Native GGUF quantization output, so step 4 above can be skipped if
  Unsloth's quantize-to-GGUF flag is set during training.
- Maintained, widely used, no exotic dependencies beyond PyTorch.

If Unsloth becomes a problem, vanilla TRL with `bitsandbytes` is the
drop-in alternative; sft.py's structure is portable.
