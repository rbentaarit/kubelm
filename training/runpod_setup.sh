#!/usr/bin/env bash
# Bootstrap the kubelm training stack on a rented RunPod PyTorch box.
#
# Distilled from the first real launch on 2026-05-13/14. Every block
# in here is here because it bit us, or its absence cost a paid run.
#
# Expected starting state on the pod:
#   - PyTorch template booted (torch 2.4-2.10 + CUDA pre-installed)
#   - Repo cloned, current working directory is the repo root
#     (i.e. `pyproject.toml` is next to this script's parent dir)
#   - Reachable PyPI (curl -I https://files.pythonhosted.org returns 200)
#
# Run as:  bash training/runpod_setup.sh
#
# What this does NOT do (assumed pre-script):
#   - Renting the pod, picking a template, exposing SSH
#   - Cloning the repo (we expect it already there)
#   - HF login (we print the next command at the end)
#   - Running the smoke test / training (also printed at the end)

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Verify we're somewhere sensible
# ---------------------------------------------------------------------------

if [[ ! -f pyproject.toml ]]; then
    echo "FAIL: must be run from the kubelm repo root (pyproject.toml not found)." >&2
    exit 1
fi

if ! command -v python3 >/dev/null; then
    echo "FAIL: python3 not on PATH — is this really a PyTorch template?" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Detect template's torch version + CUDA build
# ---------------------------------------------------------------------------
# We pin uv against this exact torch version so the resolver doesn't pick
# a different version's CUDA dep stack (~3GB of nvidia-cu12* wheels that
# we'd then have to download and that wouldn't match the system anyway).

echo "=== detecting template torch ==="

torch_info=$(python3 - <<'PY' 2>&1
import sys
try:
    import torch
    print(f"VERSION={torch.__version__}")
    print(f"CUDA_BUILD={torch.version.cuda}")
    print(f"CUDA_AVAILABLE={torch.cuda.is_available()}")
except Exception as exc:
    print(f"ERROR={exc}", file=sys.stderr)
    sys.exit(1)
PY
)

torch_version=$(grep '^VERSION=' <<<"$torch_info" | cut -d= -f2)
torch_cuda=$(grep '^CUDA_BUILD=' <<<"$torch_info" | cut -d= -f2)
torch_cuda_ok=$(grep '^CUDA_AVAILABLE=' <<<"$torch_info" | cut -d= -f2)

if [[ -z "$torch_version" || -z "$torch_cuda" ]]; then
    echo "FAIL: could not read torch info from system Python:" >&2
    echo "$torch_info" >&2
    exit 1
fi

if [[ "$torch_cuda_ok" != "True" ]]; then
    echo "FAIL: torch.cuda.is_available() is False." >&2
    echo "  Driver may be too old for this template's torch build." >&2
    echo "  Try a different RunPod community-cloud host, or pick a" >&2
    echo "  template whose cuXXX matches an older driver." >&2
    exit 1
fi

# Strip "+cuXXX" suffix from torch (e.g. 2.8.0+cu128 -> 2.8.0)
torch_base="${torch_version%%+*}"
torch_short="${torch_base//./}"           # 2.8.0 -> 280
cuda_short="${torch_cuda//./}"             # 12.8  -> 128

echo "  torch:    ${torch_base} (${torch_version})"
echo "  CUDA:     ${torch_cuda} (cu${cuda_short})"
echo "  GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# Unsloth 2026.5.2's wheel matrix covers torch 2.4-2.10 only.
case "$torch_base" in
    2.4.*|2.5.*|2.6.*|2.7.*|2.8.*|2.9.*|2.10.*) : ;;
    *)
        echo "FAIL: torch ${torch_base} is outside Unsloth 2026.5.2's wheel band (2.4-2.10)." >&2
        echo "  Pick a different RunPod template, or bump the unsloth pin in pyproject.toml." >&2
        exit 1
        ;;
esac

# Map (torch, cuda) -> Unsloth CUDA extra name (e.g. cu128-torch280).
unsloth_extra="cu${cuda_short}-torch${torch_short}"
echo "  unsloth:  [${unsloth_extra}]"

# ---------------------------------------------------------------------------
# 3. Install uv if missing
# ---------------------------------------------------------------------------

echo
echo "=== installing uv ==="

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck source=/dev/null
    source "$HOME/.local/bin/env"
fi
echo "  $(uv --version)"

# ---------------------------------------------------------------------------
# 4. Pin pyproject.toml's torch range to the system's exact version
# ---------------------------------------------------------------------------
# uv's resolver picks the *highest* version that satisfies our range
# (e.g. torch>=2.4,<2.11 resolves to 2.10.0 today). It then resolves
# transitive deps (~14 nvidia-cu12* packages) against that version's
# manifest. With --system-site-packages, the venv already has the
# system's torch+CUDA stack — but only if we resolve against the SAME
# torch version. Pin to the system's version and the venv reuses
# everything; skip this step and we pay ~3GB of wasted downloads that
# also time out on RunPod's shared network.

echo
echo "=== pinning torch==${torch_base} in pyproject.toml ==="

# Match any version-spec form: torch>=X, torch==X, torch<X, etc.
sed -i.bak -E "s|\"torch[><=][^\"]*\"|\"torch==${torch_base}\"|" pyproject.toml
rm -f pyproject.toml.bak
grep -E '^\s+"torch==' pyproject.toml >/dev/null || {
    echo "FAIL: pyproject.toml has no torch==X.Y.Z line after sed." >&2
    echo "  Check the file format — pin may use a different syntax." >&2
    exit 1
}

# ---------------------------------------------------------------------------
# 5. Create venv + sync train group
# ---------------------------------------------------------------------------
# --system-site-packages : let venv reuse template's torch + CUDA libs
# --no-install-package torch : don't redownload — we just pinned to system
# UV_HTTP_TIMEOUT=300 : default 30s is too tight for slow downloads
#                      (bitsandbytes is ~1.5GB; nvidia-cudnn is 670MB)
# UV_CONCURRENT_DOWNLOADS=4 : fewer parallel streams = each gets more
#                            bandwidth share on RunPod's shared network.
#                            Default (50) saturates the connection and
#                            individual transfers time out before finishing.

echo
echo "=== creating venv and syncing train deps (~3-10 min) ==="

uv venv --system-site-packages
UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 uv lock --quiet
UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 uv sync --group train --no-install-package torch

# Spot-check: venv should report the system torch + CUDA, not a fresh download.
echo
echo "=== verifying venv torch ==="
uv run python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not visible from venv"
print(f"  venv torch:    {torch.__version__}")
print(f"  cuda visible:  {torch.cuda.is_available()}")
print(f"  device count:  {torch.cuda.device_count()}")
print(f"  device 0:      {torch.cuda.get_device_name(0)}")
PY

# ---------------------------------------------------------------------------
# 6. Install Unsloth's CUDA extra (xformers + matching bitsandbytes)
# ---------------------------------------------------------------------------
# The base unsloth wheel imports without xformers (falls back to PyTorch
# native attention — runs, but slower), but Unsloth's 2x QLoRA speedup
# comes from the xformers kernels matched to (torch, CUDA).

echo
echo "=== installing Unsloth CUDA extra: ${unsloth_extra} ==="

UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 \
    uv pip install "unsloth[${unsloth_extra}]==2026.5.2"

# ---------------------------------------------------------------------------
# 7. Final smoke check
# ---------------------------------------------------------------------------
# Import every package we depend on for training. If any of these errors,
# we'd rather find out here than 4 hours into a paid training run.

echo
echo "=== smoke-checking installed stack ==="

uv run python - <<'PY'
import torch, unsloth, transformers, trl, peft, bitsandbytes, accelerate, datasets
print(f"  torch:        {torch.__version__}")
print(f"  unsloth:      {unsloth.__version__}")
print(f"  transformers: {transformers.__version__}")
print(f"  trl:          {trl.__version__}")
print(f"  peft:         {peft.__version__}")
print(f"  bitsandbytes: {bitsandbytes.__version__}")
print(f"  accelerate:   {accelerate.__version__}")
print(f"  datasets:     {datasets.__version__}")
print("  ALL_IMPORTS_OK")
PY

# ---------------------------------------------------------------------------
# 8. Next steps
# ---------------------------------------------------------------------------

cat <<'NEXT'

=== install complete ===

Next steps (in order):

  1. HF login (read-only token is enough for inference):
       read -s HF_TOKEN
       uv run hf auth login --token "$HF_TOKEN"
     (huggingface_hub >=0.31 renamed `huggingface-cli` to `hf`;
      the old binary still installs but prints a deprecation error.)

  2. Smoke-test the trainer — verifies assistant-only loss masking
     actually works BEFORE you spend GPU hours:
       uv run python training/sft.py \
           --config training/configs/kubelm-edge-v0.yaml \
           --out runs/kubelm-edge-v0-smoke/ \
           --smoke-test

     Expect: PASS, 30-95% of tokens masked. If FAIL or the decoded
     unmasked region contains tool-result JSON, abort and triage
     before the real run.

  3. Real training run (3 epochs, ~2-4 hrs on A100):
       uv run python training/sft.py \
           --config training/configs/kubelm-edge-v0.yaml \
           --out runs/kubelm-edge-v0-attempt-1/

NEXT
