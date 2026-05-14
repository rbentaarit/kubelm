#!/usr/bin/env bash
# One-shot bootstrap of the kubelm training stack on a rented RunPod box.
#
# Designed for iteration speed: paste two tokens, walk away while it
# installs, come back to a smoke-tested venv with the base model cached
# and the next training command printed and ready to copy-paste.
#
# Captures every gotcha from the first real launch (2026-05-13/14):
# torch pin matches system to skip ~3GB of nvidia-cu12* downloads,
# UV_HTTP_TIMEOUT+CONCURRENT_DOWNLOADS tuned for RunPod's shared
# network, Unsloth CUDA extra mapped from (torch, CUDA) dynamically,
# smoke-test verifies assistant-only loss masking before paid training.
#
# Expected starting state on the pod:
#   - PyTorch template booted (torch 2.4-2.10 + CUDA pre-installed,
#     verifiable with `nvidia-smi` and `python3 -c "import torch"`)
#   - Reachable PyPI + GitHub (default for RunPod community cloud)
#
# Run either form works:
#
#   # From a fresh pod's web terminal (no clone needed yet):
#   curl -sL https://raw.githubusercontent.com/rbentaarit/kubelm/main/training/runpod_setup.sh | bash
#
#   # From inside an already-cloned repo:
#   bash training/runpod_setup.sh
#
# What this does NOT do (intentional, you decide and copy-paste):
#   - Launch the actual training run (paid, your decision)
#   - Quantize / publish artifacts (post-train; see runpod_finalize.sh)

set -euo pipefail

REPO_URL="https://github.com/rbentaarit/kubelm.git"
REPO_DIR_DEFAULT="/workspace/kubelm"
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
UNSLOTH_PIN="2026.5.2"

# ---------------------------------------------------------------------------
# 0. Token prompts (silent — never echo, never end up in shell history)
# ---------------------------------------------------------------------------

cat <<'BANNER'
=== kubelm-edge bootstrap ===

Two tokens are required:
  1. GitHub PAT — fine-grained, Contents:Read on rbentaarit/kubelm.
     Only used to clone if the repo isn't already on disk; can be a
     throwaway token, revoke after the run.
  2. Hugging Face read token — used to pull Qwen2.5-1.5B-Instruct.

Both can be provided two ways:
  - Set GH_TOKEN and HF_TOKEN env vars before running (useful for
    SSH-driven automation, CI, etc.); the script skips the prompts.
  - Leave them unset; the script prompts silently (no echo, no
    shell history) for each.

BANNER

if [[ -z "${GH_TOKEN:-}" ]]; then
    read -s -p "GitHub PAT (Contents:Read, will not echo): " GH_TOKEN; echo
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    read -s -p "Hugging Face read token: " HF_TOKEN; echo
fi
echo

if [[ -z "$GH_TOKEN" || -z "$HF_TOKEN" ]]; then
    echo "FAIL: both tokens are required. Aborting before any side effects." >&2
    exit 1
fi
export GH_TOKEN HF_TOKEN

# ---------------------------------------------------------------------------
# 1. Clone repo if not already in one
# ---------------------------------------------------------------------------

if [[ ! -f pyproject.toml ]]; then
    target_dir="${REPO_DIR_DEFAULT}"
    mkdir -p "$(dirname "$target_dir")"
    if [[ -d "$target_dir" ]]; then
        echo "=== updating existing clone at $target_dir ==="
        cd "$target_dir"
        # Refresh the credential in the remote URL in case the previous
        # PAT was rotated. Strip any embedded token before re-adding.
        existing_url=$(git remote get-url origin)
        clean_url=$(printf '%s' "$existing_url" | sed -E 's|https://[^@]*@|https://|')
        git remote set-url origin "${clean_url/https:\/\//https:\/\/${GH_TOKEN}@}"
        git fetch --quiet origin
        git checkout --quiet main
        git reset --hard --quiet origin/main
    else
        echo "=== cloning $REPO_URL -> $target_dir ==="
        git clone --quiet "https://${GH_TOKEN}@${REPO_URL#https://}" "$target_dir"
        cd "$target_dir"
    fi
    echo "  $(git log --oneline -1)"
fi

# Sanity: we should now be in a kubelm repo root.
if [[ ! -f pyproject.toml ]]; then
    echo "FAIL: pyproject.toml still missing after clone. Aborting." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Find a Python interpreter that has torch
# ---------------------------------------------------------------------------
# RunPod's PyTorch images don't consistently set `python3` to the
# interpreter that has torch installed. The cu1290-torch280-ubuntu2204
# image, for example, ships torch in /usr/local/bin/python (Python 3.12)
# but leaves `python3` pointing at /usr/bin/python3.10 which has no
# torch. The image's intent is "use `python`" but the script can't
# assume that, so probe candidates in order until one imports torch.

PY=""
for cand in python /usr/local/bin/python python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1 \
        && "$cand" -c 'import torch' >/dev/null 2>&1; then
        PY="$cand"
        break
    fi
done

if [[ -z "$PY" ]]; then
    echo "FAIL: no Python interpreter with torch installed found." >&2
    echo "  Checked: python, /usr/local/bin/python, python3.{12,11,10}, python3" >&2
    echo "  Is this really a PyTorch template?" >&2
    exit 1
fi

echo "  using Python: $($PY --version) @ $(command -v $PY)"

# ---------------------------------------------------------------------------
# 3. Detect template's torch version + CUDA build
# ---------------------------------------------------------------------------

echo
echo "=== detecting template torch ==="

torch_info=$("$PY" - <<'PY' 2>&1
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
    echo "FAIL: torch.cuda.is_available() is False — driver too old for this template." >&2
    echo "  Try a different RunPod host (re-deploy hits a fresh community-cloud node)." >&2
    exit 1
fi

torch_base="${torch_version%%+*}"
torch_short="${torch_base//./}"
cuda_short="${torch_cuda//./}"

echo "  torch:    ${torch_base} (${torch_version})"
echo "  CUDA:     ${torch_cuda} (cu${cuda_short})"
echo "  GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

case "$torch_base" in
    2.4.*|2.5.*|2.6.*|2.7.*|2.8.*|2.9.*|2.10.*) : ;;
    *)
        echo "FAIL: torch ${torch_base} is outside Unsloth ${UNSLOTH_PIN}'s wheel band (2.4-2.10)." >&2
        exit 1
        ;;
esac

# Map host CUDA to the closest Unsloth wheel extra. Unsloth ships
# extras for specific (CUDA, torch) pairs only. The published cuda
# levels for torch 2.8 are 118, 126, 128, 130 — no 129. CUDA is
# backward-compatible by design (newer driver runs older binaries),
# so we round DOWN to the closest available extra. If torch's CUDA
# build doesn't match any of these (e.g. exotic cu127), we drop the
# extra entirely and let Unsloth fall back to PyTorch native
# attention (~2× slower training but functional).
declare -A unsloth_cuda_map=(
    [118]=118 [120]=118 [121]=118 [123]=118
    [124]=124 [125]=124
    [126]=126 [127]=126
    [128]=128 [129]=128
    [130]=130 [131]=130
)
mapped_cuda="${unsloth_cuda_map[$cuda_short]:-}"
if [[ -n "$mapped_cuda" ]]; then
    unsloth_extra="cu${mapped_cuda}-torch${torch_short}"
    echo "  unsloth:  [${unsloth_extra}] (host cu${cuda_short} -> wheel cu${mapped_cuda})"
else
    unsloth_extra=""
    echo "  unsloth:  no CUDA extra for cu${cuda_short} — will install plain unsloth"
    echo "            (slower training; xformers won't be enabled)"
fi

# ---------------------------------------------------------------------------
# 4. Install uv if missing
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
# 5. Pin pyproject.toml's torch range to the system's exact version
# ---------------------------------------------------------------------------
# Without this, uv resolves torch>=2.4,<2.11 to 2.10.0 (latest in band)
# and downloads ~3GB of mismatched nvidia-cu12* deps that time out on
# RunPod's shared network. With the pin, --system-site-packages lets the
# venv reuse the template's existing torch+CUDA stack. The .bak file
# isn't kept — we don't want pyproject committed back with the local pin.

echo
echo "=== pinning torch==${torch_base} in pyproject.toml (working-tree only) ==="

sed -i.bak -E "s|\"torch[><=][^\"]*\"|\"torch==${torch_base}\"|" pyproject.toml
rm -f pyproject.toml.bak
grep -E '^\s+"torch==' pyproject.toml >/dev/null || {
    echo "FAIL: pyproject.toml has no torch==X.Y.Z line after sed." >&2
    exit 1
}

# ---------------------------------------------------------------------------
# 6. Create venv + sync train group
# ---------------------------------------------------------------------------

echo
echo "=== creating venv and syncing train deps (~3-10 min) ==="

uv venv --system-site-packages
UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 uv lock --quiet
UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 uv sync --group train --no-install-package torch

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
# 7. Install Unsloth's CUDA extra (xformers + matching bitsandbytes)
# ---------------------------------------------------------------------------

if [[ -n "$unsloth_extra" ]]; then
    echo
    echo "=== installing Unsloth CUDA extra: ${unsloth_extra} ==="
    UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 \
        uv pip install "unsloth[${unsloth_extra}]==${UNSLOTH_PIN}"
else
    echo
    echo "=== skipping Unsloth CUDA extra (no wheel for cu${cuda_short}) ==="
    echo "  unsloth base wheel is already installed via the train group;"
    echo "  it'll fall back to PyTorch native attention (works, just slower)."
fi

# ---------------------------------------------------------------------------
# 8. Smoke-check imports
# ---------------------------------------------------------------------------

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
# 9. HF login + pre-cache base model
# ---------------------------------------------------------------------------
# Doing both here means the smoke-test below doesn't pause on the
# Qwen download, and the eventual training launch doesn't either.
# huggingface_hub >=0.31 renamed `huggingface-cli` to `hf`; we use `hf`.

echo
echo "=== HF login ==="
uv run hf auth login --token "$HF_TOKEN" >/dev/null 2>&1
echo "  authenticated"

echo
echo "=== pre-caching ${BASE_MODEL} ==="
uv run python - <<PY
from huggingface_hub import snapshot_download
path = snapshot_download("${BASE_MODEL}")
print(f"  cached at: {path}")
PY

# ---------------------------------------------------------------------------
# 10. Smoke-test the trainer (verifies assistant-only loss masking)
# ---------------------------------------------------------------------------
# This loads the model, builds the dataset, instantiates the trainer,
# pulls one batch, asserts 30-99% mask ratio, and prints the decoded
# unmasked region. If anything is wrong with the SFT plumbing, we'd
# rather find out here (~3 min) than 15 min into the paid training run.

echo
echo "=== smoke-testing trainer (verifies assistant-only loss masking) ==="

if uv run python training/sft.py \
        --config training/configs/kubelm-edge-v0.yaml \
        --out runs/kubelm-edge-v0-smoke/ \
        --smoke-test; then
    smoke_status="PASS"
else
    smoke_status="FAIL — review output above before any paid training run"
fi

# ---------------------------------------------------------------------------
# 11. Final summary
# ---------------------------------------------------------------------------

cat <<EOF

=== bootstrap complete ===
  smoke-test:    ${smoke_status}
  torch:         ${torch_base}+cu${cuda_short}
  unsloth wheel: cu${cuda_short}-torch${torch_short}

Launch the training run when you're ready (paid step, ~15 min on A100):

  uv run python training/sft.py \\
      --config training/configs/kubelm-edge-v0.yaml \\
      --out runs/kubelm-edge-v0-\$(date +%Y%m%d-%H%M)/

After training, quantize + move to /workspace via:

  bash training/runpod_finalize.sh runs/kubelm-edge-v0-<TIMESTAMP>/

EOF
