#!/usr/bin/env bash
# Quantize merged FP16 -> Q4_K_M GGUF, working around RunPod's slow MFS.
#
# Why this script exists:
#   The first quantization attempt wrote the 3GB intermediate F16 GGUF
#   directly to /workspace (RunPod's network-attached MooseFS volume),
#   which dropped throughput to ~1.5 MB/s and stalled the convert.
#   /root is local NVMe and writes at hundreds of MB/s. So we:
#     1. Build llama.cpp tooling (if not already built).
#     2. Write the F16 intermediate to /root (fast local disk).
#     3. Quantize F16 -> Q4_K_M, also on /root.
#     4. Move ONLY the final ~1GB Q4_K_M to /workspace (one network write).
#     5. Delete the 3GB F16 intermediate from /root.
#
# Usage:
#   bash training/runpod_finalize.sh runs/kubelm-edge-v0-<NAME>/
#
# After this completes you scp the GGUF off the box and destroy the pod.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: bash training/runpod_finalize.sh <run-dir>" >&2
    echo "  e.g. bash training/runpod_finalize.sh runs/kubelm-edge-v0-attempt-1/" >&2
    exit 1
fi

RUN_DIR="${1%/}"
if [[ ! -d "${RUN_DIR}/merged" ]]; then
    echo "FAIL: ${RUN_DIR}/merged does not exist (run sft.py first)." >&2
    exit 1
fi

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/root/llama.cpp}"
LOCAL_F16="/root/kubelm-edge.f16.gguf"
LOCAL_Q4="/root/kubelm-edge.Q4_K_M.gguf"
FINAL_Q4="${RUN_DIR}/kubelm-edge.Q4_K_M.gguf"

# Find a Python interpreter that has torch. RunPod's PyTorch images
# don't consistently set `python3` to the interpreter with torch
# installed (some have it under /usr/local/bin/python; `python3`
# can resolve to a no-torch system 3.10). The convert script needs
# torch to read safetensors, so probe a candidate list.
PY=""
for cand in python /usr/local/bin/python python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1 \
        && "$cand" -c 'import torch' >/dev/null 2>&1; then
        PY="$cand"
        break
    fi
done
if [[ -z "$PY" ]]; then
    echo "FAIL: no Python with torch found for convert step." >&2
    echo "  Checked: python, /usr/local/bin/python, python3.{12,11,10}, python3" >&2
    exit 1
fi
echo "using Python for convert: $($PY --version) @ $(command -v $PY)"

# ---------------------------------------------------------------------------
# 1. Build llama.cpp tooling if absent
# ---------------------------------------------------------------------------

if [[ ! -x "${LLAMA_CPP_DIR}/build/bin/llama-quantize" ]]; then
    echo "=== building llama.cpp tooling ==="
    if [[ ! -d "${LLAMA_CPP_DIR}" ]]; then
        git clone --depth=1 https://github.com/ggerganov/llama.cpp "${LLAMA_CPP_DIR}"
    fi
    # --break-system-packages: RunPod's current PyTorch templates ship an
    # externally-managed system Python (PEP 668), so a bare `pip install`
    # aborts with "externally-managed-environment". The pod is ephemeral,
    # so installing the convert-script deps into system Python is fine.
    # (Hit on the 2026-05-20 v0.1 run with the torch-2.8 template.)
    pip install --quiet --break-system-packages -r "${LLAMA_CPP_DIR}/requirements/requirements-convert_hf_to_gguf.txt"
    cmake -B "${LLAMA_CPP_DIR}/build" -S "${LLAMA_CPP_DIR}" \
          -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
    cmake --build "${LLAMA_CPP_DIR}/build" --target llama-quantize -j 4
fi

# ---------------------------------------------------------------------------
# 2. HF -> GGUF F16 (write to /root, NOT /workspace)
# ---------------------------------------------------------------------------
# Reading from /workspace is fine; writing 3GB to /workspace stalls.
# The conversion is a tensor-shuffle so CPU torch is enough here even
# if the system torch was downgraded by llama.cpp's requirements file.

echo
echo "=== converting ${RUN_DIR}/merged -> /root/.f16.gguf ==="
"$PY" "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" \
    "${RUN_DIR}/merged/" \
    --outfile "${LOCAL_F16}" \
    --outtype f16

# ---------------------------------------------------------------------------
# 3. F16 -> Q4_K_M (also on /root, native CPU binary, ~30s)
# ---------------------------------------------------------------------------

echo
echo "=== quantizing F16 -> Q4_K_M ==="
"${LLAMA_CPP_DIR}/build/bin/llama-quantize" "${LOCAL_F16}" "${LOCAL_Q4}" Q4_K_M

# ---------------------------------------------------------------------------
# 4. Single network write to /workspace
# ---------------------------------------------------------------------------

echo
echo "=== moving Q4_K_M to ${FINAL_Q4} (one MFS write) ==="
mv "${LOCAL_Q4}" "${FINAL_Q4}"
rm -f "${LOCAL_F16}"

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------

echo
ls -lh "${FINAL_Q4}"

cat <<EOF

=== finalize complete ===

Pull to local M1 via SCP (run from your M1, not the pod):

  scp -i ~/.ssh/id_ed25519 -P <DIRECT-TCP-PORT> \\
      root@<POD-DIRECT-TCP-IP>:${PWD}/${FINAL_Q4} \\
      runs/kubelm-edge-v0-<NAME>/

After the transfer succeeds, destroy the pod in the RunPod web UI.

EOF
