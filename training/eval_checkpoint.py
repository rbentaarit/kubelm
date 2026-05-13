"""Run the kubelm Phase 1 eval harness against a trained local checkpoint.

The eval harness already supports any OpenAI-compatible backend; this
script's job is to (1) help boot a checkpoint behind such an endpoint
and (2) call the bench against it the same way the published baselines
do, so post-fine-tune numbers are directly comparable to the
2026-05-12 Shape B row.

Two modes:

Mode A — assume an inference server is already running. Just point
the eval at it:

    uv run python training/eval_checkpoint.py \
        --backend-url http://localhost:8000/v1 \
        --model-name kubelm-edge-v0 \
        --scenarios-dir eval/scenarios/specs \
        --out eval/results/checkpoints/kubelm-edge-v0/

Mode B — boot a llama.cpp server (CPU-only inference, matches the
deployed kubelm runtime) and then run the eval. This requires the
`llama_cpp` Python package and a quantized GGUF on disk:

    uv run python training/eval_checkpoint.py \
        --gguf runs/kubelm-edge-v0/kubelm-edge.Q4_K_M.gguf \
        --boot-llama-cpp \
        --port 8000 \
        --model-name kubelm-edge-v0 \
        --scenarios-dir eval/scenarios/specs \
        --out eval/results/checkpoints/kubelm-edge-v0/

Either way the script writes a one-line summary JSON to
`<out>/summary.json` that matches the shape used by other Shape B
cuts in `eval/results/summaries/`, so the eval harness's existing
plot and table tooling work unchanged.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# llama-cpp-python ships several chat_format handlers. Plain `chatml`
# emits assistant turns as text — it does NOT translate Qwen 2.5's
# tool-call output into OpenAI-shape `tool_calls`, so the eval harness
# would see zero structured calls and every scenario would record
# schema_pass=False / ref_pass=False. `chatml-function-calling` is the
# standard llama-cpp-python value for ChatML-family models that need
# OpenAI-shape function calling on the wire, which is exactly the
# contract the eval harness expects.
LLAMA_CPP_CHAT_FORMAT = "chatml-function-calling"


def _wait_for_server_ready(port: int, timeout_seconds: float = 30.0) -> None:
    """Poll GET /v1/models until 200 OK or the timeout expires.

    Sleep-based ready checks race against weight load on a cold disk
    or a contended GPU box; a real poll is the only correct gate.
    """
    deadline = time.monotonic() + timeout_seconds
    url = f"http://127.0.0.1:{port}/v1/models"
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    return
        except (urllib.error.URLError, OSError) as e:
            last_err = e
        time.sleep(0.5)
    raise TimeoutError(
        f"llama_cpp.server did not answer GET {url} within {timeout_seconds}s"
        f" (last error: {last_err})"
    )


def boot_llama_cpp_server(gguf_path: Path, port: int, model_name: str) -> subprocess.Popen:
    """Spawn `python -m llama_cpp.server` on `port` and wait until it answers.

    Returns the child process. Caller is responsible for terminating.
    The python-bound server exposes an OpenAI-compatible
    `/v1/chat/completions` endpoint; with `chat_format`
    `chatml-function-calling` it translates the model's tool-call
    output into the OpenAI-shape `tool_calls` field the eval harness
    parses.
    """
    cmd = [
        "python",
        "-m",
        "llama_cpp.server",
        "--model",
        str(gguf_path),
        "--port",
        str(port),
        "--n_ctx",
        "8192",
        "--chat_format",
        LLAMA_CPP_CHAT_FORMAT,
        "--alias",
        model_name,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        _wait_for_server_ready(port)
    except TimeoutError:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if proc.poll() is not None:
            _, err = proc.communicate(timeout=2)
            raise RuntimeError(
                f"llama_cpp.server died on startup: {err.decode(errors='replace')[:500]}"
            ) from None
        raise
    if proc.poll() is not None:
        _, err = proc.communicate(timeout=2)
        raise RuntimeError(
            f"llama_cpp.server died on startup: {err.decode(errors='replace')[:500]}"
        )
    return proc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend-url", help="OpenAI-compatible endpoint to point the eval at.")
    p.add_argument(
        "--gguf",
        type=Path,
        help="Path to a quantized GGUF. With --boot-llama-cpp, this is what gets served.",
    )
    p.add_argument("--boot-llama-cpp", action="store_true")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model-name", required=True, help="Model name to record in the bench summary.")
    p.add_argument("--scenarios-dir", type=Path, default=REPO_ROOT / "eval" / "scenarios" / "specs")
    p.add_argument(
        "--profiles-dir", type=Path, default=REPO_ROOT / "eval" / "scenarios" / "profiles"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=16)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Bench-default temp=0 for reproducibility. Set to >0 only with a methodology caveat.",
    )
    args = p.parse_args()

    server_proc: subprocess.Popen | None = None
    backend_url: str

    if args.boot_llama_cpp:
        if not args.gguf:
            print("--boot-llama-cpp requires --gguf", file=sys.stderr)
            return 2
        if not args.gguf.exists():
            print(f"GGUF not found: {args.gguf}", file=sys.stderr)
            return 2
        print(f"booting llama_cpp.server on :{args.port} with {args.gguf}", file=sys.stderr)
        server_proc = boot_llama_cpp_server(args.gguf, args.port, args.model_name)
        backend_url = f"http://127.0.0.1:{args.port}/v1"
    elif args.backend_url:
        backend_url = args.backend_url
    else:
        print("Either --backend-url or --boot-llama-cpp + --gguf must be set.", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    models_file = args.out / "_models.yaml"
    models_file.write_text(
        json.dumps(
            [
                {
                    "name": args.model_name,
                    "backend_url": backend_url,
                    "model": args.model_name,
                    "temperature": args.temperature,
                    "max_tokens": 2048,
                }
            ]
        )
    )

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "eval.scenarios",
        "bench",
        "--models-file",
        str(models_file),
        "--scenarios-dir",
        str(args.scenarios_dir),
        "--profiles-dir",
        str(args.profiles_dir),
        "--output-dir",
        str(args.out),
        "--max-steps",
        str(args.max_steps),
    ]
    print("running:", " ".join(cmd), file=sys.stderr)
    try:
        proc = subprocess.run(cmd, check=False)
        rc = proc.returncode
    finally:
        if server_proc is not None:
            print("tearing down llama_cpp.server", file=sys.stderr)
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    print(f"bench exit code: {rc}", file=sys.stderr)
    print(f"results under: {args.out}", file=sys.stderr)
    print(
        "compare against the 2026-05-12 Shape B row in eval/results/summaries/README.md",
        file=sys.stderr,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
