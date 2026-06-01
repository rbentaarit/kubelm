"""kubelm-agent — a thin HTTP service that drives kubelm through K8sGPT's
MCP tools to investigate a cluster issue.

It ORCHESTRATES K8sGPT's canonical MCP tools (via ``eval.client.mcp``)
and kubelm (via ``eval.runner.openai_backend``), reusing the eval
run-loop (``eval.runner.loop.run_trajectory``). It reimplements no
analysis: K8sGPT's MCP surface stays canonical, kubelm proposes, the
operator disposes. This is the deployable form of the same loop the
eval harness runs.

  POST /investigate  {"goal": str, "max_steps": int?}
       -> {"conclusion": str, "termination": str, "steps": int,
           "tool_calls": [{"name", "arguments"}]}
  GET  /health -> {"status": "ok"}

Config via env:
  KUBELM_BACKEND_URL   kubelm OpenAI endpoint  (default http://kubelm:8080/v1)
  KUBELM_MODEL         served model name       (default kubelm-edge)
  K8SGPT_MCP_URL       K8sGPT MCP endpoint      (default http://kubelm-k8sgpt:8089/mcp)
  MAX_STEPS, REQUEST_TIMEOUT, PORT
"""

from __future__ import annotations

import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from eval.client.mcp import MCPClient
from eval.runner.loop import run_trajectory
from eval.runner.openai_backend import OpenAICompatBackend
from eval.trajectory import TrajectoryRecorder

KUBELM_BACKEND_URL = os.environ.get("KUBELM_BACKEND_URL", "http://kubelm:8080/v1")
KUBELM_MODEL = os.environ.get("KUBELM_MODEL", "kubelm-edge")
K8SGPT_MCP_URL = os.environ.get("K8SGPT_MCP_URL", "http://kubelm-k8sgpt:8089/mcp")
DEFAULT_MAX_STEPS = int(os.environ.get("MAX_STEPS", "16"))
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "600"))


def investigate(goal: str, max_steps: int) -> dict[str, Any]:
    client = MCPClient(url=K8SGPT_MCP_URL, timeout=60.0)
    client.initialize()
    tools = list(client.list_tools().values())
    backend = OpenAICompatBackend(
        base_url=KUBELM_BACKEND_URL, model=KUBELM_MODEL, timeout=REQUEST_TIMEOUT
    )
    with tempfile.TemporaryDirectory() as td:
        traj = Path(td) / "trajectory.jsonl"
        with TrajectoryRecorder(path=traj, goal=goal, model=KUBELM_MODEL) as rec:
            run_trajectory(
                goal=goal,
                backend=backend,
                tools=tools,
                call_tool=client.call_tool,
                recorder=rec,
                max_steps=max_steps,
            )
        events = [json.loads(line) for line in traj.read_text().splitlines() if line]

    assistants = [e for e in events if e.get("kind") == "assistant"]
    end = next((e for e in events if e.get("kind") == "end"), {})
    # the conclusion is the terminating assistant turn (text, no tool calls)
    conclusion = next(
        (a["text"] for a in reversed(assistants) if a.get("text") and not a.get("tool_calls")),
        assistants[-1]["text"] if assistants else "",
    )
    tool_calls = [
        {"name": tc.get("name"), "arguments": tc.get("arguments")}
        for a in assistants
        for tc in a.get("tool_calls", [])
    ]
    return {
        "conclusion": conclusion,
        "termination": end.get("status", "unknown"),
        "steps": end.get("steps"),
        "tool_calls": tool_calls,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/investigate":
            self._send(404, {"error": "not found"})
            return
        # Broad catch is intentional: this is the request boundary, and any
        # failure (MCP, backend, parse) should return 500 with the reason.
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            goal = req.get("goal")
            if not goal:
                self._send(400, {"error": "missing 'goal'"})
                return
            max_steps = int(req.get("max_steps", DEFAULT_MAX_STEPS))
            self._send(200, investigate(goal, max_steps))
        except Exception as exc:  # noqa: BLE001 — request boundary
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, *args: Any) -> None:  # quiet the default access log
        return


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(
        f"kubelm-agent on :{port} (kubelm={KUBELM_BACKEND_URL}, mcp={K8SGPT_MCP_URL})",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
