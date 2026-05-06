"""End-to-end runner test against a real K8sGPT MCP server.

Opt-in via `pytest --integration tests/`. Skipped by default to keep
the standard test run fast and dependency-free; auto-skipped if the
k8sgpt binary isn't on PATH. Methodology #2 honored: the MCP surface is
real K8sGPT 0.4.32, only the model is mocked so harness plumbing can be
validated deterministically.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest

from eval.client import MCPClient
from eval.runner import (
    DEFAULT_SYSTEM_PROMPT,
    AssistantTurn,
    MockBackend,
    emit_results,
    run_trajectory,
)
from eval.trajectory import ToolCall, TrajectoryRecorder

pytestmark = pytest.mark.integration


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"k8sgpt MCP did not bind {port} within {timeout}s")


@pytest.fixture
def k8sgpt_mcp() -> Iterator[str]:
    if shutil.which("k8sgpt") is None:
        pytest.skip("k8sgpt binary not on PATH")
    port = _free_port()
    proc = subprocess.Popen(
        ["k8sgpt", "serve", "--mcp", "--mcp-http", "--mcp-port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_port(port)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_runner_against_real_k8sgpt(tmp_path: Path, k8sgpt_mcp: str) -> None:
    client = MCPClient(url=k8sgpt_mcp)
    client.initialize()
    client.list_tools()
    assert "list-namespaces" in client.tools

    backend = MockBackend(
        script=[
            AssistantTurn(
                text="",
                tool_calls=[ToolCall(id="c1", name="list-namespaces", arguments={})],
                latency_ms=0.0,
            ),
            AssistantTurn(
                text="The cluster has the default namespace.",
                latency_ms=0.0,
            ),
        ]
    )

    traj_path = tmp_path / "trajectory.jsonl"
    extra_meta = {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "backend": {
            "base_url": "mock://",
            "model": "mock",
            "temperature": 0.0,
            "max_tokens": 2048,
        },
        "max_steps": 16,
    }
    with TrajectoryRecorder(
        path=traj_path,
        model="mock",
        scenario_id="integration-smoke",
        goal="What namespaces exist?",
        extra_meta=extra_meta,
    ) as rec:
        run_trajectory(
            goal="What namespaces exist?",
            backend=backend,
            tools=list(client.tools.values()),
            call_tool=client.call_tool,
            recorder=rec,
        )

    results_path = tmp_path / "results.json"
    results = emit_results(
        trajectory_path=traj_path,
        tools=list(client.tools.values()),
        output_path=results_path,
        started_at="2026-05-06T00:00:00.000+00:00",
    )

    assert results["schema_report"]["total_calls"] == 1
    assert results["schema_report"]["valid_calls"] == 1
    assert results["schema_report"]["name_hallucinations"] == 0
    assert results["schema_report"]["argument_hallucinations"] == 0
    assert results["totals"]["model_calls"] == 2
    assert results["totals"]["tool_calls"] == 1
    assert results["termination_report"]["label"] == "complete"
    assert results["k8sgpt_version"] == "0.4.32"
    assert results["mcp_protocol_version"] == "2025-03-26"
