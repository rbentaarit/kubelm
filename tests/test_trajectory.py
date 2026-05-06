from __future__ import annotations

from pathlib import Path

import pytest

from eval.trajectory import (
    SCHEMA_VERSION,
    ToolCall,
    ToolResult,
    TrajectoryRecorder,
    load_trajectory,
)


def test_records_meta_steps_and_end(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with TrajectoryRecorder(path=path, model="m", scenario_id="s", goal="g") as rec:
        rec.assistant(text="thinking", tool_calls=[ToolCall("c1", "list-namespaces", {})])
        rec.tool_result(ToolResult("c1", "list-namespaces", {"items": ["default"]}))
        rec.assistant(text="default is the only namespace")
        rec.end("complete")

    events = load_trajectory(path)
    kinds = [e["kind"] for e in events]
    assert kinds == ["meta", "assistant", "tool_result", "assistant", "end"]

    meta = events[0]
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["model"] == "m"
    assert meta["goal"] == "g"
    assert meta["k8sgpt_version"]  # pinned
    assert meta["mcp_protocol_version"]

    assert events[1]["tool_calls"][0]["name"] == "list-namespaces"
    assert events[2]["content"] == {"items": ["default"]}
    assert events[-1]["status"] == "complete"
    assert events[-1]["steps"] == 3


def test_step_counter_increments_and_end_records_total(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with TrajectoryRecorder(path=path) as rec:
        for i in range(4):
            rec.assistant(text=f"step-{i}")
        rec.end("complete")

    events = load_trajectory(path)
    steps = [e["step"] for e in events if e["kind"] == "assistant"]
    assert steps == [0, 1, 2, 3]
    assert events[-1]["steps"] == 4


def test_unhandled_exception_marks_end_error(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with pytest.raises(RuntimeError, match="boom"), TrajectoryRecorder(path=path) as rec:
        rec.assistant(text="before crash")
        raise RuntimeError("boom")

    events = load_trajectory(path)
    assert events[-1]["kind"] == "end"
    assert events[-1]["status"] == "error"
    assert "boom" in events[-1]["message"]


def test_explicit_end_not_overwritten_by_exit(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with TrajectoryRecorder(path=path) as rec:
        rec.end("complete")

    events = load_trajectory(path)
    end_events = [e for e in events if e["kind"] == "end"]
    assert len(end_events) == 1
    assert end_events[0]["status"] == "complete"


def test_use_outside_context_raises(tmp_path: Path) -> None:
    rec = TrajectoryRecorder(path=tmp_path / "run.jsonl")
    with pytest.raises(RuntimeError, match="outside its context"):
        rec.assistant(text="nope")
