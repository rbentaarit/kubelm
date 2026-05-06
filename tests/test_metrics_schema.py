from __future__ import annotations

from pathlib import Path

import pytest

from eval.metrics import validate_trajectory
from eval.trajectory import ToolCall, ToolResult, TrajectoryRecorder, load_trajectory


@pytest.fixture
def schemas() -> dict[str, dict]:
    return {
        "list-namespaces": {"type": "object", "properties": {}},
        "get-logs": {
            "type": "object",
            "properties": {
                "podName": {"type": "string"},
                "namespace": {"type": "string"},
                "tailLines": {"type": "integer"},
            },
            "required": ["podName", "namespace"],
        },
    }


def _record(path: Path, calls: list[ToolCall]) -> list[dict]:
    with TrajectoryRecorder(path=path) as rec:
        rec.assistant(text="", tool_calls=calls)
        for c in calls:
            rec.tool_result(ToolResult(c.id, c.name, {}))
        rec.end("complete")
    return load_trajectory(path)


def test_valid_call_no_errors(tmp_path: Path, schemas: dict) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        [ToolCall("c1", "get-logs", {"podName": "api", "namespace": "default"})],
    )
    report = validate_trajectory(events, schemas)
    assert report.total_calls == 1
    assert report.name_hallucinations == 0
    assert report.argument_hallucinations == 0
    assert report.valid_calls == 1
    assert report.calls[0].valid


def test_unknown_tool_is_name_hallucination(tmp_path: Path, schemas: dict) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        [ToolCall("c1", "list-pods", {})],  # not in catalog
    )
    report = validate_trajectory(events, schemas)
    assert report.name_hallucinations == 1
    assert report.argument_hallucinations == 0
    assert report.calls[0].name_known is False
    assert report.calls[0].schema_errors == []


def test_missing_required_is_argument_hallucination(tmp_path: Path, schemas: dict) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        [ToolCall("c1", "get-logs", {"podName": "api"})],  # missing `namespace`
    )
    report = validate_trajectory(events, schemas)
    assert report.name_hallucinations == 0
    assert report.argument_hallucinations == 1
    assert report.calls[0].name_known
    assert any("namespace" in e for e in report.calls[0].schema_errors)


def test_wrong_type_is_argument_hallucination(tmp_path: Path, schemas: dict) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        [
            ToolCall(
                "c1",
                "get-logs",
                {"podName": "api", "namespace": "default", "tailLines": "fifty"},
            )
        ],
    )
    report = validate_trajectory(events, schemas)
    assert report.argument_hallucinations == 1
    assert any("tailLines" in e for e in report.calls[0].schema_errors)


def test_multiple_tool_calls_in_one_assistant_event(tmp_path: Path, schemas: dict) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        [
            ToolCall("c1", "list-namespaces", {}),
            ToolCall("c2", "get-logs", {"podName": "api"}),  # bad args
            ToolCall("c3", "fabricated-tool", {}),  # unknown
        ],
    )
    report = validate_trajectory(events, schemas)
    assert report.total_calls == 3
    assert report.valid_calls == 1
    assert report.argument_hallucinations == 1
    assert report.name_hallucinations == 1


def test_no_tool_calls_yields_empty_report(tmp_path: Path, schemas: dict) -> None:
    with TrajectoryRecorder(path=tmp_path / "t.jsonl") as rec:
        rec.assistant(text="just thinking, no calls")
        rec.end("complete")
    events = load_trajectory(tmp_path / "t.jsonl")
    report = validate_trajectory(events, schemas)
    assert report.total_calls == 0
    assert report.valid_calls == 0
