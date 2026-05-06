from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from eval.client import Tool
from eval.runner import AssistantTurn, MockBackend, run_trajectory
from eval.trajectory import ToolCall, TrajectoryRecorder, load_trajectory


def _tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"description of {name}",
        input_schema={"type": "object", "properties": {}},
        annotations={},
    )


def _call_turn(call_id: str, name: str, args: dict[str, Any] | None = None) -> AssistantTurn:
    return AssistantTurn(
        text="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=args or {})],
        latency_ms=10.0,
    )


def _text_turn(text: str) -> AssistantTurn:
    return AssistantTurn(text=text, latency_ms=5.0)


def _responder(responses: dict[str, Any]):
    def _call(name: str, arguments: dict[str, Any]) -> Any:
        return responses[name]

    return _call


def test_run_loop_completes_with_conclusion(tmp_path: Path) -> None:
    backend = MockBackend(
        script=[
            _call_turn("c1", "list-namespaces"),
            _text_turn("Only `default` exists."),
        ]
    )
    call_tool = _responder({"list-namespaces": {"content": [{"type": "text", "text": "default"}]}})

    with TrajectoryRecorder(path=tmp_path / "t.jsonl") as rec:
        run_trajectory(
            goal="list namespaces",
            backend=backend,
            tools=[_tool("list-namespaces")],
            call_tool=call_tool,
            recorder=rec,
        )

    events = load_trajectory(tmp_path / "t.jsonl")
    kinds = [e["kind"] for e in events]
    assert kinds == ["meta", "assistant", "tool_result", "assistant", "end"]
    assert events[-1]["status"] == "complete"
    assert events[1]["latency_ms"] == 10.0
    assert events[3]["text"] == "Only `default` exists."


def test_run_loop_hits_step_budget(tmp_path: Path) -> None:
    script = [_call_turn(f"c{i}", "list-namespaces") for i in range(20)]
    backend = MockBackend(script=script)
    call_tool = _responder({"list-namespaces": {}})

    with TrajectoryRecorder(path=tmp_path / "t.jsonl") as rec:
        run_trajectory(
            goal="...",
            backend=backend,
            tools=[_tool("list-namespaces")],
            call_tool=call_tool,
            recorder=rec,
            max_steps=5,
        )

    events = load_trajectory(tmp_path / "t.jsonl")
    assert events[-1]["status"] == "incomplete"
    assert "step budget" in events[-1]["message"]
    assistants = [e for e in events if e["kind"] == "assistant"]
    assert len(assistants) == 5


def test_run_loop_records_tool_call_exception(tmp_path: Path) -> None:
    backend = MockBackend(
        script=[
            _call_turn("c1", "get-logs", {"podName": "x", "namespace": "y"}),
            _text_turn("Logs unavailable."),
        ]
    )

    def call_tool(name: str, args: dict[str, Any]) -> Any:
        raise RuntimeError("MCP transport failure")

    with TrajectoryRecorder(path=tmp_path / "t.jsonl") as rec:
        run_trajectory(
            goal="...",
            backend=backend,
            tools=[_tool("get-logs")],
            call_tool=call_tool,
            recorder=rec,
        )

    events = load_trajectory(tmp_path / "t.jsonl")
    tool_result = next(e for e in events if e["kind"] == "tool_result")
    assert tool_result["is_error"] is True
    assert "transport failure" in tool_result["content"]


def test_run_loop_propagates_is_error_from_mcp_response(tmp_path: Path) -> None:
    backend = MockBackend(
        script=[
            _call_turn("c1", "get-logs", {"podName": "x", "namespace": "y"}),
            _text_turn("done"),
        ]
    )
    call_tool = _responder(
        {"get-logs": {"content": [{"type": "text", "text": "boom"}], "isError": True}}
    )

    with TrajectoryRecorder(path=tmp_path / "t.jsonl") as rec:
        run_trajectory(
            goal="...",
            backend=backend,
            tools=[_tool("get-logs")],
            call_tool=call_tool,
            recorder=rec,
        )

    events = load_trajectory(tmp_path / "t.jsonl")
    tool_result = next(e for e in events if e["kind"] == "tool_result")
    assert tool_result["is_error"] is True


def test_run_loop_immediate_text_conclusion(tmp_path: Path) -> None:
    backend = MockBackend(script=[_text_turn("trivial answer")])

    with TrajectoryRecorder(path=tmp_path / "t.jsonl") as rec:
        run_trajectory(
            goal="...",
            backend=backend,
            tools=[],
            call_tool=lambda *_: {},
            recorder=rec,
        )

    events = load_trajectory(tmp_path / "t.jsonl")
    assert events[-1]["status"] == "complete"
    assistants = [e for e in events if e["kind"] == "assistant"]
    assert len(assistants) == 1
    assert assistants[0]["text"] == "trivial answer"
    assert all(e["kind"] != "tool_result" for e in events)


def test_chat_history_grows_with_assistant_and_tool_messages(tmp_path: Path) -> None:
    backend = MockBackend(
        script=[
            _call_turn("c1", "list-namespaces"),
            _text_turn("done"),
        ]
    )
    call_tool = _responder({"list-namespaces": {"items": ["default"]}})

    with TrajectoryRecorder(path=tmp_path / "t.jsonl") as rec:
        run_trajectory(
            goal="hello",
            backend=backend,
            tools=[_tool("list-namespaces")],
            call_tool=call_tool,
            recorder=rec,
            system_prompt="custom-system",
        )

    first_msgs, first_tools = backend.calls[0]
    assert first_msgs[0] == {"role": "system", "content": "custom-system"}
    assert first_msgs[1] == {"role": "user", "content": "hello"}
    assert any(t.name == "list-namespaces" for t in first_tools)

    second_msgs, _ = backend.calls[1]
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in second_msgs)
    tool_msg = next(m for m in second_msgs if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == "c1"
    assert "default" in tool_msg["content"]


def test_mock_backend_exhaust_marks_recorder_errored(tmp_path: Path) -> None:
    backend = MockBackend(script=[_call_turn("c1", "list-namespaces")])
    call_tool = _responder({"list-namespaces": {}})

    with (
        pytest.raises(RuntimeError, match="exhausted"),
        TrajectoryRecorder(path=tmp_path / "t.jsonl") as rec,
    ):
        run_trajectory(
            goal="...",
            backend=backend,
            tools=[_tool("list-namespaces")],
            call_tool=call_tool,
            recorder=rec,
        )

    events = load_trajectory(tmp_path / "t.jsonl")
    assert events[-1]["kind"] == "end"
    assert events[-1]["status"] == "error"
