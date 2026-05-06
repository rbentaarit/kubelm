from __future__ import annotations

from pathlib import Path

from eval.metrics import classify_termination
from eval.trajectory import ToolCall, ToolResult, TrajectoryRecorder, load_trajectory


def _events(path: Path, build) -> list[dict]:
    with TrajectoryRecorder(path=path) as rec:
        build(rec)
    return load_trajectory(path)


def test_complete_trajectory(tmp_path: Path) -> None:
    def build(rec):
        rec.assistant(text="", tool_calls=[ToolCall("c1", "list-namespaces", {})])
        rec.tool_result(ToolResult("c1", "list-namespaces", {"items": ["default"]}))
        rec.assistant(text="The only namespace is `default`.")
        rec.end("complete")

    report = classify_termination(_events(tmp_path / "t.jsonl", build))
    assert report.label == "complete"
    assert report.has_conclusion
    assert report.successful_tool_calls == 1
    assert report.is_failure is False


def test_last_assistant_with_tool_calls_is_no_conclusion(tmp_path: Path) -> None:
    def build(rec):
        rec.assistant(text="let me check", tool_calls=[ToolCall("c1", "list-namespaces", {})])
        rec.tool_result(ToolResult("c1", "list-namespaces", {}))
        rec.assistant(text="and again", tool_calls=[ToolCall("c2", "list-namespaces", {})])
        rec.end("incomplete")

    report = classify_termination(_events(tmp_path / "t.jsonl", build))
    assert report.label == "no_conclusion"
    assert report.has_conclusion is False


def test_empty_text_last_assistant_is_no_conclusion(tmp_path: Path) -> None:
    def build(rec):
        rec.assistant(text="", tool_calls=[ToolCall("c1", "list-namespaces", {})])
        rec.tool_result(ToolResult("c1", "list-namespaces", {}))
        rec.assistant(text="")
        rec.end("incomplete")

    report = classify_termination(_events(tmp_path / "t.jsonl", build))
    assert report.label == "no_conclusion"


def test_error_event_marks_errored(tmp_path: Path) -> None:
    def build(rec):
        rec.assistant(text="", tool_calls=[ToolCall("c1", "list-namespaces", {})])
        rec.error("transport", "connection reset")
        rec.assistant(text="Best guess: it's the network.")
        rec.end("complete")

    report = classify_termination(_events(tmp_path / "t.jsonl", build))
    assert report.label == "errored"
    assert report.error_events == 1


def test_end_status_error_marks_errored(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    with TrajectoryRecorder(path=path) as rec:
        rec.assistant(text="A conclusion of sorts.")
        rec.end("error", message="harness died")
    report = classify_termination(load_trajectory(path))
    assert report.label == "errored"


def test_repeated_call_three_times_is_looping(tmp_path: Path) -> None:
    def build(rec):
        for i in range(3):
            rec.assistant(text="", tool_calls=[ToolCall(f"c{i}", "list-namespaces", {})])
            rec.tool_result(ToolResult(f"c{i}", "list-namespaces", {"items": ["default"]}))
        rec.assistant(text="The only namespace is default.")
        rec.end("complete")

    report = classify_termination(_events(tmp_path / "t.jsonl", build))
    assert report.label == "looping"
    assert len(report.duplicate_call_groups) == 1
    assert len(report.duplicate_call_groups[0]) == 3


def test_repeated_call_twice_is_not_looping(tmp_path: Path) -> None:
    def build(rec):
        for i in range(2):
            rec.assistant(text="", tool_calls=[ToolCall(f"c{i}", "list-namespaces", {})])
            rec.tool_result(ToolResult(f"c{i}", "list-namespaces", {}))
        rec.assistant(text="Conclusion based on two looks.")
        rec.end("complete")

    report = classify_termination(_events(tmp_path / "t.jsonl", build))
    assert report.looping is False
    assert report.label == "complete"


def test_different_arguments_are_not_a_loop(tmp_path: Path) -> None:
    def build(rec):
        for ns in ("default", "kube-system", "production"):
            rec.assistant(text="", tool_calls=[ToolCall("c", "get-resource", {"namespace": ns})])
            rec.tool_result(ToolResult("c", "get-resource", {}))
        rec.assistant(text="Investigated three namespaces.")
        rec.end("complete")

    report = classify_termination(_events(tmp_path / "t.jsonl", build))
    assert report.looping is False


def test_zero_tool_calls_with_conclusion_is_premature(tmp_path: Path) -> None:
    def build(rec):
        rec.assistant(text="The answer is 42.")
        rec.end("complete")

    report = classify_termination(_events(tmp_path / "t.jsonl", build))
    assert report.label == "premature"
    assert report.successful_tool_calls == 0


def test_only_error_tool_results_is_premature(tmp_path: Path) -> None:
    def build(rec):
        rec.assistant(
            text="", tool_calls=[ToolCall("c1", "get-logs", {"podName": "x", "namespace": "y"})]
        )
        rec.tool_result(ToolResult("c1", "get-logs", "boom", is_error=True))
        rec.assistant(text="Concluding without data.")
        rec.end("complete")

    report = classify_termination(_events(tmp_path / "t.jsonl", build))
    assert report.label == "premature"
    assert report.successful_tool_calls == 0


def test_one_successful_tool_result_not_premature(tmp_path: Path) -> None:
    def build(rec):
        rec.assistant(text="", tool_calls=[ToolCall("c1", "list-namespaces", {})])
        rec.tool_result(ToolResult("c1", "list-namespaces", {"items": ["default"]}))
        rec.assistant(text="Default is the only namespace.")
        rec.end("complete")

    report = classify_termination(_events(tmp_path / "t.jsonl", build))
    assert report.premature is False
    assert report.label == "complete"


def test_priority_errored_over_no_conclusion(tmp_path: Path) -> None:
    def build(rec):
        rec.error("transport", "boom")
        rec.end("error")

    report = classify_termination(_events(tmp_path / "t.jsonl", build))
    assert report.errored
    assert report.no_conclusion
    assert report.label == "errored"


def test_priority_no_conclusion_over_looping(tmp_path: Path) -> None:
    def build(rec):
        for i in range(3):
            rec.assistant(text="", tool_calls=[ToolCall(f"c{i}", "list-namespaces", {})])
            rec.tool_result(ToolResult(f"c{i}", "list-namespaces", {}))
        rec.end("incomplete")

    report = classify_termination(_events(tmp_path / "t.jsonl", build))
    assert report.no_conclusion
    assert report.looping
    assert report.label == "no_conclusion"


def test_priority_looping_over_premature(tmp_path: Path) -> None:
    def build(rec):
        for i in range(3):
            rec.assistant(
                text="",
                tool_calls=[ToolCall(f"c{i}", "get-logs", {"podName": "x", "namespace": "y"})],
            )
            rec.tool_result(ToolResult(f"c{i}", "get-logs", "boom", is_error=True))
        rec.assistant(text="No useful data, guessing.")
        rec.end("complete")

    report = classify_termination(_events(tmp_path / "t.jsonl", build))
    assert report.looping
    assert report.premature
    assert report.label == "looping"


def test_thresholds_are_tunable(tmp_path: Path) -> None:
    def build(rec):
        for i in range(2):
            rec.assistant(text="", tool_calls=[ToolCall(f"c{i}", "list-namespaces", {})])
            rec.tool_result(ToolResult(f"c{i}", "list-namespaces", {}))
        rec.assistant(text="Done.")
        rec.end("complete")

    events = _events(tmp_path / "t.jsonl", build)
    assert classify_termination(events).label == "complete"
    assert classify_termination(events, duplicate_threshold=2).label == "looping"
