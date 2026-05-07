from __future__ import annotations

from pathlib import Path

from eval.metrics import evaluate_reference_calls
from eval.scenarios.spec import ReferenceCall, ReferenceCalls
from eval.trajectory import ToolCall, ToolResult, TrajectoryRecorder, load_trajectory


def _record(path: Path, calls: list[ToolCall]) -> list[dict]:
    with TrajectoryRecorder(path=path) as rec:
        for c in calls:
            rec.assistant(text="", tool_calls=[c])
            rec.tool_result(ToolResult(c.id, c.name, {}))
        rec.end("complete")
    return load_trajectory(path)


def test_must_include_hit(tmp_path: Path) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        [ToolCall("c1", "list-resources", {"resourceType": "pods", "namespace": "x"})],
    )
    expected = ReferenceCalls(
        must_include=[ReferenceCall(name="list-resources", args_match={"resourceType": "pods"})]
    )
    report = evaluate_reference_calls(events, expected)
    assert report.passed
    assert report.must_include_hits == 1
    assert report.must_include[0].matched
    assert report.must_include[0].matched_step is not None


def test_must_include_miss(tmp_path: Path) -> None:
    events = _record(tmp_path / "t.jsonl", [ToolCall("c1", "cluster-info", {})])
    expected = ReferenceCalls(
        must_include=[ReferenceCall(name="get-logs", args_match={"podName": "x"})]
    )
    report = evaluate_reference_calls(events, expected)
    assert not report.passed
    assert report.must_include_misses == 1


def test_forbidden_hit_makes_report_fail(tmp_path: Path) -> None:
    events = _record(tmp_path / "t.jsonl", [ToolCall("c1", "add-filters", {"filters": ["Pod"]})])
    expected = ReferenceCalls(forbidden=[ReferenceCall(name="add-filters", args_match={})])
    report = evaluate_reference_calls(events, expected)
    assert not report.passed
    assert report.forbidden_hits == 1


def test_args_match_is_subset_semantics(tmp_path: Path) -> None:
    # Recorded call has more keys than the matcher; should still match.
    events = _record(
        tmp_path / "t.jsonl",
        [
            ToolCall(
                "c1",
                "get-logs",
                {"podName": "p", "namespace": "ns", "tailLines": 100},
            )
        ],
    )
    expected = ReferenceCalls(
        must_include=[ReferenceCall(name="get-logs", args_match={"podName": "p"})]
    )
    assert evaluate_reference_calls(events, expected).passed


def test_args_match_value_mismatch_does_not_match(tmp_path: Path) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        [ToolCall("c1", "get-logs", {"podName": "wrong", "namespace": "ns"})],
    )
    expected = ReferenceCalls(
        must_include=[ReferenceCall(name="get-logs", args_match={"podName": "right"})]
    )
    assert not evaluate_reference_calls(events, expected).passed


def test_multiple_must_include_partial_pass(tmp_path: Path) -> None:
    events = _record(tmp_path / "t.jsonl", [ToolCall("c1", "list-namespaces", {})])
    expected = ReferenceCalls(
        must_include=[
            ReferenceCall(name="list-namespaces", args_match={}),
            ReferenceCall(name="list-resources", args_match={}),
        ]
    )
    report = evaluate_reference_calls(events, expected)
    assert report.must_include_hits == 1
    assert report.must_include_misses == 1
    assert not report.passed


def test_empty_expected_passes_trivially(tmp_path: Path) -> None:
    events = _record(tmp_path / "t.jsonl", [])
    assert evaluate_reference_calls(events, ReferenceCalls()).passed


def test_any_of_satisfied_when_one_matches(tmp_path: Path) -> None:
    """any_of: at least one matcher must hit (multiple valid investigation paths)."""
    events = _record(
        tmp_path / "t.jsonl",
        [ToolCall("c1", "list-events", {"involvedObjectName": "crash-pod"})],
    )
    expected = ReferenceCalls(
        any_of=[
            ReferenceCall(name="get-logs", args_match={"podName": "crash-pod"}),
            ReferenceCall(name="list-events", args_match={"involvedObjectName": "crash-pod"}),
        ]
    )
    report = evaluate_reference_calls(events, expected)
    assert report.passed
    assert report.any_of_hits == 1
    assert report.any_of_satisfied


def test_any_of_unsatisfied_when_none_match(tmp_path: Path) -> None:
    events = _record(tmp_path / "t.jsonl", [ToolCall("c1", "cluster-info", {})])
    expected = ReferenceCalls(
        any_of=[
            ReferenceCall(name="get-logs"),
            ReferenceCall(name="list-events"),
        ]
    )
    report = evaluate_reference_calls(events, expected)
    assert not report.passed
    assert report.any_of_hits == 0


def test_any_of_empty_means_no_constraint(tmp_path: Path) -> None:
    events = _record(tmp_path / "t.jsonl", [ToolCall("c1", "cluster-info", {})])
    report = evaluate_reference_calls(events, ReferenceCalls())
    assert report.any_of_satisfied  # vacuously true


def test_must_include_and_any_of_combined(tmp_path: Path) -> None:
    """must_include passes (AND), any_of passes (one of)."""
    events = _record(
        tmp_path / "t.jsonl",
        [
            ToolCall("c1", "list-resources", {"resourceType": "pods"}),
            ToolCall("c2", "list-events", {}),
        ],
    )
    expected = ReferenceCalls(
        must_include=[ReferenceCall(name="list-resources", args_match={"resourceType": "pods"})],
        any_of=[
            ReferenceCall(name="get-logs"),
            ReferenceCall(name="list-events"),
        ],
    )
    assert evaluate_reference_calls(events, expected).passed
