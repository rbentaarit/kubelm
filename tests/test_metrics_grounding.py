from __future__ import annotations

from pathlib import Path

from eval.metrics import analyze_grounding
from eval.trajectory import ToolCall, ToolResult, TrajectoryRecorder, load_trajectory


def _record(
    path: Path,
    *,
    goal: str = "",
    tool_results: list[tuple[str, object]] | None = None,
    conclusion: str = "",
) -> list[dict]:
    with TrajectoryRecorder(path=path, goal=goal) as rec:
        for name, content in tool_results or []:
            rec.assistant(text="", tool_calls=[ToolCall("c", name, {})])
            rec.tool_result(ToolResult("c", name, content))
        if conclusion:
            rec.assistant(text=conclusion)
        rec.end("complete")
    return load_trajectory(path)


def test_kebab_identifier_grounded_by_tool_result(tmp_path: Path) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        tool_results=[("list-resources", {"items": [{"name": "auth-service-7d8c9b"}]})],
        conclusion="The pod auth-service-7d8c9b is failing.",
    )
    report = analyze_grounding(events)
    assert report.has_grounding_failure is False
    assert any(f.fact == "auth-service-7d8c9b" and f.grounded for f in report.facts)
    assert report.facts[0].source and report.facts[0].source.startswith("tool_result:")


def test_identifier_grounded_by_goal(tmp_path: Path) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        goal="Investigate production-ns where things look bad.",
        tool_results=[("list-namespaces", {"items": ["default"]})],
        conclusion="Nothing wrong was found in production-ns.",
    )
    report = analyze_grounding(events)
    grounded = {f.fact: f.source for f in report.facts if f.grounded}
    assert grounded.get("production-ns") == "goal"


def test_fabricated_identifier_is_ungrounded(tmp_path: Path) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        tool_results=[("list-namespaces", {"items": ["default"]})],
        conclusion="The pod fabricated-pod-xyz is the cause.",
    )
    report = analyze_grounding(events)
    assert report.has_grounding_failure
    bad = [f for f in report.facts if not f.grounded]
    assert any(f.fact == "fabricated-pod-xyz" for f in bad)


def test_status_reason_grounded_and_ungrounded(tmp_path: Path) -> None:
    grounded_events = _record(
        tmp_path / "g.jsonl",
        tool_results=[("get-resource", {"status": {"reason": "OOMKilled", "exitCode": 137}})],
        conclusion="The container was OOMKilled.",
    )
    assert analyze_grounding(grounded_events).has_grounding_failure is False

    ungrounded_events = _record(
        tmp_path / "u.jsonl",
        tool_results=[("get-resource", {"status": {"phase": "Running"}})],
        conclusion="The container hit CrashLoopBackOff repeatedly.",
    )
    report = analyze_grounding(ungrounded_events)
    assert report.has_grounding_failure
    assert any(f.fact == "CrashLoopBackOff" and not f.grounded for f in report.facts)


def test_no_final_conclusion_returns_empty_report(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    with TrajectoryRecorder(path=path, goal="check things") as rec:
        rec.assistant(text="", tool_calls=[ToolCall("c", "list-namespaces", {})])
        rec.tool_result(ToolResult("c", "list-namespaces", {"items": ["default"]}))
        rec.end("incomplete")
    events = load_trajectory(path)
    report = analyze_grounding(events)
    assert report.conclusion_text == ""
    assert report.total_facts == 0
    assert report.has_grounding_failure is False


def test_structured_tool_result_content_is_searched_recursively(tmp_path: Path) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        tool_results=[
            (
                "list-resources",
                {"items": [{"metadata": {"name": "deeply-nested-name"}}]},
            )
        ],
        conclusion="Found deeply-nested-name in the listing.",
    )
    report = analyze_grounding(events)
    assert report.has_grounding_failure is False


def test_image_ref_grounded(tmp_path: Path) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        tool_results=[("get-resource", {"spec": {"image": "nginx:1.21.4"}})],
        conclusion="Container runs nginx:1.21.4 which has known issues.",
    )
    report = analyze_grounding(events)
    assert any(f.fact == "nginx:1.21.4" and f.grounded for f in report.facts)
    assert report.has_grounding_failure is False


def test_non_fact_hyphenation_is_filtered(tmp_path: Path) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        tool_results=[("get-resource", {"status": "killed"})],
        conclusion="The pod was out-of-memory and stored data in-memory.",
    )
    report = analyze_grounding(events)
    facts = {f.fact for f in report.facts}
    assert "out-of-memory" not in facts
    assert "in-memory" not in facts


def test_backtick_fenced_text_is_extracted(tmp_path: Path) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        tool_results=[("get-resource", {"data": "config-key=value"})],
        conclusion="The setting `config-key` is set.",
    )
    report = analyze_grounding(events)
    assert any(f.fact == "config-key" and f.grounded for f in report.facts)
