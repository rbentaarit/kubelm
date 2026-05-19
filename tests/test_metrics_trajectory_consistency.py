from __future__ import annotations

from pathlib import Path

from eval.metrics.trajectory_consistency import (
    TrajectoryConsistencyReport,
    analyze_trajectory_consistency,
)
from eval.trajectory import ToolCall, ToolResult, TrajectoryRecorder, load_trajectory


def _record(
    path: Path,
    *,
    goal: str = "",
    tool_calls: list[tuple[str, dict, object | None, bool]] | None = None,
    conclusion: str = "",
) -> list[dict]:
    """Build a trajectory.

    Each `tool_calls` entry is (name, args, result_content, is_error).
    If result_content is None the call still appears in the assistant
    turn but no tool_result is recorded — simulates an unanswered call.
    """
    with TrajectoryRecorder(path=path, goal=goal) as rec:
        for i, (name, args, result, is_error) in enumerate(tool_calls or []):
            cid = f"c{i}"
            rec.assistant(text="", tool_calls=[ToolCall(cid, name, args)])
            if result is not None:
                rec.tool_result(ToolResult(cid, name, result, is_error=is_error))
        if conclusion:
            rec.assistant(text=conclusion)
        rec.end("complete")
    return load_trajectory(path)


def _score(events: list[dict]) -> TrajectoryConsistencyReport:
    return analyze_trajectory_consistency(events)


# ---------------------------------------------------------------------------
# Happy path: claims match real tool calls
# ---------------------------------------------------------------------------


def test_events_show_claim_consistent_with_list_events_call(tmp_path: Path) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        tool_calls=[("list-events", {"namespace": "n"}, {"items": []}, False)],
        conclusion="The events show CrashLoopBackOff on the api-pod.",
    )
    r = _score(events)
    assert r.passed is True
    assert r.total_claims == 1
    assert r.consistent_claims == 1
    assert r.inconsistent_claims == []


def test_events_show_claim_consistent_with_analyze_call(tmp_path: Path) -> None:
    # Analyzer fan-out: a claim about events is satisfied if `analyze`
    # was called, since analyze internally consults events.
    events = _record(
        tmp_path / "t.jsonl",
        tool_calls=[("analyze", {"namespace": "n"}, {"results": []}, False)],
        conclusion="Events indicate the pod is unhealthy.",
    )
    r = _score(events)
    assert r.passed is True
    assert r.consistent_claims == 1


def test_explicit_tool_name_claim_must_match_exact_tool(tmp_path: Path) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        tool_calls=[("list-events", {}, {"items": []}, False)],
        # Model claims output came from analyze, but only list-events ran.
        conclusion="From the analyze result, the pod is in CrashLoopBackOff.",
    )
    r = _score(events)
    assert r.passed is False
    assert len(r.inconsistent_claims) == 1
    assert r.inconsistent_claims[0].accepted_tools == frozenset({"analyze"})


# ---------------------------------------------------------------------------
# Hallucination patterns the metric should catch
# ---------------------------------------------------------------------------


def test_claim_about_events_with_no_supporting_call_fails(tmp_path: Path) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        tool_calls=[("get-resource", {"name": "p"}, {"phase": "Pending"}, False)],
        # Model invents an events check that never happened.
        conclusion="The events show that the pod was OOMKilled.",
    )
    r = _score(events)
    assert r.passed is False
    assert r.total_claims == 1
    assert r.consistent_claims == 0
    [missing] = r.inconsistent_claims
    assert missing.pattern_name == "events_show"
    assert "list-events" in missing.accepted_tools


def test_analyzer_reported_claim_without_analyze_call_fails(tmp_path: Path) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        tool_calls=[("list-resources", {"resourceType": "pod"}, {"items": []}, False)],
        conclusion="The analyzer reported CreateContainerConfigError on api-pod.",
    )
    r = _score(events)
    assert r.passed is False
    [missing] = r.inconsistent_claims
    assert missing.accepted_tools == frozenset({"analyze"})


def test_errored_tool_call_does_not_satisfy_claim(tmp_path: Path) -> None:
    # The model invoked analyze, but K8sGPT returned isError: true.
    # The model didn't actually see analyzer output, so a "the analyzer
    # said" claim is still inconsistent.
    events = _record(
        tmp_path / "t.jsonl",
        tool_calls=[("analyze", {}, {"error": "401 Unauthorized"}, True)],
        conclusion="The analyzer reported missing ConfigMap.",
    )
    r = _score(events)
    assert r.passed is False
    [missing] = r.inconsistent_claims
    assert "analyze" in missing.accepted_tools


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_no_claims_no_tools_passes(tmp_path: Path) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        tool_calls=[],
        conclusion="I don't have enough evidence to diagnose this.",
    )
    r = _score(events)
    assert r.passed is True
    assert r.total_claims == 0


def test_no_conclusion_passes(tmp_path: Path) -> None:
    # If the model terminated without writing a conclusion, there's
    # nothing to be inconsistent about.
    events = _record(
        tmp_path / "t.jsonl",
        tool_calls=[("analyze", {}, {"results": []}, False)],
        conclusion="",
    )
    r = _score(events)
    assert r.passed is True
    assert r.total_claims == 0


def test_multiple_claims_partial_failure(tmp_path: Path) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        tool_calls=[("list-events", {}, {"items": []}, False)],
        conclusion=(
            "Events showed the pod is failing. The analyzer reported a CreateContainerConfigError."
            # Only list-events ran — the analyzer claim is a fabrication.
        ),
    )
    r = _score(events)
    assert r.total_claims == 2
    assert r.consistent_claims == 1
    assert r.passed is False
    inconsistent_patterns = {c.pattern_name for c in r.inconsistent_claims}
    assert inconsistent_patterns == {"analyzer_reported"}


def test_resource_examination_claim_satisfied_by_get_resource(tmp_path: Path) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        tool_calls=[("get-resource", {"name": "p"}, {"spec": {}}, False)],
        conclusion="I examined the pod spec and found the missing field.",
    )
    r = _score(events)
    assert r.passed is True
    assert r.consistent_claims == 1


def test_resource_examination_claim_satisfied_by_list_resources(tmp_path: Path) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        tool_calls=[
            ("list-resources", {"resourceType": "pod"}, {"items": []}, False),
        ],
        conclusion="I examined the pod to confirm the state.",
    )
    r = _score(events)
    assert r.passed is True
