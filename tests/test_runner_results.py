from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.client import Tool
from eval.runner.results import RESULTS_SCHEMA_VERSION, emit_results
from eval.trajectory import ToolCall, ToolResult, TrajectoryRecorder


def _tool(name: str, schema: dict[str, Any] | None = None) -> Tool:
    return Tool(
        name=name,
        description=f"description of {name}",
        input_schema=schema or {"type": "object", "properties": {}},
        annotations={},
    )


def test_emit_results_writes_well_formed_summary(tmp_path: Path) -> None:
    traj = tmp_path / "trajectory.jsonl"
    extra_meta = {
        "system_prompt": "be helpful",
        "backend": {
            "base_url": "http://localhost:11434/v1",
            "model": "llama3.2:3b",
            "temperature": 0.0,
            "max_tokens": 2048,
        },
        "max_steps": 16,
    }
    with TrajectoryRecorder(
        path=traj,
        goal="list namespaces",
        scenario_id="scn-1",
        model="llama3.2:3b",
        extra_meta=extra_meta,
    ) as rec:
        rec.assistant(
            text="",
            tool_calls=[ToolCall("c1", "list-namespaces", {})],
            latency_ms=120.0,
        )
        rec.tool_result(
            ToolResult("c1", "list-namespaces", {"items": ["default"]}),
            latency_ms=15.0,
        )
        rec.assistant(text="Only `default` exists.", latency_ms=80.0)
        rec.end("complete")

    output = tmp_path / "results.json"
    emit_results(
        trajectory_path=traj,
        tools=[_tool("list-namespaces")],
        output_path=output,
        started_at="2026-05-06T12:00:00.000+00:00",
        ended_at="2026-05-06T12:00:01.000+00:00",
    )

    data = json.loads(output.read_text())
    assert data["schema_version"] == RESULTS_SCHEMA_VERSION
    assert data["goal"] == "list namespaces"
    assert data["scenario_id"] == "scn-1"
    assert data["k8sgpt_version"]
    assert data["mcp_protocol_version"]
    assert data["backend"]["model"] == "llama3.2:3b"
    assert data["system_prompt"] == "be helpful"
    assert data["max_steps"] == 16
    assert data["trajectory_path"] == "trajectory.jsonl"

    totals = data["totals"]
    assert totals["model_calls"] == 2
    assert totals["tool_calls"] == 1
    assert totals["model_latency_ms"] == 200.0
    assert totals["tool_latency_ms"] == 15.0

    assert data["schema_report"]["total_calls"] == 1
    assert data["schema_report"]["valid_calls"] == 1
    assert data["grounding_report"]["has_grounding_failure"] is False
    assert data["termination_report"]["label"] == "complete"

    # No narrative claims in the conclusion → trivially consistent.
    tc = data["trajectory_consistency_report"]
    assert tc["total_claims"] == 0
    assert tc["consistent_claims"] == 0
    assert tc["passed"] is True
    assert tc["inconsistent_claims"] == []


def test_emit_results_captures_failure_modes(tmp_path: Path) -> None:
    traj = tmp_path / "trajectory.jsonl"
    with TrajectoryRecorder(path=traj, goal="...") as rec:
        rec.assistant(
            text="",
            tool_calls=[ToolCall("c1", "fabricated-tool", {})],
            latency_ms=100.0,
        )
        rec.tool_result(
            ToolResult("c1", "fabricated-tool", "boom", is_error=True),
            latency_ms=5.0,
        )
        rec.assistant(text="The pod imaginary-svc is failing.", latency_ms=50.0)
        rec.end("complete")

    output = tmp_path / "results.json"
    emit_results(
        trajectory_path=traj,
        tools=[_tool("list-namespaces")],
        output_path=output,
        started_at="2026-05-06T12:00:00.000+00:00",
    )

    data = json.loads(output.read_text())
    assert data["schema_report"]["name_hallucinations"] == 1
    assert data["grounding_report"]["has_grounding_failure"]
    assert data["termination_report"]["label"] == "premature"


def test_emit_results_creates_output_dir(tmp_path: Path) -> None:
    traj = tmp_path / "trajectory.jsonl"
    with TrajectoryRecorder(path=traj, goal="g") as rec:
        rec.assistant(text="hello")
        rec.end("complete")

    output = tmp_path / "nested" / "deep" / "results.json"
    emit_results(
        trajectory_path=traj,
        tools=[],
        output_path=output,
        started_at="2026-05-06T12:00:00.000+00:00",
    )
    assert output.exists()
    data = json.loads(output.read_text())
    # Zero successful tool calls + a conclusion is correctly "premature".
    assert data["termination_report"]["label"] == "premature"


def test_emit_results_omits_scenario_reports_when_none(tmp_path: Path) -> None:
    traj = tmp_path / "trajectory.jsonl"
    with TrajectoryRecorder(path=traj, goal="g") as rec:
        rec.assistant(text="ok")
        rec.end("complete")

    output = tmp_path / "results.json"
    emit_results(
        trajectory_path=traj,
        tools=[],
        output_path=output,
        started_at="2026-05-06T12:00:00.000+00:00",
    )
    data = json.loads(output.read_text())
    assert "reference_calls_report" not in data
    assert "conclusion_rubric_report" not in data


def test_emit_results_flags_narrative_inconsistency(tmp_path: Path) -> None:
    """Conclusion claims to have run the analyzer, but the trajectory has no analyze call."""
    traj = tmp_path / "trajectory.jsonl"
    with TrajectoryRecorder(path=traj, goal="diagnose pod") as rec:
        rec.assistant(
            text="",
            tool_calls=[ToolCall("c1", "list-namespaces", {})],
            latency_ms=10.0,
        )
        rec.tool_result(
            ToolResult("c1", "list-namespaces", {"items": ["default"]}),
            latency_ms=2.0,
        )
        rec.assistant(
            text="The analyzer reported a CrashLoopBackOff on pod web-1.",
            latency_ms=5.0,
        )
        rec.end("complete")

    output = tmp_path / "results.json"
    emit_results(
        trajectory_path=traj,
        tools=[_tool("list-namespaces")],
        output_path=output,
        started_at="2026-05-06T12:00:00.000+00:00",
    )

    data = json.loads(output.read_text())
    tc = data["trajectory_consistency_report"]
    assert tc["total_claims"] == 1
    assert tc["consistent_claims"] == 0
    assert tc["passed"] is False
    assert len(tc["inconsistent_claims"]) == 1
    claim = tc["inconsistent_claims"][0]
    assert claim["pattern_name"] == "analyzer_reported"
    assert "analyze" in claim["accepted_tools"]
    assert claim["actual_calls_seen"] == []


def test_emit_results_with_scenario_includes_both_reports(tmp_path: Path) -> None:
    from eval.scenarios.spec import (
        ConclusionRubric,
        ExpectedOutcome,
        ReferenceCall,
        ReferenceCalls,
        Scenario,
    )

    traj = tmp_path / "trajectory.jsonl"
    with TrajectoryRecorder(path=traj, goal="list ns", scenario_id="scn-1") as rec:
        rec.assistant(
            text="",
            tool_calls=[ToolCall("c1", "list-namespaces", {})],
            latency_ms=10.0,
        )
        rec.tool_result(
            ToolResult("c1", "list-namespaces", {"items": ["default"]}),
            latency_ms=2.0,
        )
        rec.assistant(text="The default namespace exists.", latency_ms=5.0)
        rec.end("complete")

    scenario = Scenario(
        id="scn-1",
        profile="base",
        goal="list ns",
        expected=ExpectedOutcome(
            reference_calls=ReferenceCalls(must_include=[ReferenceCall(name="list-namespaces")]),
            conclusion_rubric=ConclusionRubric(must_mention=["default"]),
        ),
    )

    output = tmp_path / "results.json"
    emit_results(
        trajectory_path=traj,
        tools=[_tool("list-namespaces")],
        output_path=output,
        started_at="2026-05-06T12:00:00.000+00:00",
        scenario=scenario,
    )
    data = json.loads(output.read_text())
    assert data["reference_calls_report"]["passed"] is True
    assert data["reference_calls_report"]["must_include_hits"] == 1
    assert data["conclusion_rubric_report"]["passed"] is True
    assert data["conclusion_rubric_report"]["missing_mentions"] == []
