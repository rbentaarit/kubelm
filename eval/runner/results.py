"""results.json writer.

Reads a finished trajectory file, runs all three Phase 1 metrics, and
emits a self-contained summary alongside it. results.json is the
diff-friendly artifact a benchmark dashboard scans across runs;
trajectory.jsonl is the audit log for a single run. Some fields are
duplicated between the two on purpose — each file stands alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.client import Tool
from eval.metrics import (
    ConclusionRubricReport,
    GroundingReport,
    ReferenceCallsReport,
    TerminationReport,
    TrajectorySchemaReport,
    analyze_grounding,
    classify_termination,
    evaluate_conclusion_rubric,
    evaluate_reference_calls,
    validate_trajectory,
)
from eval.scenarios.spec import Scenario
from eval.trajectory import load_trajectory

RESULTS_SCHEMA_VERSION = 1


def _totals(events: list[dict[str, Any]]) -> dict[str, Any]:
    model_calls = 0
    tool_calls = 0
    model_latency_ms = 0.0
    tool_latency_ms = 0.0
    for e in events:
        kind = e.get("kind")
        if kind == "assistant":
            model_calls += 1
            tool_calls += len(e.get("tool_calls") or [])
            if (lat := e.get("latency_ms")) is not None:
                model_latency_ms += float(lat)
        elif kind == "tool_result":
            if (lat := e.get("latency_ms")) is not None:
                tool_latency_ms += float(lat)
    end_event = next((e for e in events if e.get("kind") == "end"), {})
    steps = end_event.get("steps", model_calls + tool_calls)
    return {
        "steps": steps,
        "model_calls": model_calls,
        "tool_calls": tool_calls,
        "model_latency_ms": round(model_latency_ms, 3),
        "tool_latency_ms": round(tool_latency_ms, 3),
    }


def _schema_dict(report: TrajectorySchemaReport) -> dict[str, Any]:
    return {
        "total_calls": report.total_calls,
        "name_hallucinations": report.name_hallucinations,
        "argument_hallucinations": report.argument_hallucinations,
        "valid_calls": report.valid_calls,
        "calls": [asdict(c) for c in report.calls],
    }


def _grounding_dict(report: GroundingReport) -> dict[str, Any]:
    return {
        "conclusion_text": report.conclusion_text,
        "total_facts": report.total_facts,
        "ungrounded_facts": report.ungrounded_facts,
        "has_grounding_failure": report.has_grounding_failure,
        "facts": [asdict(f) for f in report.facts],
    }


def _termination_dict(report: TerminationReport) -> dict[str, Any]:
    return {
        "label": report.label,
        "is_failure": report.is_failure,
        "errored": report.errored,
        "no_conclusion": report.no_conclusion,
        "looping": report.looping,
        "premature": report.premature,
        "end_status": report.end_status,
        "has_conclusion": report.has_conclusion,
        "successful_tool_calls": report.successful_tool_calls,
        "error_events": report.error_events,
        "duplicate_call_groups": report.duplicate_call_groups,
    }


def _reference_calls_dict(report: ReferenceCallsReport) -> dict[str, Any]:
    return {
        "must_include_hits": report.must_include_hits,
        "must_include_misses": report.must_include_misses,
        "forbidden_hits": report.forbidden_hits,
        "passed": report.passed,
        "must_include": [asdict(m) for m in report.must_include],
        "forbidden": [asdict(m) for m in report.forbidden],
    }


def _conclusion_rubric_dict(report: ConclusionRubricReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "missing_mentions": report.missing_mentions,
        "forbidden_mentions": report.forbidden_mentions,
        "semantic_intent": report.semantic_intent,
        "conclusion_text": report.conclusion_text,
    }


def emit_results(
    *,
    trajectory_path: Path,
    tools: list[Tool],
    output_path: Path,
    started_at: str,
    ended_at: str | None = None,
    scenario: Scenario | None = None,
) -> dict[str, Any]:
    events = load_trajectory(trajectory_path)
    meta = next(e for e in events if e.get("kind") == "meta")

    schemas = {t.name: t.input_schema for t in tools}
    schema_report = validate_trajectory(events, schemas)
    grounding_report = analyze_grounding(events)
    termination_report = classify_termination(events)

    results: dict[str, Any] = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "run_id": meta.get("run_id"),
        "scenario_id": meta.get("scenario_id"),
        "goal": meta.get("goal"),
        "started_at": started_at,
        "ended_at": ended_at or datetime.now(UTC).isoformat(timespec="milliseconds"),
        "k8sgpt_version": meta.get("k8sgpt_version"),
        "mcp_protocol_version": meta.get("mcp_protocol_version"),
        "backend": meta.get("backend"),
        "system_prompt": meta.get("system_prompt"),
        "max_steps": meta.get("max_steps"),
        "trajectory_path": trajectory_path.name,
        "totals": _totals(events),
        "schema_report": _schema_dict(schema_report),
        "grounding_report": _grounding_dict(grounding_report),
        "termination_report": _termination_dict(termination_report),
    }

    if scenario is not None:
        ref_report = evaluate_reference_calls(events, scenario.expected.reference_calls)
        rubric_report = evaluate_conclusion_rubric(events, scenario.expected.conclusion_rubric)
        results["reference_calls_report"] = _reference_calls_dict(ref_report)
        results["conclusion_rubric_report"] = _conclusion_rubric_dict(rubric_report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results
