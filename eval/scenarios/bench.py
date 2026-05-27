"""Benchmark orchestrator: run a list of models across a list of scenarios.

`run_bench` iterates the (model x scenario) cartesian product, calls the
single-scenario `run_one_scenario` end-to-end for each, and aggregates
the per-run reports into a `summary.json` plus per-model totals.

A failed run (cluster create error, model API timeout, settle timeout,
anything else) is captured as a RunRecord with `error` set; the bench
proceeds to the next pair instead of aborting.

The output is a self-contained `eval/results/benchmarks/<bench-id>/
summary.json` that points back at each per-run trajectory + results
directory by relative path.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from eval import K8SGPT_VERSION, MCP_PROTOCOL_VERSION
from eval.client import MCPClient
from eval.runner import (
    DEFAULT_SYSTEM_PROMPT,
    OpenAICompatBackend,
    emit_results,
    run_trajectory,
)
from eval.scenarios.cluster import k8sgpt_mcp_server
from eval.scenarios.profile import Profile, compose_profile
from eval.scenarios.runner import scenario_context
from eval.scenarios.spec import Scenario
from eval.trajectory import TrajectoryRecorder

log = logging.getLogger(__name__)

BENCH_SCHEMA_VERSION = 3


class BenchParseError(ValueError):
    """Raised when a bench models file is structurally invalid."""


@dataclass
class ModelConfig:
    name: str
    backend_url: str
    model: str
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2048
    reasoning_effort: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None

    def resolve_api_key(self) -> str | None:
        if self.api_key_env is None:
            return None
        return os.environ.get(self.api_key_env)

    def to_backend(self) -> OpenAICompatBackend:
        return OpenAICompatBackend(
            base_url=self.backend_url,
            model=self.model,
            api_key=self.resolve_api_key(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            chat_template_kwargs=self.chat_template_kwargs,
        )


@dataclass
class RunRecord:
    model: str
    scenario: str
    run_id: str
    results_path: str | None = None
    termination_label: str | None = None
    schema_passed: bool | None = None
    schema_name_halluc: int | None = None
    schema_arg_halluc: int | None = None
    grounding_failed: bool | None = None
    fabrications: int | None = None
    reference_calls_passed: bool | None = None
    conclusion_rubric_passed: bool | None = None
    trajectory_consistency_passed: bool | None = None
    narrative_inconsistencies: int | None = None
    model_latency_ms: float | None = None
    duration_seconds: float = 0.0
    error: str | None = None


def _require(data: Mapping[str, Any], key: str, ctx: str) -> Any:
    if key not in data:
        raise BenchParseError(f"{ctx}: missing required field {key!r}")
    return data[key]


def _parse_model(raw: Mapping[str, Any], ctx: str) -> ModelConfig:
    ctk = raw.get("chat_template_kwargs")
    if ctk is not None and not isinstance(ctk, dict):
        raise BenchParseError(f"{ctx}: chat_template_kwargs must be a mapping")
    return ModelConfig(
        name=_require(raw, "name", ctx),
        backend_url=_require(raw, "backend_url", ctx),
        model=_require(raw, "model", ctx),
        api_key_env=raw.get("api_key_env"),
        temperature=float(raw.get("temperature") or 0.0),
        max_tokens=int(raw.get("max_tokens") or 2048),
        reasoning_effort=raw.get("reasoning_effort"),
        chat_template_kwargs=ctk,
    )


def load_models(path: Path | str) -> list[ModelConfig]:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise BenchParseError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, list):
        raise BenchParseError(f"{path}: top level must be a YAML list of model entries")
    models = [_parse_model(m, f"{path}[{i}]") for i, m in enumerate(raw)]
    seen = set()
    for m in models:
        if m.name in seen:
            raise BenchParseError(f"{path}: duplicate model name {m.name!r}")
        seen.add(m.name)
    return models


def run_one_scenario(
    *,
    scenario: Scenario,
    profile: Profile,
    model_cfg: ModelConfig,
    run_id: str,
    output_root: Path,
    max_steps: int = 16,
    bench_id: str | None = None,
) -> dict[str, Any]:
    """Run one (scenario, model) pair end-to-end and return the results dict."""
    backend = model_cfg.to_backend()
    started_at = datetime.now(UTC).isoformat(timespec="milliseconds")

    with (
        scenario_context(
            scenario=scenario,
            profile=profile,
            run_id=run_id,
            output_root=output_root,
        ) as ctx,
        k8sgpt_mcp_server(ctx.kubeconfig_path) as mcp_url,
    ):
        client = MCPClient(url=mcp_url)
        client.initialize()
        client.list_tools()

        traj_path = ctx.output_dir / "trajectory.jsonl"
        extra_meta = {
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "backend": {
                "base_url": backend.base_url,
                "model": backend.model,
                "temperature": backend.temperature,
                "max_tokens": backend.max_tokens,
            },
            "model_name": model_cfg.name,
            "max_steps": max_steps,
            "cluster_strategy": "fresh",
            "parallelism": 1,
        }
        if bench_id is not None:
            extra_meta["bench_id"] = bench_id

        with TrajectoryRecorder(
            path=traj_path,
            run_id=run_id,
            model=model_cfg.model,
            scenario_id=scenario.id,
            goal=scenario.goal,
            extra_meta=extra_meta,
        ) as rec:
            run_trajectory(
                goal=scenario.goal,
                backend=backend,
                tools=list(client.tools.values()),
                call_tool=client.call_tool,
                recorder=rec,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                max_steps=max_steps,
            )

        ended_at = datetime.now(UTC).isoformat(timespec="milliseconds")
        return emit_results(
            trajectory_path=traj_path,
            tools=list(client.tools.values()),
            output_path=ctx.output_dir / "results.json",
            started_at=started_at,
            ended_at=ended_at,
            scenario=scenario,
        )


def _record_from_results(
    model_cfg: ModelConfig,
    scenario: Scenario,
    run_id: str,
    results: Mapping[str, Any],
    output_root: Path,
    duration: float,
) -> RunRecord:
    schema = results["schema_report"]
    grounding = results["grounding_report"]
    grounding_v2 = results.get("grounding_v2_report") or {}
    termination = results["termination_report"]
    ref_calls = results.get("reference_calls_report") or {}
    rubric = results.get("conclusion_rubric_report") or {}
    tc = results.get("trajectory_consistency_report") or {}
    inconsistent = tc.get("inconsistent_claims")
    results_path = output_root / run_id / scenario.id / "results.json"
    # Schema 3+ headline: grounding_failed means "fabrication present"
    # (the v2 metric). Pre-v2 results.json files (which lack
    # grounding_v2_report) fall back to v1's broader
    # "any-ungrounded-fact" definition. The bench summary's
    # schema_version disambiguates which interpretation is in force.
    grounding_failed = grounding_v2.get("has_fabrication")
    if grounding_failed is None:
        grounding_failed = grounding["has_grounding_failure"]
    return RunRecord(
        model=model_cfg.name,
        scenario=scenario.id,
        run_id=run_id,
        results_path=str(results_path),
        termination_label=termination["label"],
        schema_passed=schema["valid_calls"] == schema["total_calls"],
        schema_name_halluc=schema["name_hallucinations"],
        schema_arg_halluc=schema["argument_hallucinations"],
        grounding_failed=grounding_failed,
        fabrications=grounding_v2.get("fabrications"),
        reference_calls_passed=ref_calls.get("passed"),
        conclusion_rubric_passed=rubric.get("passed"),
        trajectory_consistency_passed=tc.get("passed"),
        narrative_inconsistencies=len(inconsistent) if inconsistent is not None else None,
        model_latency_ms=results["totals"]["model_latency_ms"],
        duration_seconds=duration,
    )


def _model_summaries(runs: list[RunRecord], models: list[ModelConfig]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for m in models:
        mr = [r for r in runs if r.model == m.name]
        summaries[m.name] = {
            "scenarios_attempted": len(mr),
            "scenarios_errored": sum(1 for r in mr if r.error is not None),
            "termination_complete": sum(1 for r in mr if r.termination_label == "complete"),
            "schema_passed": sum(1 for r in mr if r.schema_passed is True),
            "name_hallucinations_total": sum(r.schema_name_halluc or 0 for r in mr),
            "argument_hallucinations_total": sum(r.schema_arg_halluc or 0 for r in mr),
            "grounding_failures": sum(1 for r in mr if r.grounding_failed),
            "fabrications_total": sum(r.fabrications or 0 for r in mr),
            "reference_calls_passed": sum(1 for r in mr if r.reference_calls_passed is True),
            "conclusion_rubric_passed": sum(1 for r in mr if r.conclusion_rubric_passed is True),
            "trajectory_consistency_passed": sum(
                1 for r in mr if r.trajectory_consistency_passed is True
            ),
            "narrative_inconsistencies_total": sum(r.narrative_inconsistencies or 0 for r in mr),
            "total_model_latency_ms": round(sum(r.model_latency_ms or 0.0 for r in mr), 1),
            "total_duration_seconds": round(sum(r.duration_seconds for r in mr), 1),
        }
    return summaries


def run_bench(
    *,
    scenarios: list[Scenario],
    profiles: dict[str, Profile],
    models: list[ModelConfig],
    output_root: Path,
    bench_id: str | None = None,
    max_steps: int = 16,
    on_run_start: Any = None,
    on_run_end: Any = None,
) -> dict[str, Any]:
    """Run every (model, scenario) pair end-to-end and write summary.json."""
    bench_id = bench_id or str(uuid.uuid4())
    started_at = datetime.now(UTC).isoformat(timespec="milliseconds")
    runs: list[RunRecord] = []

    for model_cfg in models:
        for scenario in scenarios:
            if on_run_start is not None:
                on_run_start(model_cfg, scenario)
            t0 = time.monotonic()
            run_id = str(uuid.uuid4())
            try:
                profile = compose_profile(scenario.profile, profiles)
                results = run_one_scenario(
                    scenario=scenario,
                    profile=profile,
                    model_cfg=model_cfg,
                    run_id=run_id,
                    output_root=output_root,
                    max_steps=max_steps,
                    bench_id=bench_id,
                )
                rec = _record_from_results(
                    model_cfg, scenario, run_id, results, output_root, time.monotonic() - t0
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("%s x %s failed: %s", model_cfg.name, scenario.id, exc)
                rec = RunRecord(
                    model=model_cfg.name,
                    scenario=scenario.id,
                    run_id=run_id,
                    error=f"{type(exc).__name__}: {exc}",
                    duration_seconds=time.monotonic() - t0,
                )
            runs.append(rec)
            if on_run_end is not None:
                on_run_end(rec)

    ended_at = datetime.now(UTC).isoformat(timespec="milliseconds")
    summary: dict[str, Any] = {
        "schema_version": BENCH_SCHEMA_VERSION,
        "bench_id": bench_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "k8sgpt_version": K8SGPT_VERSION,
        "mcp_protocol_version": MCP_PROTOCOL_VERSION,
        "cluster_strategy": "fresh",
        "parallelism": 1,
        "models": [asdict(m) for m in models],
        "scenarios": [s.id for s in scenarios],
        "runs": [asdict(r) for r in runs],
        "model_summaries": _model_summaries(runs, models),
    }

    bench_dir = output_root / "benchmarks" / bench_id
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary


def format_summary_table(summary: Mapping[str, Any]) -> str:
    """Markdown-ish per-model totals table (printed to stdout by the CLI)."""
    headers = [
        "model",
        "complete",
        "schema_pass",
        "name_halluc",
        "arg_halluc",
        "fab_runs",
        "fabs",
        "narr_pass",
        "ref_pass",
        "rubric_pass",
        "errored",
        "duration_s",
    ]
    rows = []
    for name, s in summary["model_summaries"].items():
        attempted = s["scenarios_attempted"] or 1
        rows.append(
            [
                name,
                f"{s['termination_complete']}/{attempted}",
                f"{s['schema_passed']}/{attempted}",
                str(s["name_hallucinations_total"]),
                str(s["argument_hallucinations_total"]),
                str(s["grounding_failures"]),
                str(s.get("fabrications_total", "-")),
                f"{s.get('trajectory_consistency_passed', 0)}/{attempted}",
                f"{s['reference_calls_passed']}/{attempted}",
                f"{s['conclusion_rubric_passed']}/{attempted}",
                str(s["scenarios_errored"]),
                f"{s['total_duration_seconds']:.0f}",
            ]
        )
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    lines = [
        " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        "-+-".join("-" * w for w in widths),
    ]
    for r in rows:
        lines.append(" | ".join(r[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)
