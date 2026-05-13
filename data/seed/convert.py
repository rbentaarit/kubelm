"""Convert an eval bench run's per-(model, scenario) artifacts into
the kubelm training-trajectory format described in `FORMAT.md`.

Usage (single trajectory):
    uv run python data/seed/convert.py \
        --results-dir eval/results/<run_id>/<scenario_id> \
        --out data/seed/v0/<output>.jsonl

Usage (whole bench's gpt-5.4 trajectories):
    uv run python data/seed/convert.py \
        --bench eval/results/benchmarks/<bench_id>/summary.json \
        --model gpt-5.4 \
        --out data/seed/v0/gpt-5.4-2026-05-12.jsonl

The output is one JSONL line per (model, scenario), exactly matching
FORMAT.md schema_version 1. By default only trajectories whose eval
`conclusion_rubric_passed` is True are emitted; pass --include-rubric-fail
to keep failures (e.g., for inspection or as candidates for handwritten
correction).

Tool definitions (`tools` field in FORMAT.md) are not recoverable from
the eval results alone — the eval harness only validates against the
runtime MCP schemas without persisting them. If a cache exists at
`data/seed/tools/<k8sgpt_version>.json`, this script reads it; otherwise
it emits `null` for `tools` and a note in `quality.tools_cache_status`.
A separate `data/seed/snapshot_tools.py` helper generates the cache.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_LICENSE = "CC-BY-4.0"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_tools_cache(k8sgpt_version: str) -> tuple[list[dict[str, Any]] | None, str]:
    """Return (tools_list, status_note). Tools list is None if no cache."""
    path = REPO_ROOT / "data" / "seed" / "tools" / f"{k8sgpt_version}.json"
    if not path.exists():
        return None, f"missing: run data/seed/snapshot_tools.py against {k8sgpt_version}"
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            return None, f"malformed: {path} top level must be a list"
        return data, "loaded"
    except json.JSONDecodeError as exc:
        return None, f"malformed: {path} ({exc})"


def _load_trajectory_events(traj_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with traj_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _events_to_messages(
    events: list[dict[str, Any]],
    system_prompt: str,
    goal: str,
) -> list[dict[str, Any]]:
    """Project eval trajectory events into an OpenAI-shaped messages array."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": goal},
    ]
    for ev in events:
        kind = ev.get("kind")
        if kind == "assistant":
            tool_calls = ev.get("tool_calls") or []
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": ev.get("text") or "",
            }
            if tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": json.dumps(tc.get("arguments") or {}),
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(msg)
        elif kind == "tool_result":
            content = ev.get("content")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": ev.get("tool_call_id", ""),
                    "content": json.dumps(content) if not isinstance(content, str) else content,
                }
            )
    return messages


def convert_one(results_dir: Path, source_bench_id: str | None = None) -> dict[str, Any] | None:
    """Convert one (model, scenario) eval-result directory to a training trajectory.

    Returns the trajectory dict, or None if the source is malformed.
    """
    results_path = results_dir / "results.json"
    traj_path = results_dir / "trajectory.jsonl"
    if not (results_path.exists() and traj_path.exists()):
        print(
            f"  skip {results_dir.name}: missing results.json or trajectory.jsonl",
            file=sys.stderr,
        )
        return None

    results = json.loads(results_path.read_text())
    events = _load_trajectory_events(traj_path)
    meta = next((e for e in events if e.get("kind") == "meta"), {})

    k8sgpt_version = results.get("k8sgpt_version") or meta.get("k8sgpt_version") or "unknown"
    tools_list, tools_status = _load_tools_cache(k8sgpt_version)

    backend = results.get("backend") or meta.get("backend") or {}
    schema_report = results.get("schema_report") or {}
    grounding_report = results.get("grounding_report") or {}
    termination_report = results.get("termination_report") or {}
    ref_report = results.get("reference_calls_report") or {}
    rubric_report = results.get("conclusion_rubric_report") or {}
    totals = results.get("totals") or {}

    messages = _events_to_messages(
        events,
        system_prompt=results.get("system_prompt") or "",
        goal=results.get("goal") or "",
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": str(uuid.uuid4()),
        "k8sgpt_version": k8sgpt_version,
        "mcp_protocol_version": results.get("mcp_protocol_version")
        or meta.get("mcp_protocol_version")
        or "unknown",
        "scenario_id": results.get("scenario_id"),
        "scenario_source_path": f"eval/scenarios/specs/{results.get('scenario_id')}.yaml",
        "provenance": {
            "source": "eval_bench",
            "source_run_id": results.get("run_id"),
            "source_bench_id": source_bench_id,
            "generator_model": meta.get("model_name") or backend.get("model") or "unknown",
            "generator_backend": backend.get("base_url") or "unknown",
            "generated_at": results.get("started_at"),
            "license": DEFAULT_LICENSE,
            "review_status": "unreviewed",
        },
        "system_prompt": results.get("system_prompt") or "",
        "goal": results.get("goal") or "",
        "tools": tools_list,
        "messages": messages,
        "quality": {
            "termination_label": termination_report.get("label"),
            "schema_passed": schema_report.get("valid_calls") == schema_report.get("total_calls"),
            "schema_name_halluc": schema_report.get("name_hallucinations", 0),
            "schema_arg_halluc": schema_report.get("argument_hallucinations", 0),
            "reference_calls_passed": bool(ref_report.get("passed")),
            "conclusion_rubric_passed": bool(rubric_report.get("passed")),
            "grounding_failed": bool(grounding_report.get("has_grounding_failure")),
            "grounding_failed_v1_artifact": None,  # reviewer fills during REVIEW.md walkthrough
            "step_count": sum(1 for e in events if e.get("kind") == "assistant"),
            "model_latency_ms": totals.get("model_latency_ms"),
            "tools_cache_status": tools_status,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--results-dir", type=Path, help="Path to one <run_id>/<scenario_id>/ dir.")
    src.add_argument(
        "--bench",
        type=Path,
        help="Path to a bench summary.json; converts all (model, scenario) runs.",
    )
    p.add_argument("--model", help="With --bench, only emit trajectories from this model name.")
    p.add_argument(
        "--include-rubric-fail",
        action="store_true",
        help="Include trajectories whose conclusion_rubric_passed is False.",
    )
    p.add_argument("--out", type=Path, required=True, help="Output JSONL path.")
    args = p.parse_args()

    out_records: list[dict[str, Any]] = []

    if args.results_dir:
        rec = convert_one(args.results_dir)
        if rec is not None:
            out_records.append(rec)
    else:
        summary = json.loads(args.bench.read_text())
        bench_id = summary.get("bench_id")
        runs = summary.get("runs") or []
        if args.model:
            runs = [r for r in runs if r.get("model") == args.model]
        for run in runs:
            if run.get("error"):
                continue
            rp = REPO_ROOT / run["results_path"]
            results_dir = rp.parent
            rec = convert_one(results_dir, source_bench_id=bench_id)
            if rec is None:
                continue
            if not args.include_rubric_fail and not rec["quality"]["conclusion_rubric_passed"]:
                continue
            out_records.append(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, default=str) + "\n")

    print(f"Wrote {len(out_records)} trajectory record(s) to {args.out}")
    if out_records and out_records[0]["tools"] is None:
        print(
            f"NOTE: tools cache missing for k8sgpt {out_records[0]['k8sgpt_version']}.",
            "Run data/seed/snapshot_tools.py to populate it.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
