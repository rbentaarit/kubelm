"""CLI: run a model against one scenario end-to-end.

Usage:
    uv run python -m eval.scenarios run \\
        --scenario-id pod-crashloop-001 \\
        --backend-url http://localhost:11434/v1 \\
        --model llama3.2:3b

For every scenario the CLI brings up a fresh kind cluster (per-scenario
fresh is the determinism floor), installs the composed profile, applies
the scenario manifests, settles, spawns a per-cluster k8sgpt MCP server
on an ephemeral port, drives the model through run_trajectory, computes
all five metrics, writes trajectory.jsonl + results.json, and tears
the cluster down.

The run is sequential (parallelism=1) by design — Phase 3 introduces
parallel reliability passes plus a serial latency pass per the
parallel-vs-serial protocol in docs/blog/scenario-methodology.md.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from eval.client import MCPClient
from eval.runner import (
    DEFAULT_SYSTEM_PROMPT,
    OpenAICompatBackend,
    emit_results,
    run_trajectory,
)
from eval.scenarios import (
    Scenario,
    compose_profile,
    load_profiles,
    load_scenario,
    load_scenarios,
)
from eval.scenarios.cluster import k8sgpt_mcp_server
from eval.scenarios.runner import scenario_context
from eval.trajectory import TrajectoryRecorder

DEFAULT_SCENARIOS_DIR = Path(__file__).parent / "specs"
DEFAULT_PROFILES_DIR = Path(__file__).parent / "profiles"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m eval.scenarios",
        description="Run a model against scenarios from the kubelm scenario library.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a model against one scenario")
    target = run.add_mutually_exclusive_group(required=True)
    target.add_argument("--scenario", type=Path, help="Path to a scenario YAML.")
    target.add_argument(
        "--scenario-id",
        help="Scenario id to look up in --scenarios-dir.",
    )
    run.add_argument("--backend-url", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--api-key", default=None, help="API key (falls back to $OPENAI_API_KEY).")
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--max-tokens", type=int, default=2048)
    run.add_argument("--max-steps", type=int, default=16)
    run.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results"),
        help="Per-run subdir is created under here.",
    )
    run.add_argument("--scenarios-dir", type=Path, default=DEFAULT_SCENARIOS_DIR)
    run.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    return p.parse_args(argv)


def _resolve_scenario(args: argparse.Namespace) -> Scenario:
    if args.scenario:
        return load_scenario(args.scenario)
    library = load_scenarios(args.scenarios_dir)
    for s in library:
        if s.id == args.scenario_id:
            return s
    available = ", ".join(sorted(s.id for s in library)) or "(none)"
    raise SystemExit(f"error: no scenario with id {args.scenario_id!r}; available: {available}")


def _print_summary(results: dict, output_dir: Path) -> None:
    schema = results["schema_report"]
    grounding = results["grounding_report"]
    totals = results["totals"]
    print(f"output:    {output_dir}")
    print(f"label:     {results['termination_report']['label']}")
    print(
        f"schema:    {schema['valid_calls']}/{schema['total_calls']} valid"
        f"  name_halluc={schema['name_hallucinations']}"
        f"  arg_halluc={schema['argument_hallucinations']}"
    )
    print(
        f"grounding: {grounding['total_facts'] - grounding['ungrounded_facts']}"
        f"/{grounding['total_facts']} grounded"
    )
    if "reference_calls_report" in results:
        rc = results["reference_calls_report"]
        rc_total = rc["must_include_hits"] + rc["must_include_misses"]
        print(
            f"refcalls:  passed={rc['passed']}"
            f"  hits={rc['must_include_hits']}/{rc_total}"
            f"  forbidden_hits={rc['forbidden_hits']}"
        )
    if "conclusion_rubric_report" in results:
        cr = results["conclusion_rubric_report"]
        print(
            f"rubric:    passed={cr['passed']}"
            f"  missing={cr['missing_mentions']}"
            f"  forbidden={cr['forbidden_mentions']}"
        )
    print(
        f"latency:   model={totals['model_latency_ms']:.0f}ms"
        f"  tools={totals['tool_latency_ms']:.0f}ms"
        f"  steps={totals['steps']}"
    )


def cmd_run(args: argparse.Namespace) -> int:
    scenario = _resolve_scenario(args)
    profiles = load_profiles(args.profiles_dir)
    profile = compose_profile(scenario.profile, profiles)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    backend = OpenAICompatBackend(
        base_url=args.backend_url,
        model=args.model,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    run_id = str(uuid.uuid4())
    started_at = datetime.now(UTC).isoformat(timespec="milliseconds")

    with (
        scenario_context(
            scenario=scenario,
            profile=profile,
            run_id=run_id,
            output_root=args.output_dir,
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
            "max_steps": args.max_steps,
            "cluster_strategy": "fresh",
            "parallelism": 1,
        }

        with TrajectoryRecorder(
            path=traj_path,
            run_id=run_id,
            model=args.model,
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
                max_steps=args.max_steps,
            )

        ended_at = datetime.now(UTC).isoformat(timespec="milliseconds")
        results = emit_results(
            trajectory_path=traj_path,
            tools=list(client.tools.values()),
            output_path=ctx.output_dir / "results.json",
            started_at=started_at,
            ended_at=ended_at,
            scenario=scenario,
        )

    _print_summary(results, ctx.output_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
