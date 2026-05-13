"""CLI: run a model (or several) against the kubelm scenario library.

Subcommands:

    run    -- one model x one scenario, end to end.
    bench  -- list of models x list of scenarios, with a summary.json
              and a per-model totals table.

Both invocations bring up a fresh kind cluster per scenario (the
determinism floor — see docs/blog/scenario-methodology.md), spawn a
per-cluster k8sgpt MCP server, drive the model through run_trajectory,
compute all five metrics, and tear the cluster down.

Examples:

    uv run python -m eval.scenarios run \\
        --scenario-id pod-crashloop-001 \\
        --backend-url http://localhost:11434/v1 \\
        --model llama3.2:3b

    uv run python -m eval.scenarios bench \\
        --models-file eval/scenarios/benchmarks/shape-a.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from eval.scenarios import (
    Scenario,
    compose_profile,
    load_profiles,
    load_scenario,
    load_scenarios,
)
from eval.scenarios.bench import (
    ModelConfig,
    format_summary_table,
    load_models,
    run_bench,
    run_one_scenario,
)
from eval.scenarios.cluster import is_local_ollama, manage_ollama

DEFAULT_SCENARIOS_DIR = Path(__file__).parent / "specs"
DEFAULT_PROFILES_DIR = Path(__file__).parent / "profiles"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m eval.scenarios",
        description="Run a model against scenarios from the kubelm scenario library.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one model against one scenario")
    target = run.add_mutually_exclusive_group(required=True)
    target.add_argument("--scenario", type=Path, help="Path to a scenario YAML.")
    target.add_argument(
        "--scenario-id",
        help="Scenario id to look up in --scenarios-dir.",
    )
    run.add_argument("--backend-url", required=True)
    run.add_argument("--model", required=True)
    run.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Env var holding the API key (default OPENAI_API_KEY; ignored if unset).",
    )
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

    bench = sub.add_parser(
        "bench", help="Run a list of models against (some of) the scenario library"
    )
    bench.add_argument("--models-file", type=Path, required=True)
    bench.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Scenario ids to include (default: all scenarios in --scenarios-dir).",
    )
    bench.add_argument("--max-steps", type=int, default=16)
    bench.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results"),
    )
    bench.add_argument("--scenarios-dir", type=Path, default=DEFAULT_SCENARIOS_DIR)
    bench.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)

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


def _filter_scenarios(library: list[Scenario], ids: list[str] | None) -> list[Scenario]:
    if ids is None:
        return library
    by_id = {s.id: s for s in library}
    missing = [i for i in ids if i not in by_id]
    if missing:
        available = ", ".join(sorted(by_id)) or "(none)"
        raise SystemExit(f"error: unknown scenario ids: {missing}; available: {available}")
    return [by_id[i] for i in ids]


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

    model_cfg = ModelConfig(
        name=args.model,
        backend_url=args.backend_url,
        model=args.model,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    run_id = str(uuid.uuid4())
    results = run_one_scenario(
        scenario=scenario,
        profile=profile,
        model_cfg=model_cfg,
        run_id=run_id,
        output_root=args.output_dir,
        max_steps=args.max_steps,
    )
    output_dir = args.output_dir / run_id / scenario.id
    _print_summary(results, output_dir)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    models = load_models(args.models_file)
    profiles = load_profiles(args.profiles_dir)
    library = load_scenarios(args.scenarios_dir)
    scenarios = _filter_scenarios(library, args.scenarios)

    total = len(models) * len(scenarios)
    print(f"bench: {len(models)} models x {len(scenarios)} scenarios = {total} runs")
    for m in models:
        print(f"  - {m.name:20s}  {m.backend_url}  ({m.model})")
    print(f"  scenarios: {[s.id for s in scenarios]}\n")

    def on_run_start(model_cfg: ModelConfig, scenario: Scenario) -> None:
        print(f"=== {model_cfg.name} x {scenario.id} ===")

    def on_run_end(rec: Any) -> None:
        if rec.error:
            print(f"  ERROR: {rec.error}")
        else:
            print(
                f"  label={rec.termination_label}"
                f"  schema_pass={rec.schema_passed}"
                f"  ref={rec.reference_calls_passed}"
                f"  rubric={rec.conclusion_rubric_passed}"
                f"  ({rec.duration_seconds:.0f}s)"
            )

    ollama_models = [m.model for m in models if is_local_ollama(m.backend_url)]

    # Only manage the ollama daemon when at least one model needs it.
    # Otherwise the bench would unconditionally try to spawn
    # `ollama serve` even on machines (e.g. rented GPU boxes) where
    # ollama is not installed, crashing the run with FileNotFoundError
    # before the first scenario.
    ollama_ctx: Any = (
        manage_ollama(models_to_unload=ollama_models) if ollama_models else contextlib.nullcontext()
    )
    with ollama_ctx:
        summary = run_bench(
            scenarios=scenarios,
            profiles=profiles,
            models=models,
            output_root=args.output_dir,
            max_steps=args.max_steps,
            on_run_start=on_run_start,
            on_run_end=on_run_end,
        )

    print(f"\nbench_id: {summary['bench_id']}")
    print(f"summary:  {args.output_dir / 'benchmarks' / summary['bench_id'] / 'summary.json'}\n")
    print(format_summary_table(summary))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "bench":
        return cmd_bench(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
