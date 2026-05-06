"""CLI entry point: drive a model through K8sGPT's MCP tools and emit metrics.

Usage:
    uv run python -m eval.runner \\
        --goal "Why is the auth pod failing?" \\
        --backend-url http://localhost:11434/v1 \\
        --model llama3.2:3b \\
        [--api-key XXX  | $OPENAI_API_KEY] \\
        [--mcp-url http://localhost:8089/mcp] \\
        [--scenario-id pod-crashloop-001] \\
        [--output-dir eval/results] \\
        [--max-steps 16] \\
        [--temperature 0.0] \\
        [--max-tokens 2048]

Writes <output-dir>/<run-id>/{trajectory.jsonl, results.json}.
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
from eval.trajectory import TrajectoryRecorder


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m eval.runner",
        description="Run a model against K8sGPT's MCP server and record a trajectory.",
    )
    p.add_argument("--goal", required=True, help="Investigation question.")
    p.add_argument("--backend-url", required=True, help="Base URL of the OpenAI-compatible API.")
    p.add_argument("--model", required=True, help="Model identifier.")
    p.add_argument("--api-key", default=None, help="API key (falls back to $OPENAI_API_KEY).")
    p.add_argument(
        "--mcp-url",
        default=os.environ.get("KUBELM_MCP_URL", "http://localhost:8089/mcp"),
        help="K8sGPT MCP server URL.",
    )
    p.add_argument("--scenario-id", default="", help="Scenario identifier.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval/results"),
        help="Output directory; a per-run subdir is created here.",
    )
    p.add_argument("--max-steps", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=2048)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    backend = OpenAICompatBackend(
        base_url=args.backend_url,
        model=args.model,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    mcp = MCPClient(url=args.mcp_url)
    mcp.initialize()
    mcp.list_tools()

    run_id = str(uuid.uuid4())
    run_dir = args.output_dir / run_id
    traj_path = run_dir / "trajectory.jsonl"
    results_path = run_dir / "results.json"

    started_at = datetime.now(UTC).isoformat(timespec="milliseconds")
    extra_meta = {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "backend": {
            "base_url": backend.base_url,
            "model": backend.model,
            "temperature": backend.temperature,
            "max_tokens": backend.max_tokens,
        },
        "max_steps": args.max_steps,
    }

    with TrajectoryRecorder(
        path=traj_path,
        run_id=run_id,
        model=args.model,
        scenario_id=args.scenario_id,
        goal=args.goal,
        extra_meta=extra_meta,
    ) as rec:
        run_trajectory(
            goal=args.goal,
            backend=backend,
            tools=list(mcp.tools.values()),
            call_tool=mcp.call_tool,
            recorder=rec,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            max_steps=args.max_steps,
        )

    ended_at = datetime.now(UTC).isoformat(timespec="milliseconds")
    results = emit_results(
        trajectory_path=traj_path,
        tools=list(mcp.tools.values()),
        output_path=results_path,
        started_at=started_at,
        ended_at=ended_at,
    )

    schema = results["schema_report"]
    grounding = results["grounding_report"]
    totals = results["totals"]
    print(f"run_id:    {run_id}")
    print(f"results:   {results_path}")
    print(f"trajectory:{traj_path}")
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
    print(
        f"latency:   model={totals['model_latency_ms']:.0f}ms"
        f"  tools={totals['tool_latency_ms']:.0f}ms"
        f"  steps={totals['steps']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
