"""End-to-end test: scenario_context against a real kind cluster.

Brings up `pod-crashloop-001` against a real fresh kind cluster, spawns
a real k8sgpt MCP server, and drives a scripted MockBackend through
run_trajectory. Validates that the full lifecycle works: cluster
create -> manifests apply -> settle on CrashLoopBackOff -> MCP server
up -> tool calls return real K8sGPT data -> all five metric reports
populate -> cluster torn down.

Methodology #2 honored: real K8sGPT MCP surface, only the model is
mocked. Opt-in via `pytest --integration`. Auto-skipped if any of
kind / kubectl / k8sgpt are missing from PATH.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from eval.client import MCPClient
from eval.runner import AssistantTurn, MockBackend, emit_results, run_trajectory
from eval.scenarios import compose_profile, load_profiles, load_scenario
from eval.scenarios.cluster import k8sgpt_mcp_server
from eval.scenarios.runner import scenario_context
from eval.trajectory import ToolCall, TrajectoryRecorder

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parent.parent
SPECS_DIR = REPO / "eval" / "scenarios" / "specs"
PROFILES_DIR = REPO / "eval" / "scenarios" / "profiles"


def _check_binaries() -> None:
    missing = [b for b in ("kind", "kubectl", "k8sgpt") if shutil.which(b) is None]
    if missing:
        pytest.skip(f"binaries not on PATH: {', '.join(missing)}")


def test_pod_crashloop_scenario_runs_end_to_end(tmp_path: Path) -> None:
    _check_binaries()

    scenario = load_scenario(SPECS_DIR / "pod-crashloop-001.yaml")
    profiles = load_profiles(PROFILES_DIR)
    profile = compose_profile(scenario.profile, profiles)

    ns = "scenario-pod-crashloop-001"
    backend = MockBackend(
        script=[
            AssistantTurn(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="list-resources",
                        arguments={"resourceType": "pods", "namespace": ns},
                    )
                ],
            ),
            AssistantTurn(
                tool_calls=[
                    ToolCall(
                        id="c2",
                        name="get-logs",
                        arguments={"podName": "crash-pod", "namespace": ns},
                    )
                ],
            ),
            AssistantTurn(
                text="The pod crash-pod is in CrashLoopBackOff after a startup error.",
            ),
        ]
    )

    with (
        scenario_context(
            scenario=scenario,
            profile=profile,
            run_id="testrun1",
            output_root=tmp_path,
        ) as ctx,
        k8sgpt_mcp_server(ctx.kubeconfig_path) as mcp_url,
    ):
        client = MCPClient(url=mcp_url)
        client.initialize()
        client.list_tools()
        assert "list-resources" in client.tools
        assert "get-logs" in client.tools

        traj_path = ctx.output_dir / "trajectory.jsonl"
        with TrajectoryRecorder(
            path=traj_path,
            model="mock",
            scenario_id=scenario.id,
            goal=scenario.goal,
            extra_meta={
                "backend": {"base_url": "mock://", "model": "mock"},
                "cluster_strategy": "fresh",
                "parallelism": 1,
            },
        ) as rec:
            run_trajectory(
                goal=scenario.goal,
                backend=backend,
                tools=list(client.tools.values()),
                call_tool=client.call_tool,
                recorder=rec,
            )

        results = emit_results(
            trajectory_path=traj_path,
            tools=list(client.tools.values()),
            output_path=ctx.output_dir / "results.json",
            started_at="2026-05-06T00:00:00.000+00:00",
            scenario=scenario,
        )

    assert results["termination_report"]["label"] == "complete"
    assert results["totals"]["model_calls"] == 3
    assert results["totals"]["tool_calls"] == 2
    assert results["reference_calls_report"]["passed"]
    assert results["reference_calls_report"]["must_include_hits"] == 2
    assert results["conclusion_rubric_report"]["passed"]
    assert results["k8sgpt_version"] == "0.4.32"
