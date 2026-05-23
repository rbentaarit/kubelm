"""Scenario runner: cluster lifecycle + profile install + scenario apply + settle.

`scenario_context` is a context manager that brings up everything a
scenario needs to be evaluated against, yields a ScenarioContext to the
caller, and tears the cluster down on exit even if the caller raises.
The body of the with-block is where the model run plugs in (slice 2.5).

Per-scenario fresh kind cluster is the only strategy v0.1 ships — that
is the determinism floor for benchmarks (per the design discussion in
keen-coalescing-shannon.md). All shared paths (cluster name,
kubeconfig, helm home, output dir) are derived from `(run_id,
scenario_id)`, so calling this from multiple worker processes in
parallel is safe without changes — the parallel orchestration sits
above this module.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from eval.scenarios.cluster import (
    helm_install,
    helm_repo_add,
    kind_create_cluster,
    kind_delete_cluster,
    kubectl_apply,
    kubectl_apply_file,
    wait_for_status,
)
from eval.scenarios.profile import Profile
from eval.scenarios.spec import Scenario

log = logging.getLogger(__name__)

KIND_NAME_MAX = 63


@dataclass
class ScenarioContext:
    scenario: Scenario
    profile: Profile
    cluster_name: str
    kubeconfig_path: Path
    helm_home: Path
    output_dir: Path


def cluster_name_for(run_id: str, scenario_id: str) -> str:
    """Derive a DNS-compliant kind cluster name unique to this run+scenario."""
    short = run_id[:8]
    name = f"kubelm-{short}-{scenario_id}".lower()
    return name[:KIND_NAME_MAX]


def _resolve_apply_file(scenario: Scenario, relative: str) -> Path:
    p = Path(relative)
    if p.is_absolute():
        return p
    if scenario.source_path is None:
        raise ValueError(
            f"scenario {scenario.id!r}: 'apply_file: {relative}' is relative "
            f"but the scenario was not loaded from a file"
        )
    return scenario.source_path.parent / p


def _install_profile(profile: Profile, *, kubeconfig_path: Path, helm_home: Path) -> None:
    if profile.prebuilt_image is not None:
        log.info("profile %s uses prebuilt_image; skipping install steps", profile.name)
        return
    for step in profile.install:
        if step.helm_repo is not None:
            helm_repo_add(step.helm_repo.name, step.helm_repo.url, helm_home=helm_home)
        elif step.helm_install is not None:
            hi = step.helm_install
            helm_install(
                name=hi.name,
                chart=hi.chart,
                namespace=hi.namespace,
                kubeconfig_path=kubeconfig_path,
                helm_home=helm_home,
                version=hi.version,
            )
    for wf in profile.wait_for:
        wait_for_status(
            kind=wf.kind,
            name=wf.name,
            namespace=wf.namespace,
            kubeconfig_path=kubeconfig_path,
            reason=wf.reason,
            phase=wf.phase,
            condition=wf.condition,
            timeout_seconds=wf.timeout_seconds,
        )


def _apply_scenario(scenario: Scenario, *, kubeconfig_path: Path) -> None:
    for step in scenario.setup:
        if step.apply_inline is not None:
            kubectl_apply(step.apply_inline, kubeconfig_path=kubeconfig_path)
        elif step.apply_file is not None:
            kubectl_apply_file(
                _resolve_apply_file(scenario, step.apply_file),
                kubeconfig_path=kubeconfig_path,
            )


def _settle_scenario(scenario: Scenario, *, kubeconfig_path: Path) -> None:
    for cond in scenario.settle:
        wait_for_status(
            kind=cond.kind,
            name=cond.name,
            namespace=cond.namespace,
            kubeconfig_path=kubeconfig_path,
            reason=cond.reason,
            phase=cond.phase,
            condition=cond.condition,
            message_contains=cond.message_contains,
            timeout_seconds=cond.timeout_seconds,
        )


@contextmanager
def scenario_context(
    *,
    scenario: Scenario,
    profile: Profile,
    run_id: str,
    output_root: Path,
    create_cluster_timeout: float = 180,
) -> Iterator[ScenarioContext]:
    """Stand up everything the scenario needs and yield a ScenarioContext.

    The cluster is always torn down on exit, even if the body of the
    with-block raises. `profile` must already be composed via
    compose_profile() — this function does not resolve `extends:`.
    """
    output_dir = output_root / run_id / scenario.id
    workspace = output_dir / "_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    kubeconfig_path = workspace / "kubeconfig"
    helm_home = workspace / "helm"

    cluster_name = cluster_name_for(run_id, scenario.id)
    image = profile.prebuilt_image or profile.node_image

    log.info("creating kind cluster %s for scenario %s", cluster_name, scenario.id)
    kind_create_cluster(
        cluster_name,
        kubeconfig_path=kubeconfig_path,
        image=image,
        timeout=create_cluster_timeout,
    )
    try:
        _install_profile(profile, kubeconfig_path=kubeconfig_path, helm_home=helm_home)
        _apply_scenario(scenario, kubeconfig_path=kubeconfig_path)
        _settle_scenario(scenario, kubeconfig_path=kubeconfig_path)
        yield ScenarioContext(
            scenario=scenario,
            profile=profile,
            cluster_name=cluster_name,
            kubeconfig_path=kubeconfig_path,
            helm_home=helm_home,
            output_dir=output_dir,
        )
    finally:
        try:
            kind_delete_cluster(cluster_name, kubeconfig_path=kubeconfig_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("failed to delete kind cluster %s: %s", cluster_name, exc)
