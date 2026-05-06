"""Validates the shipped scenario library and profile library.

Loads every YAML under eval/scenarios/specs/ and eval/scenarios/profiles/
and asserts they parse cleanly, declare valid references, and meet the
v0.1 coverage targets from ROADMAP.md Phase 2.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.scenarios import (
    Scenario,
    compose_profile,
    load_profiles,
    load_scenarios,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SPECS_DIR = REPO_ROOT / "eval" / "scenarios" / "specs"
PROFILES_DIR = REPO_ROOT / "eval" / "scenarios" / "profiles"


@pytest.fixture(scope="module")
def shipped_scenarios() -> list[Scenario]:
    return load_scenarios(SPECS_DIR)


@pytest.fixture(scope="module")
def shipped_profiles() -> dict[str, object]:
    return load_profiles(PROFILES_DIR)


def test_scenarios_load_cleanly(shipped_scenarios: list[Scenario]) -> None:
    assert len(shipped_scenarios) >= 10, "expected at least 10 scenarios per ROADMAP Phase 2"


def test_scenario_ids_are_unique(shipped_scenarios: list[Scenario]) -> None:
    ids = [s.id for s in shipped_scenarios]
    assert len(ids) == len(set(ids))


def test_every_scenario_has_a_goal_and_at_least_one_setup_step(
    shipped_scenarios: list[Scenario],
) -> None:
    for s in shipped_scenarios:
        assert s.goal.strip(), f"{s.id}: empty goal"
        assert s.setup, f"{s.id}: no setup steps"


def test_every_scenario_references_an_existing_profile(
    shipped_scenarios: list[Scenario], shipped_profiles: dict[str, object]
) -> None:
    for s in shipped_scenarios:
        assert s.profile in shipped_profiles, f"{s.id}: unknown profile {s.profile!r}"


def test_every_scenario_has_a_conclusion_rubric_with_at_least_one_mention(
    shipped_scenarios: list[Scenario],
) -> None:
    for s in shipped_scenarios:
        rubric = s.expected.conclusion_rubric
        assert rubric.must_mention, f"{s.id}: conclusion_rubric.must_mention is empty"


def test_every_scenario_has_at_least_one_must_include_reference_call(
    shipped_scenarios: list[Scenario],
) -> None:
    for s in shipped_scenarios:
        assert s.expected.reference_calls.must_include, (
            f"{s.id}: reference_calls.must_include is empty"
        )


def test_v0_1_coverage_targets_are_met(shipped_scenarios: list[Scenario]) -> None:
    """ROADMAP.md Phase 2: pod-startup, networking, resources, RBAC, storage, config, scheduling."""
    ids = {s.id for s in shipped_scenarios}
    assert "pod-crashloop-001" in ids
    assert "image-pull-001" in ids
    assert "oom-killed-001" in ids
    assert "service-selector-mismatch-001" in ids
    assert "network-policy-block-001" in ids
    assert "rbac-denied-001" in ids
    assert "pvc-unbound-001" in ids
    assert "resource-quota-block-001" in ids
    assert "secret-missing-001" in ids
    assert "node-selector-unschedulable-001" in ids


def test_profiles_load_cleanly(shipped_profiles: dict[str, object]) -> None:
    assert "base" in shipped_profiles


def test_argocd_profile_composes_against_base(
    shipped_profiles: dict[str, object],
) -> None:
    if "argocd" not in shipped_profiles:
        pytest.skip("argocd profile not shipped")
    composed = compose_profile("argocd", shipped_profiles)  # type: ignore[arg-type]
    assert composed.node_image == "kindest/node:v1.31.4"  # inherited from base
    assert composed.install, "expected helm install steps from argocd profile"
