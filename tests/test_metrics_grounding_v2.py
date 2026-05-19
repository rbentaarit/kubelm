"""Unit tests + calibration regression test for grounding_v2."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eval.audits.grounding.calibrate import _join_corpus, _load_scenario_goals, evaluate
from eval.metrics.grounding_v2 import V2Classifier

REPO = Path(__file__).resolve().parent.parent
AUDIT = REPO / "eval" / "audits" / "grounding" / "2026-05-19-kubelm-edge-v0" / "audit.yaml"
CORPUS = Path("/tmp/audit-with-corpus.yaml")


def _classify(fact: str, corpus: str = "", scenario_id: str = "x", goal: str = "") -> str:
    return V2Classifier().classify(fact=fact, corpus=corpus, scenario_id=scenario_id, goal=goal)


def test_literal_substring_match_is_rephrase() -> None:
    corpus = '{"phase": "Pending", "reason": "ContainersNotReady"}'
    assert _classify("phase: Pending", corpus=corpus) == "structural_rephrase"


def test_camel_case_vs_hyphen_match_is_rephrase() -> None:
    corpus = '"key": "node.kubernetes.io/not-ready", "effect": "NoExecute"'
    assert _classify("NotReady", corpus=corpus) == "structural_rephrase"


def test_json_to_dotted_rephrase_via_squash() -> None:
    corpus = '"state": {"waiting": {"reason": "CrashLoopBackOff"}}'
    pred = _classify('state.waiting.reason="CrashLoopBackOff"', corpus=corpus)
    assert pred == "structural_rephrase"


def test_invented_field_is_fabrication() -> None:
    corpus = '"httpGet": {"path": "/healthz"}, "failureThreshold": 2'
    assert _classify("expectedResponse", corpus=corpus) == "fabrication"
    assert _classify("successPath", corpus=corpus) == "fabrication"


def test_invented_taint_effect_is_fabrication() -> None:
    corpus = '"key": "node.kubernetes.io/not-ready", "effect": "NoExecute"'
    assert _classify("NoGo", corpus=corpus) == "fabrication"


def test_version_tag_pattern_is_fabrication() -> None:
    corpus = '"image": "ghcr.io/example/app:v1.2.3"'
    assert _classify("v1.2.4", corpus=corpus) == "fabrication"


def test_version_tag_in_corpus_is_not_fabrication() -> None:
    """Don't false-positive when the actual recommended version is in corpus."""
    corpus = '"image": "ghcr.io/example/app:v1.2.4"'
    assert _classify("v1.2.4", corpus=corpus) != "fabrication"


def test_short_shorthand_defaults_to_composed() -> None:
    """Cluster shorthand like 2/2 has no significant tokens; treat as composed."""
    corpus = '"replicas": 2, "availableReplicas": 2, "readyReplicas": 2'
    assert _classify("2/2", corpus=corpus) == "composed_inference"


def test_tokens_all_present_but_not_contiguous_is_composed() -> None:
    """The 'Available: False' style: each token in corpus, never side-by-side."""
    corpus = '"type": "Available", "status": "False", "reason": "MinimumReplicas"'
    assert _classify("Available=False", corpus=corpus) == "composed_inference"


def test_empty_fact_is_rephrase() -> None:
    assert _classify("", corpus="anything") == "structural_rephrase"


def test_unsupported_resource_type_is_flagged() -> None:
    """A standalone 'networkpolicy' token names a K8sGPT MCP unsupported type."""
    assert _classify("networkpolicy", corpus="") == "unsupported_tool"


def test_namespace_containing_resource_quota_is_not_unsupported() -> None:
    """`resource-quota` inside a hyphenated namespace name shouldn't trip the
    unsupported-types rule — that rule fires only on the standalone token."""
    # No corpus support → should fall through to fabrication, not unsupported_tool.
    out = _classify(
        "scenario-resource-quota-block-001/local-path-provisioner",
        corpus="unrelated text",
    )
    assert out != "unsupported_tool"


# --- calibration regression test ---


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus YAML not regenerated")
def test_v2_calibration_against_v0_audit() -> None:
    """Pin v2's calibration on the n=114 Stage 2 labels.

    The thresholds here are the Stage 3 bar. A regression that drops
    fab precision below 90% or recall below 80% means a downstream
    rule change broke calibration — investigate before merging.
    """
    classified = yaml.safe_load(AUDIT.read_text())
    with_corpus = yaml.safe_load(CORPUS.read_text())
    corpora = _join_corpus(classified, with_corpus)
    goals = _load_scenario_goals()

    report = evaluate(V2Classifier(), classified, corpora, goals)

    assert report.n == 114
    assert report.fab_precision >= 0.90, f"v2 fab precision {report.fab_precision:.1%} < 90%"
    assert report.fab_recall >= 0.80, f"v2 fab recall {report.fab_recall:.1%} < 80%"
    assert report.rephrase_precision >= 0.95, (
        f"v2 rephrase precision {report.rephrase_precision:.1%} < 95%"
    )
