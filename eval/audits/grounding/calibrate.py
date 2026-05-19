"""Calibration harness for grounding analyzers.

Takes a per-fact classifier and a classified audit YAML (the Stage 2
output: 114 ungrounded-fact records with hand-applied labels), runs
the classifier against every fact, and reports the confusion matrix +
fabrication precision/recall + structural_rephrase precision.

Optionally runs leave-one-scenario-out cross-validation so a classifier
that was tuned by examining specific facts can be tested on facts it
hasn't seen. With n=27 scenarios in the v0 audit, this is 27 folds.

The harness is independent of which classifier is being graded — it
takes a Classifier protocol with one method, ``classify``. Used to
iterate ``grounding_v2`` against the v0 labels until precision/recall
clear the Stage 3 bar (>=90% fab precision, >=80% fab recall, >=95%
structural_rephrase precision under k-fold).

Usage:
    # Score v1 baseline (v1 has no label distinction, so this is the
    # "ceiling for recall, floor for precision" reference point).
    uv run python eval/audits/grounding/calibrate.py \\
        --classifier v1 \\
        --audit eval/audits/grounding/2026-05-19-kubelm-edge-v0/audit.yaml \\
        --corpus-yaml /tmp/audit-with-corpus.yaml

    # Score v2 once it exists.
    uv run python eval/audits/grounding/calibrate.py \\
        --classifier v2 \\
        --audit eval/audits/grounding/2026-05-19-kubelm-edge-v0/audit.yaml \\
        --corpus-yaml /tmp/audit-with-corpus.yaml \\
        --kfold-by-scenario

The audit YAML carries (scenario, fact, classification, rationale)
but NOT the tool corpus per-record (the slim form is 41 KB; the
corpus form is ~3 MB and gitignored). The harness joins on
(scenario_id, fact) against the corpus-inflated YAML in /tmp.
Regenerate with prepare.py --include-corpus.
"""

from __future__ import annotations

import argparse
import collections
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml

# Audit labels: 4 "not really fabrications" + 1 real defect.
LABELS = (
    "fabrication",
    "structural_rephrase",
    "composed_inference",
    "scenario_fill",
    "unsupported_tool",
)


class Classifier(Protocol):
    """Per-fact classifier protocol.

    The bench-time analyzer's interface is trajectory-level, but for
    calibration we extract per-fact (fact + tool corpus + scenario
    metadata) and ask one decision per fact.
    """

    name: str

    def classify(
        self,
        *,
        fact: str,
        corpus: str,
        scenario_id: str,
        goal: str,
    ) -> str: ...


@dataclass
class V1Classifier:
    """v1 analyzer wrapped to fit the per-fact protocol.

    v1 is binary: any fact it surfaces is "ungrounded". The audit was
    constructed from v1's ungrounded list, so v1 predicts "fabrication"
    for every audit fact (the closest of the 5 labels to v1's binary
    output). This gives us: fab-precision = 14/114 = 12.3%, fab-recall
    = 100%. The floor for precision, the ceiling for recall.
    """

    name: str = "v1"

    def classify(self, *, fact, corpus, scenario_id, goal) -> str:
        return "fabrication"


@dataclass
class CalibrationReport:
    """One pass through the audit."""

    classifier_name: str
    n: int
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)  # (gold, pred) -> count
    fab_precision: float = 0.0
    fab_recall: float = 0.0
    rephrase_precision: float = 0.0
    mistakes: list[tuple[str, str, str, str]] = field(default_factory=list)
    # (scenario, fact, gold_label, predicted_label)

    def headline(self) -> str:
        return (
            f"{self.classifier_name}: "
            f"fab P={self.fab_precision:.1%} "
            f"R={self.fab_recall:.1%}  "
            f"rephrase P={self.rephrase_precision:.1%}  "
            f"(n={self.n})"
        )


def _join_corpus(
    classified_audit: list[dict],
    corpus_audit: list[dict],
) -> dict[tuple[str, str], str]:
    """Map (scenario_id, fact) -> tool corpus string."""
    return {
        (r["scenario_id"], r["fact"]): r.get("tool_results_searchable", "") for r in corpus_audit
    }


def _load_scenario_goals() -> dict[str, str]:
    goals: dict[str, str] = {}
    for sp in Path("eval/scenarios/specs").glob("*.yaml"):
        try:
            d = yaml.safe_load(sp.read_text())
            goals[sp.stem] = d.get("goal", "") if isinstance(d, dict) else ""
        except Exception:  # noqa: BLE001
            pass
    return goals


def evaluate(
    classifier: Classifier,
    records: list[dict],
    corpora: dict[tuple[str, str], str],
    goals: dict[str, str],
) -> CalibrationReport:
    """Score a classifier against every record in ``records``.

    Returns precision/recall numbers for fabrication and
    structural_rephrase, plus the full confusion matrix and a
    mistake list for hand-iteration.
    """
    confusion: dict[tuple[str, str], int] = collections.Counter()
    mistakes: list[tuple[str, str, str, str]] = []

    for rec in records:
        gold = rec["classification"]
        scen = rec["scenario_id"]
        fact = rec["fact"]
        corpus = corpora.get((scen, fact), "")
        goal = goals.get(scen, "")
        pred = classifier.classify(fact=fact, corpus=corpus, scenario_id=scen, goal=goal)
        confusion[(gold, pred)] += 1
        if pred != gold:
            mistakes.append((scen, fact, gold, pred))

    # fab precision/recall
    tp = confusion.get(("fabrication", "fabrication"), 0)
    fp = sum(n for (g, p), n in confusion.items() if p == "fabrication" and g != "fabrication")
    fn = sum(n for (g, p), n in confusion.items() if g == "fabrication" and p != "fabrication")
    fab_precision = tp / (tp + fp) if (tp + fp) else 0.0
    fab_recall = tp / (tp + fn) if (tp + fn) else 0.0

    # rephrase precision (recall isn't a useful number here — if v2
    # decides a rephrase is grounded under another label, that's still
    # correct for the bench's headline)
    r_tp = confusion.get(("structural_rephrase", "structural_rephrase"), 0)
    r_fp = sum(
        n
        for (g, p), n in confusion.items()
        if p == "structural_rephrase" and g != "structural_rephrase"
    )
    rephrase_precision = r_tp / (r_tp + r_fp) if (r_tp + r_fp) else 0.0

    return CalibrationReport(
        classifier_name=classifier.name,
        n=sum(confusion.values()),
        confusion=dict(confusion),
        fab_precision=fab_precision,
        fab_recall=fab_recall,
        rephrase_precision=rephrase_precision,
        mistakes=mistakes,
    )


def kfold_by_scenario(
    classifier_factory: Callable[[], Classifier],
    records: list[dict],
    corpora: dict[tuple[str, str], str],
    goals: dict[str, str],
) -> tuple[float, float, float]:
    """Leave-one-scenario-out cross-validation.

    The classifier is stateless in our setup (no training step), so
    "leave-one-out" really just means scoring on each fold separately
    and averaging — it surfaces whether the classifier's correctness
    is concentrated on a few easy scenarios or evenly distributed.
    """
    scenarios = sorted({r["scenario_id"] for r in records})
    # Average each metric over folds where the metric is *defined* —
    # folds with no positives for a label return 0/0 in evaluate(),
    # which would drag the macro average to zero artifactually.
    fab_p_terms, fab_r_terms, reph_p_terms = [], [], []
    for held in scenarios:
        fold = [r for r in records if r["scenario_id"] == held]
        rep = evaluate(classifier_factory(), fold, corpora, goals)
        fab_pred = sum(1 for s, f, g, p in rep.mistakes if p == "fabrication") + rep.confusion.get(
            ("fabrication", "fabrication"), 0
        )
        fab_gold = sum(1 for r in fold if r["classification"] == "fabrication")
        reph_pred = sum(
            1 for s, f, g, p in rep.mistakes if p == "structural_rephrase"
        ) + rep.confusion.get(("structural_rephrase", "structural_rephrase"), 0)
        if fab_pred > 0:
            fab_p_terms.append(rep.fab_precision)
        if fab_gold > 0:
            fab_r_terms.append(rep.fab_recall)
        if reph_pred > 0:
            reph_p_terms.append(rep.rephrase_precision)
    fp = sum(fab_p_terms) / len(fab_p_terms) if fab_p_terms else 0.0
    fr = sum(fab_r_terms) / len(fab_r_terms) if fab_r_terms else 0.0
    rp = sum(reph_p_terms) / len(reph_p_terms) if reph_p_terms else 0.0
    return (fp, fr, rp)


def print_confusion(report: CalibrationReport) -> None:
    print(f"\n{report.classifier_name} confusion (rows=gold, cols=predicted):")
    cols = LABELS
    header = "                       " + "  ".join(f"{c[:6]:>6}" for c in cols)
    print(header)
    for gold in LABELS:
        row = [report.confusion.get((gold, c), 0) for c in cols]
        cells = "  ".join(f"{n:>6}" for n in row)
        print(f"  {gold[:20]:<20} {cells}")


def print_mistakes(report: CalibrationReport, limit: int = 20) -> None:
    if not report.mistakes:
        return
    print(f"\nMistakes ({len(report.mistakes)} total, showing up to {limit}):")
    for scen, fact, gold, pred in report.mistakes[:limit]:
        print(f"  [{scen:<32}] gold={gold:<22} pred={pred:<22} fact={fact!r}")


CLASSIFIER_REGISTRY: dict[str, Callable[[], Classifier]] = {
    "v1": V1Classifier,
}


def _try_register_v2() -> None:
    """Register v2 if the module has shipped. Quiet no-op otherwise so
    this harness is useful before v2 exists."""
    try:
        from eval.metrics.grounding_v2 import V2Classifier  # type: ignore[import-not-found]

        CLASSIFIER_REGISTRY["v2"] = V2Classifier
    except Exception:
        pass


def main() -> int:
    _try_register_v2()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--classifier",
        choices=sorted(CLASSIFIER_REGISTRY),
        required=True,
        help="Which classifier to grade.",
    )
    p.add_argument(
        "--audit",
        type=Path,
        required=True,
        help="Path to the classified audit.yaml (the Stage 2 output).",
    )
    p.add_argument(
        "--corpus-yaml",
        type=Path,
        required=True,
        help="Path to the corpus-inflated audit YAML "
        "(produced by prepare.py --include-corpus). Provides tool_results_searchable.",
    )
    p.add_argument(
        "--kfold-by-scenario",
        action="store_true",
        help="Also report leave-one-scenario-out cross-validation averages.",
    )
    p.add_argument(
        "--show-mistakes",
        type=int,
        default=20,
        help="Max number of mistakes to print (set 0 to suppress).",
    )
    args = p.parse_args()

    classified = yaml.safe_load(args.audit.read_text())
    with_corpus = yaml.safe_load(args.corpus_yaml.read_text())
    if not classified or not with_corpus:
        print("empty input")
        return 1

    corpora = _join_corpus(classified, with_corpus)
    goals = _load_scenario_goals()

    factory = CLASSIFIER_REGISTRY[args.classifier]
    classifier = factory()

    report = evaluate(classifier, classified, corpora, goals)
    print(report.headline())
    print_confusion(report)
    if args.show_mistakes:
        print_mistakes(report, limit=args.show_mistakes)

    if args.kfold_by_scenario:
        fp, fr, rp = kfold_by_scenario(factory, classified, corpora, goals)
        print(f"\nk-fold-by-scenario (avg of {len({r['scenario_id'] for r in classified})} folds):")
        print(f"  fab P={fp:.1%}  R={fr:.1%}  rephrase P={rp:.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
