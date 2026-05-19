"""Add ``trajectory_consistency_report`` to a Shape B summary in-place.

Standalone helper used for Stage 1.6 of the v0.1 plan: backfill the
narrative-consistency metric (committed at ``bab8733`` / wired in at
``a9218be``) onto every committed summary so historical artifacts
carry the same five-metric shape as fresh runs.

For each non-errored run, the script loads the corresponding
``trajectory.jsonl``, runs ``analyze_trajectory_consistency``, rewrites
the per-run ``results.json`` to include the new report block, and
records ``trajectory_consistency_passed`` + ``narrative_inconsistencies``
on the summary's ``RunRecord``. Per-model totals are recomputed and the
summary's ``schema_version`` is bumped 1 -> 2 if any run was rescored.

Other metric blocks are left as-recorded — this is additive only.

Usage:
    uv run python eval/results/summaries/regrade_trajectory_consistency.py \\
        eval/results/summaries/shape-b-2026-05-11.json

Trajectories must still be on disk under
``eval/results/<run_id>/<scenario_id>/trajectory.jsonl`` (or the
checkpoints subtree for kubelm-edge attempts).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from eval.metrics import analyze_trajectory_consistency
from eval.runner.results import _trajectory_consistency_dict
from eval.trajectory import load_trajectory

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def main(summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text())

    rescored = 0
    skipped: list[tuple[str, str, str]] = []
    failing_runs: list[tuple[str, str, int]] = []

    for run in summary["runs"]:
        if run.get("error"):
            skipped.append((run["model"], run["scenario"], "errored"))
            continue

        results_path = REPO_ROOT / run["results_path"]
        traj_path = results_path.parent / "trajectory.jsonl"
        if not traj_path.exists():
            skipped.append((run["model"], run["scenario"], "no trajectory"))
            continue

        events = load_trajectory(traj_path)
        report = analyze_trajectory_consistency(events)

        if results_path.exists():
            results = json.loads(results_path.read_text())
            results["trajectory_consistency_report"] = _trajectory_consistency_dict(report)
            results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

        run["trajectory_consistency_passed"] = bool(report.passed)
        run["narrative_inconsistencies"] = len(report.inconsistent_claims)
        rescored += 1

        if not report.passed:
            failing_runs.append((run["model"], run["scenario"], len(report.inconsistent_claims)))

    for m in summary["models"]:
        name = m["name"]
        runs = [r for r in summary["runs"] if r["model"] == name]
        ms = summary["model_summaries"].setdefault(name, {})
        ms["trajectory_consistency_passed"] = sum(
            1 for r in runs if r.get("trajectory_consistency_passed") is True
        )
        ms["narrative_inconsistencies_total"] = sum(
            r.get("narrative_inconsistencies") or 0 for r in runs
        )

    if rescored > 0 and summary.get("schema_version", 1) < 2:
        summary["schema_version"] = 2

    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"Re-graded {summary_path.name}: rescored {rescored} runs, skipped {len(skipped)}")
    if skipped:
        for model, scenario, reason in skipped:
            print(f"  skip {model:<28} x {scenario:<35} ({reason})")
    print()
    print("Per-model narrative-consistency totals:")
    for name, s in summary["model_summaries"].items():
        runs_attempted = s.get("scenarios_attempted", 0)
        passed = s.get("trajectory_consistency_passed", 0)
        incons = s.get("narrative_inconsistencies_total", 0)
        print(f"  {name:<28} narr_pass = {passed}/{runs_attempted}  inconsistencies = {incons}")
    if failing_runs:
        print()
        print("Runs with narrative inconsistencies (model | scenario | n claims):")
        for model, scenario, n in failing_runs:
            print(f"  {model:<28} {scenario:<35} {n}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(Path(sys.argv[1]))
