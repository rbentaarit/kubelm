"""Re-grade the ``reference_calls_report`` in a Shape B summary in-place.

Standalone helper used after the ``ref_pass`` bug fix
(commit ee8c75c). Re-applies ``evaluate_reference_calls`` to each
run's stored trajectory, rewrites its ``results.json`` and the
parent ``summary.json``, and re-rolls ``model_summaries``. Only the
reference-calls metric changes; schema / grounding / termination /
conclusion_rubric are left as-recorded.

Usage:
    uv run python eval/results/summaries/regrade_ref_pass.py \
        eval/results/summaries/shape-b-2026-05-11.json

Trajectories must still be on disk under
``eval/results/<run_id>/<scenario_id>/trajectory.jsonl``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from eval.metrics import evaluate_reference_calls
from eval.runner.results import _reference_calls_dict
from eval.scenarios.spec import load_scenario
from eval.trajectory import load_trajectory

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SPECS_DIR = REPO_ROOT / "eval" / "scenarios" / "specs"


def main(summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text())
    scenarios = {p.stem: load_scenario(p) for p in SPECS_DIR.glob("*.yaml")}

    changes: list[tuple[str, str, bool, bool]] = []
    for run in summary["runs"]:
        if run.get("error"):
            continue
        scenario = scenarios[run["scenario"]]
        results_path = REPO_ROOT / run["results_path"]
        traj_path = results_path.parent / "trajectory.jsonl"
        if not traj_path.exists():
            print(f"  skip {run['model']} x {run['scenario']}: missing trajectory", file=sys.stderr)
            continue

        events = load_trajectory(traj_path)
        new_report = evaluate_reference_calls(events, scenario.expected.reference_calls)

        results = json.loads(results_path.read_text())
        results["reference_calls_report"] = _reference_calls_dict(new_report)
        results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

        old_passed = bool(run.get("reference_calls_passed"))
        new_passed = bool(new_report.passed)
        if old_passed != new_passed:
            changes.append((run["model"], run["scenario"], old_passed, new_passed))
        run["reference_calls_passed"] = new_passed

    for m in summary["models"]:
        name = m["name"]
        runs = [r for r in summary["runs"] if r["model"] == name]
        summary["model_summaries"][name]["reference_calls_passed"] = sum(
            1 for r in runs if r.get("reference_calls_passed") is True
        )

    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"Re-graded {summary_path.name}")
    print(f"  flipped {len(changes)} (model, scenario) cells:")
    for model, scenario, old, new in changes:
        print(f"    {model:<13} x {scenario:<35} {old} -> {new}")
    print()
    print("Updated model_summaries.reference_calls_passed:")
    for name, s in summary["model_summaries"].items():
        print(f"  {name:<13} ref_pass = {s['reference_calls_passed']}/10")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(Path(sys.argv[1]))
