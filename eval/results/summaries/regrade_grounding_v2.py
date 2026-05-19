"""Add ``grounding_v2_report`` to a Shape B summary in-place.

Standalone helper used for Stage 3 of the v0.1 plan: backfill the
v2 grounding analyzer (committed at ``5ff0c2b``) onto every committed
summary so historical artifacts carry the same six-metric shape as
fresh runs.

For each non-errored run, the script loads the corresponding
``trajectory.jsonl``, runs ``analyze_grounding_v2``, rewrites the
per-run ``results.json`` to include the new report block, and updates
the summary's ``RunRecord`` fields with the v2 semantics:
- ``grounding_failed`` is redefined to "fabrication present"
  (was: "any ungrounded fact"). Schema 3+ readers should rely on the
  schema_version to interpret this column.
- ``fabrications`` is added as a per-run count.

``model_summaries`` gains ``fabrications_total``. The summary's
``schema_version`` is bumped 2 -> 3 if any run was rescored.

All other metric blocks are left as-recorded -- this is additive only.

Usage:
    uv run python eval/results/summaries/regrade_grounding_v2.py \\
        eval/results/summaries/shape-b-2026-05-11.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from eval.metrics import analyze_grounding_v2
from eval.runner.results import _grounding_v2_dict
from eval.trajectory import load_trajectory

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def main(summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text())

    rescored = 0
    skipped: list[tuple[str, str, str]] = []
    fabricating_runs: list[tuple[str, str, int]] = []
    old_flags: dict[tuple[str, str], bool | None] = {}

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
        report = analyze_grounding_v2(events)

        if results_path.exists():
            results = json.loads(results_path.read_text())
            results["grounding_v2_report"] = _grounding_v2_dict(report)
            results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

        old_flags[(run["model"], run["scenario"])] = run.get("grounding_failed")
        run["grounding_failed"] = bool(report.has_fabrication)
        run["fabrications"] = report.fabrications
        rescored += 1

        if report.has_fabrication:
            fabricating_runs.append((run["model"], run["scenario"], report.fabrications))

    for m in summary["models"]:
        name = m["name"]
        runs = [r for r in summary["runs"] if r["model"] == name]
        ms = summary["model_summaries"].setdefault(name, {})
        ms["grounding_failures"] = sum(1 for r in runs if r.get("grounding_failed"))
        ms["fabrications_total"] = sum(r.get("fabrications") or 0 for r in runs)

    if rescored > 0 and summary.get("schema_version", 1) < 3:
        summary["schema_version"] = 3

    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"Re-graded {summary_path.name}: rescored {rescored} runs, skipped {len(skipped)}")
    if skipped:
        for model, scenario, reason in skipped:
            print(f"  skip {model:<28} x {scenario:<35} ({reason})")
    print()
    print("Per-model fabrication totals (v2 semantics):")
    for name, s in summary["model_summaries"].items():
        runs_attempted = s.get("scenarios_attempted", 0)
        fab_runs = s.get("grounding_failures", 0)
        fab_total = s.get("fabrications_total", 0)
        print(
            f"  {name:<28} fab_runs = {fab_runs}/{runs_attempted}  fabrications_total = {fab_total}"
        )

    # Cells whose grounding_failed flipped: meaningful for the methodology log.
    new_flag = {(r["model"], r["scenario"]): r.get("grounding_failed") for r in summary["runs"]}
    flipped = [
        (m, s, bool(old), bool(new_flag.get((m, s))))
        for (m, s), old in old_flags.items()
        if bool(new_flag.get((m, s))) != bool(old)
    ]
    if flipped:
        print()
        print("Cells whose grounding_failed flipped under v2 (model | scenario | v1 -> v2):")
        for model, scenario, old, new in flipped:
            print(f"  {model:<28} {scenario:<35} {old} -> {new}")

    if fabricating_runs:
        print()
        print("Runs WITH v2 fabrications (model | scenario | n fabrications):")
        for model, scenario, n in fabricating_runs:
            print(f"  {model:<28} {scenario:<35} {n}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(Path(sys.argv[1]))
