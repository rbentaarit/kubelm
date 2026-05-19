"""Aggregate a classified audit YAML into a markdown summary.

Reads a YAML produced by prepare.py and human-classified by the
auditor. Prints a five-row table (one per classification label) plus
a per-scenario breakdown. Writes a `summary-<date>.md` next to the
audit YAML so it can be committed alongside.

Usage:
    uv run python eval/audits/grounding/2026-05-19-kubelm-edge-v0/summarize.py \\
        eval/audits/grounding/2026-05-19-kubelm-edge-v0/audit.yaml
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
from pathlib import Path

import yaml

VALID_LABELS = (
    "fabrication",
    "structural_rephrase",
    "composed_inference",
    "scenario_fill",
    "unsupported_tool",
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audit_yaml", type=Path)
    args = p.parse_args()

    records = yaml.safe_load(args.audit_yaml.read_text())
    total = len(records)
    if total == 0:
        print("no records")
        return 0

    # Bucket by classification + by scenario
    by_label: collections.Counter[str] = collections.Counter()
    by_scenario: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    unclassified: list[dict] = []
    invalid: list[dict] = []

    for r in records:
        label = (r.get("classification") or "").strip()
        if not label:
            unclassified.append(r)
            continue
        if label not in VALID_LABELS:
            invalid.append(r)
            continue
        by_label[label] += 1
        by_scenario[r["scenario_id"]][label] += 1

    classified = total - len(unclassified) - len(invalid)

    # --- text output ---
    lines: list[str] = []
    lines.append(f"# Grounding audit summary — {args.audit_yaml.parent.name}")
    lines.append("")
    all_scenarios = {r["scenario_id"] for r in records}
    lines.append(
        f"Audit file: `{args.audit_yaml.name}`  ({total} ungrounded-fact records "
        f"across {len(all_scenarios)} scenarios)"
    )
    lines.append("")
    if unclassified or invalid:
        lines.append(
            f"⚠️  {len(unclassified)} unclassified, {len(invalid)} invalid labels — "
            f"summary covers only the {classified} classified records."
        )
        lines.append("")

    lines.append("## Classification totals")
    lines.append("")
    lines.append("| Label | Count | % of classified |")
    lines.append("|---|---|---|")
    for label in VALID_LABELS:
        n = by_label[label]
        pct = (100.0 * n / classified) if classified else 0
        lines.append(f"| `{label}` | {n} | {pct:.1f}% |")
    lines.append(f"| **total classified** | **{classified}** | 100.0% |")
    lines.append("")
    lines.append(
        "Only `fabrication` counts as a real model defect. The other four are "
        "metric blind-spots or expected behavior."
    )
    lines.append("")

    lines.append("## Per-scenario breakdown")
    lines.append("")
    lines.append("| Scenario | fab | reph | comp | sc_fill | unsup | total |")
    lines.append("|---|---|---|---|---|---|---|")
    for scen in sorted(by_scenario):
        counts = by_scenario[scen]
        row = [
            scen,
            counts["fabrication"],
            counts["structural_rephrase"],
            counts["composed_inference"],
            counts["scenario_fill"],
            counts["unsupported_tool"],
            sum(counts.values()),
        ]
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    lines.append("")

    fabrications = [r for r in records if (r.get("classification") or "").strip() == "fabrication"]
    if fabrications:
        lines.append("## Fabrications (the load-bearing list)")
        lines.append("")
        for r in fabrications:
            lines.append(f"- **{r['scenario_id']}** — `{r['fact']}`")
            if r.get("rationale"):
                lines.append(f"  - {r['rationale']}")
        lines.append("")

    if unclassified:
        lines.append("## Unclassified entries")
        lines.append("")
        for r in unclassified[:20]:
            lines.append(f"- {r['scenario_id']} — `{r['fact']}`")
        if len(unclassified) > 20:
            lines.append(f"- ... ({len(unclassified) - 20} more)")
        lines.append("")

    if invalid:
        lines.append("## Invalid labels")
        lines.append("")
        for r in invalid:
            lines.append(
                f"- {r['scenario_id']} — `{r['fact']}` — "
                f"label `{r['classification']}` not in {list(VALID_LABELS)}"
            )
        lines.append("")

    print("\n".join(lines))

    out_path = args.audit_yaml.with_name(f"summary-{dt.date.today().isoformat()}.md")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
