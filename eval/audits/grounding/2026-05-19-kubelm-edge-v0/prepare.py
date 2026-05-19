"""Build the working YAML for the kubelm-edge-v0 grounding audit.

Walks every scenario under a run directory (e.g. attempt-2's output)
and emits one record per ungrounded fact, with enough context for a
human auditor to classify it without opening the trajectory files.

Output shape (YAML, one document with a top-level list):

  - scenario_id: pod-crashloop-001
    fact: "exit code 1"
    conclusion_excerpt: "...around the fact in the conclusion..."
    tool_results_searchable: "...flat concat of every tool result..."
    classification: ""   # human fills in
    rationale: ""        # human fills in

The script is intentionally simple and idempotent: re-running
overwrites the output but preserves any committed audit.yaml in
git history. Don't run on a partially-classified working file
unless you're prepared to re-do the classification.

Usage:
    uv run python eval/audits/grounding/2026-05-19-kubelm-edge-v0/prepare.py \\
        --run-dir eval/results/checkpoints/kubelm-edge-v0-attempt-2 \\
        --out    eval/audits/grounding/2026-05-19-kubelm-edge-v0/audit.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load_trajectory(path: Path) -> list[dict[str, Any]]:
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def _extract_conclusion(traj: list[dict[str, Any]]) -> str:
    """The last assistant turn with non-empty text is the conclusion."""
    last = ""
    for e in traj:
        if e.get("kind") == "assistant" and e.get("text"):
            last = e["text"]
    return last


def _extract_tool_results_corpus(traj: list[dict[str, Any]]) -> str:
    """Flatten every tool result into a single searchable string."""
    parts: list[str] = []
    for e in traj:
        if e.get("kind") != "tool_result":
            continue
        content = e.get("content")
        if isinstance(content, dict):
            # MCP shape: {"content": [{"type": "text", "text": "..."}]}
            for c in content.get("content", []):
                if isinstance(c, dict) and c.get("type") == "text":
                    parts.append(c.get("text", ""))
        elif isinstance(content, str):
            parts.append(content)
    return "\n--- next tool result ---\n".join(parts)


def _excerpt_around(text: str, needle: str, window: int = 200) -> str:
    """~window chars around the first match of needle in text, else
    the first window chars of text. Whitespace-collapsed."""
    if not text:
        return ""
    idx = text.find(needle)
    if idx < 0:
        # try a softer match — just first window chars of conclusion
        snip = text[:window]
    else:
        start = max(0, idx - window // 2)
        end = min(len(text), idx + len(needle) + window // 2)
        snip = text[start:end]
    return " ".join(snip.split())


def build_audit_records(run_dir: Path) -> list[dict[str, Any]]:
    """Collect one record per ungrounded fact across all scenarios.

    The bench writes one run-id dir per (model × scenario) pair, each
    containing exactly one scenario subdir with results.json +
    trajectory.jsonl. Bench-level aggregation lives under
    `benchmarks/<bench_id>/summary.json`. We walk every run-id dir
    (excluding `benchmarks/`) and aggregate; duplicate scenario IDs
    (e.g. from multiple bench passes) keep the most recent run by
    mtime, so a re-run overrides an earlier one.
    """
    run_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name != "benchmarks"]
    if not run_dirs:
        raise SystemExit(f"no run-id subdirs under {run_dir}")

    # scenario_id -> (mtime, run_dir, scenario_dir)
    latest: dict[str, tuple[float, Path, Path]] = {}
    for run_root in run_dirs:
        scen_subdirs = [d for d in run_root.iterdir() if d.is_dir()]
        if len(scen_subdirs) != 1:
            # bench's contract is one scenario per run-id; skip anything else
            print(f"warn: {run_root} has {len(scen_subdirs)} scenario dirs, skipping")
            continue
        scen_dir = scen_subdirs[0]
        mtime = scen_dir.stat().st_mtime
        prior = latest.get(scen_dir.name)
        if prior is None or mtime > prior[0]:
            latest[scen_dir.name] = (mtime, run_root, scen_dir)

    print(f"found {len(latest)} unique scenarios under {run_dir}")

    records: list[dict[str, Any]] = []
    for scen_id in sorted(latest):
        _, _, scen_dir = latest[scen_id]
        results_path = scen_dir / "results.json"
        traj_path = scen_dir / "trajectory.jsonl"
        if not results_path.exists() or not traj_path.exists():
            continue

        results = json.loads(results_path.read_text())
        gr = results.get("grounding_report", {})
        if not gr.get("has_grounding_failure"):
            continue

        traj = _load_trajectory(traj_path)
        conclusion = _extract_conclusion(traj)
        tool_corpus = _extract_tool_results_corpus(traj)

        for fact in gr.get("facts", []):
            if fact.get("grounded"):
                continue
            rec = {
                "scenario_id": scen_id,
                "fact": fact.get("fact", ""),
                "conclusion_excerpt": _excerpt_around(conclusion, fact.get("fact", "")),
                "classification": "",
                "rationale": "",
            }
            # tool_results_searchable is large (multi-MB across 114 records) and
            # reproducible from trajectory.jsonl. Only include when explicitly
            # requested for offline grep; the committed audit.yaml stays slim.
            if INCLUDE_CORPUS:
                rec["tool_results_searchable"] = tool_corpus
            records.append(rec)

    return records


# Module-level toggle set by --include-corpus. Read inside build_audit_records.
INCLUDE_CORPUS = False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Bench output dir (e.g. eval/results/checkpoints/kubelm-edge-v0-attempt-2).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Where to write the working YAML.",
    )
    p.add_argument(
        "--include-corpus",
        action="store_true",
        help=(
            "Inline the full per-scenario tool-result corpus into each record. "
            "Useful for offline grep workflows but makes the YAML several MB. "
            "Off by default; the committed audit.yaml is the slim form."
        ),
    )
    args = p.parse_args()

    global INCLUDE_CORPUS
    INCLUDE_CORPUS = args.include_corpus

    records = build_audit_records(args.run_dir)
    print(
        f"emitted {len(records)} ungrounded-fact records across "
        f"{len({r['scenario_id'] for r in records})} scenarios"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Sort by scenario then fact for deterministic diffs.
    records.sort(key=lambda r: (r["scenario_id"], r["fact"]))
    args.out.write_text(
        "# kubelm-edge-v0 grounding audit — working file\n"
        "# See README.md in this directory for the classification taxonomy.\n"
        "# Fill in `classification:` and `rationale:` for each entry.\n"
        + yaml.safe_dump(records, sort_keys=False, width=120, allow_unicode=True)
    )
    print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
