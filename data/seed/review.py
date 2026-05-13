"""Apply REVIEW.md's checklist to a seed-corpus JSONL file in place.

Auto-review pass: walks each trajectory, applies the hard rules from
REVIEW.md, and for the grounding section uses a heuristic that
checks whether each "ungrounded" fact reported by the v1 grounding
analyzer is actually present (in some form) in the trajectory's
tool results. This implements the audit pattern from PROJECT.md
decisions log 2026-05-12: gpt-5.4 phrases tool output in
YAML-path / quoted / dotted-status formats the v1 analyzer can't
substring-match, but the underlying facts ARE in the tool results.

Status taxonomy (per FORMAT.md):

  accepted           — every checklist box passes
  accepted_with_edits— passes after a noted edit
  rejected           — at least one hard-rule violation
  unreviewed         — heuristic isn't confident; needs human eyes

Hard rules (any failure → rejected):

  - schema_name_halluc > 0
  - schema_arg_halluc > 0
  - termination_label != "complete"
  - conclusion_rubric_passed != True
    (we already filter to rubric-passing in convert.py, so a fail
     here implies upstream data drift)

Heuristic rule (grounding):

  Each "ungrounded" fact F from the eval grounding_report is
  normalized (lowercase, strip whitespace, strip outer
  quotes/brackets) and matched against a concatenation of all
  tool_result content (also normalized). We also strip dotted-path
  prefixes ("foo.bar.baz: value" → "value"), which is the most
  common gpt-5.4 verbose-paraphrase pattern.

  - If ≥ ARTIFACT_THRESHOLD (default 0.6) of ungrounded facts match,
    grounding_failed_v1_artifact = True and the trajectory is
    eligible for `accepted` despite grounding_failed: true.
  - Otherwise grounding_failed_v1_artifact = False and the
    trajectory is left `unreviewed` for human inspection.

Usage:
    uv run python data/seed/review.py data/seed/v0/<file>.jsonl

The script overwrites the input file with the reviewed records and
prints a summary table.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ARTIFACT_THRESHOLD = 0.6

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _normalize(s: str) -> str:
    """Lowercase, collapse whitespace, strip quotes/brackets/equals."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    # strip outer quotes/braces
    s = s.strip("\"'`{}[]()")
    return s


def _strip_path_prefix(fact: str) -> str:
    """For facts like 'foo.bar.baz: value' or 'foo.bar = value', return the value side."""
    # "x.y.z: value" or "x.y.z = value"
    m = re.match(r"^[\w\.\[\]\-]+\s*[:=]\s*(.+)$", fact)
    if m:
        return m.group(1).strip()
    return fact


def _tool_result_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        if m.get("role") == "tool":
            content = m.get("content") or ""
            if isinstance(content, str):
                parts.append(content)
            else:
                parts.append(json.dumps(content))
    return _normalize(" ".join(parts))


def _conclusion_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            return m.get("content") or ""
    return ""


def _ungrounded_facts(rec: dict[str, Any]) -> list[str]:
    """Pull ungrounded facts from the matching eval results.json on disk."""
    src_run = rec["provenance"].get("source_run_id")
    scen = rec["scenario_id"]
    if not src_run or not scen:
        return []
    results_path = REPO_ROOT / "eval" / "results" / src_run / scen / "results.json"
    if not results_path.exists():
        return []
    results = json.loads(results_path.read_text())
    grounding = results.get("grounding_report") or {}
    return [
        f.get("fact", "")
        for f in (grounding.get("facts") or [])
        if not f.get("grounded") and f.get("fact")
    ]


_STOPWORDS = {
    "true",
    "false",
    "null",
    "name",
    "type",
    "kind",
    "spec",
    "status",
    "items",
    "list",
    "the",
    "and",
    "with",
    "from",
    "this",
    "that",
    "for",
    "has",
    "have",
    "was",
}


def _fact_tokens(fact: str) -> list[str]:
    """Split a fact into 'meaningful' tokens. Drops short tokens, stopwords,
    and pure-punctuation tokens. Used for the multi-token containment check
    when raw substring matching fails (the common case for gpt-5.4's
    YAML-path / dotted-status / quoted-value renderings)."""
    raw = re.split(r"[\s\.\:\=\,\;\"\'\{\}\[\]\(\)/]+", fact.lower())
    out: list[str] = []
    for t in raw:
        t = t.strip()
        if len(t) < 4:
            continue
        if t in _STOPWORDS:
            continue
        out.append(t)
    return out


def _fact_grounded(fact: str, tool_text: str) -> bool:
    """True if the fact is plausibly present in tool_text after normalization."""
    for cand in {fact, _strip_path_prefix(fact)}:
        n = _normalize(cand)
        if not n:
            continue
        if n in tool_text:
            return True
        n_no_q = n.replace('"', "").replace("'", "")
        if n_no_q and n_no_q in tool_text:
            return True
    # Token-level fallback: every meaningful token of the fact appears in tool text.
    tokens = _fact_tokens(fact)
    return bool(tokens) and all(t in tool_text for t in tokens)


def _grounding_artifact_check(rec: dict[str, Any]) -> tuple[bool | None, list[str]]:
    """Return (is_v1_artifact, list_of_genuinely_ungrounded_facts)."""
    facts = _ungrounded_facts(rec)
    if not facts:
        return None, []

    tool_text = _tool_result_text(rec["messages"])
    genuinely_ungrounded: list[str] = []
    matched = 0
    for fact in facts:
        if _fact_grounded(fact, tool_text):
            matched += 1
        else:
            genuinely_ungrounded.append(fact)

    ratio = matched / len(facts)
    if ratio >= ARTIFACT_THRESHOLD:
        return True, genuinely_ungrounded
    return False, genuinely_ungrounded


def review_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Mutate `rec` in place with review_status + grounding artifact flag. Returns rec."""
    q = rec["quality"]
    notes: list[str] = []

    # Hard rules
    if q["schema_name_halluc"] > 0:
        rec["provenance"]["review_status"] = "rejected"
        notes.append("schema_name_halluc > 0")
        rec["provenance"]["review_notes"] = "; ".join(notes)
        return rec
    if q["schema_arg_halluc"] > 0:
        rec["provenance"]["review_status"] = "rejected"
        notes.append("schema_arg_halluc > 0")
        rec["provenance"]["review_notes"] = "; ".join(notes)
        return rec
    if q["termination_label"] != "complete":
        rec["provenance"]["review_status"] = "rejected"
        notes.append(f"termination_label={q['termination_label']}")
        rec["provenance"]["review_notes"] = "; ".join(notes)
        return rec
    if not q["conclusion_rubric_passed"]:
        rec["provenance"]["review_status"] = "rejected"
        notes.append("conclusion_rubric_passed=False")
        rec["provenance"]["review_notes"] = "; ".join(notes)
        return rec

    # Grounding heuristic
    is_artifact = None
    if q.get("grounding_failed"):
        is_artifact, genuine = _grounding_artifact_check(rec)
        q["grounding_failed_v1_artifact"] = is_artifact
        if is_artifact is True:
            notes.append(f"grounding: {len(genuine)} of N facts genuinely ungrounded (≤ threshold)")
        elif is_artifact is False:
            notes.append(
                f"grounding: too many genuinely ungrounded facts ({len(genuine)}); needs human look"
            )

    # Default verdict
    if q.get("grounding_failed") and is_artifact is False:
        rec["provenance"]["review_status"] = "unreviewed"
    else:
        rec["provenance"]["review_status"] = "accepted"

    if notes:
        rec["provenance"]["review_notes"] = "; ".join(notes)
    return rec


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl_path", type=Path, help="Path to the seed corpus JSONL.")
    p.add_argument(
        "--dry-run", action="store_true", help="Print verdicts without writing the file."
    )
    args = p.parse_args()

    records = [
        json.loads(line) for line in args.jsonl_path.read_text().splitlines() if line.strip()
    ]
    counts: dict[str, int] = {}
    rows: list[tuple[str, str, str]] = []
    for rec in records:
        review_record(rec)
        status = rec["provenance"]["review_status"]
        notes = rec["provenance"].get("review_notes", "")
        rows.append((rec["scenario_id"], status, notes))
        counts[status] = counts.get(status, 0) + 1

    # Pretty summary
    width = max(len(s) for s, _, _ in rows)
    for scen, status, notes in rows:
        print(f"  {scen:<{width}}  {status:<22}  {notes}")
    print()
    print("Summary:")
    for k, v in sorted(counts.items()):
        print(f"  {k:<22}  {v}")

    if not args.dry_run:
        with args.jsonl_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, default=str) + "\n")
        print(f"\nWrote {len(records)} reviewed records back to {args.jsonl_path}")
    else:
        print("\n(dry-run; file not modified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
