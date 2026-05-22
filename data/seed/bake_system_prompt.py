"""Bake the canonical system prompt into a corpus version.

Copies every trajectory from a source corpus version to a destination
version, replacing only the system prompt (top-level ``system_prompt``
field AND ``messages[0]`` where role==system) with the current canonical
``DEFAULT_SYSTEM_PROMPT`` from ``eval.runner.loop``. The investigation
trajectory is otherwise untouched — only the instruction the model
learns to obey changes.

This exists because the deployment prompt is the inference contract:
training data must use the same system prompt the model is served with,
or train/inference diverge (PROJECT.md commitment #4, #5). Keeping the
prompt in one place (loop.py) and baking from it guarantees they match.

Usage:
    uv run python -m data.seed.bake_system_prompt --src-version v01 --dst-version v02
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.runner.loop import DEFAULT_SYSTEM_PROMPT

SEED_ROOT = Path("data/seed")
# (subdir-under-seed-root, filename-suffix) for each tracked corpus file.
SOURCE_LAYOUT = [
    ("{ver}", "gpt-5.4-2026-05-20.jsonl"),
    ("varied/{ver}", "gpt-5.4-2026-05-20-varied.jsonl"),
    ("{ver}", "qwen2.5-7b-2026-05-20.jsonl"),
    ("varied/{ver}", "qwen2.5-7b-2026-05-20-varied.jsonl"),
]


def _rebake_record(rec: dict) -> dict:
    rec["system_prompt"] = DEFAULT_SYSTEM_PROMPT
    msg0 = rec["messages"][0]
    if msg0.get("role") != "system":
        raise ValueError(f"messages[0] is not a system message in {rec.get('trajectory_id')}")
    msg0["content"] = DEFAULT_SYSTEM_PROMPT
    return rec


def _bake_file(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(json.dumps(_rebake_record(json.loads(line))) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-version", required=True, help="e.g. v01")
    ap.add_argument("--dst-version", required=True, help="e.g. v02")
    args = ap.parse_args()

    total = 0
    for subdir_tmpl, fname in SOURCE_LAYOUT:
        src = SEED_ROOT / subdir_tmpl.format(ver=args.src_version) / fname
        dst = SEED_ROOT / subdir_tmpl.format(ver=args.dst_version) / fname
        n = _bake_file(src, dst)
        total += n
        print(f"{src} -> {dst}: {n} records")
    print(f"\nbaked {total} records with canonical system prompt:")
    print(f"  {DEFAULT_SYSTEM_PROMPT[:80]}...")


if __name__ == "__main__":
    main()
