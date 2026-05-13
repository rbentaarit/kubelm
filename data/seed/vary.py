"""Generate generalization variants of a seed trajectory corpus.

The seed corpus produced by `convert.py` carries the exact identifiers
used in the kind cluster (namespace `scenario-<scenario_id>`, resource
names like `api-pod`, `data-pvc`, etc.). Training only on those seeds
would teach the model to memorize specific strings rather than the
structural pattern of each failure mode. This script produces N
variants per seed by substituting:

  - The namespace (`scenario-<scenario_id>` → a realistic name from
    a curated pool)
  - The primary resource names referenced in tool calls (`api-pod`,
    `data-pvc`, etc. → pool of realistic names)

Substitutions are consistent within a trajectory — every occurrence
across the system prompt, goal, assistant turns, tool_call arguments,
tool_result content (including JSON-stringified bodies), and the
final conclusion is updated together. The substitution order is
length-descending so longer keys can't be corrupted by being a
suffix of a shorter substitution.

Variants are deterministic given (scenario_id, variant_index): the
same input file + the same N always produces byte-identical output,
which makes diffs across seed-corpus versions reviewable.

Usage:
    uv run python data/seed/vary.py \
        --in data/seed/v0/gpt-5.4-2026-05-12.jsonl \
        --variants 5 \
        --out data/seed/varied/v0/gpt-5.4-2026-05-12-varied.jsonl

Output format matches FORMAT.md (schema_version 1) with provenance
fields adjusted:

  provenance.source         = "eval_bench_variation"
  provenance.variation_of   = <source trajectory_id>
  provenance.variation_idx  = 0..N-1
  provenance.variation_map  = {original: variant, ...}
  provenance.review_status  = "accepted"   (inherits from source;
                                            a tightening review can
                                            stamp these unreviewed)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any

# ---------- name pools ----------

NAMESPACE_POOL = [
    "prod-api",
    "staging-web",
    "payments-svc",
    "data-pipeline-prod",
    "ml-training",
    "observability",
    "kube-system-monitoring",
    "team-alpha",
    "team-beta",
    "backend-services",
    "frontend-edge",
    "sandbox-mlops",
    "ci-runners",
    "devops-tools",
    "dev-shared",
    "ingress-system",
    "security-tools",
    "monitoring-prod",
    "monitoring-staging",
    "messaging",
    "search-cluster",
    "auth-service-prod",
    "auth-service-stage",
    "growth-experiments",
    "media-pipeline",
    "billing-jobs",
    "marketing-events",
    "rec-engine",
    "logs-aggregator",
    "spinnaker-pipeline",
]

POD_NAME_POOL = [
    "web-server",
    "api-gateway",
    "worker-1",
    "batch-processor",
    "etl-job",
    "auth-service",
    "notification-svc",
    "payment-handler",
    "scheduler",
    "cache-warmer",
    "db-init",
    "log-aggregator",
    "metric-shipper",
    "image-resizer",
    "video-transcoder",
    "fraud-checker",
    "inventory-svc",
    "search-indexer",
    "ml-trainer",
    "feature-pipeline",
    "report-generator",
    "audit-logger",
    "session-proxy",
    "rate-limiter",
    "cdn-purger",
    "graph-traverser",
    "alert-router",
    "rollout-controller",
    "tenant-provisioner",
    "feature-flag-svc",
]

PVC_NAME_POOL = [
    "user-uploads",
    "ml-checkpoints",
    "session-cache",
    "build-artifacts",
    "audit-logs",
    "metrics-store",
    "media-uploads",
    "search-index",
    "shared-config",
    "tenant-data",
]

# Names that appear in our seed corpus as primary resources. The key
# is the resource name; the value is the pool to draw replacements
# from. Anything not listed here is left untouched.
PRIMARY_RESOURCE_POOLS: dict[str, list[str]] = {
    # pods
    "api-pod": POD_NAME_POOL,
    "crash-pod": POD_NAME_POOL,
    "hungry-pod": POD_NAME_POOL,
    "env-pod": POD_NAME_POOL,
    "bad-image": POD_NAME_POOL,
    "picky-pod": POD_NAME_POOL,
    "rbac-probe": POD_NAME_POOL,
    "blocked-app": POD_NAME_POOL,
    "healthcheck-pod": POD_NAME_POOL,
    "workload-pod": POD_NAME_POOL,
    "heavy-pod": POD_NAME_POOL,
    "db-pod": POD_NAME_POOL,
    "worker": POD_NAME_POOL,
    # deployments / statefulsets / cronjobs / etc.
    "api": POD_NAME_POOL,
    "web": POD_NAME_POOL,
    "db": POD_NAME_POOL,
    "metrics-collector": POD_NAME_POOL,
    "backup": POD_NAME_POOL,
    "data-import": POD_NAME_POOL,
    # pvcs
    "data-pvc": PVC_NAME_POOL,
    "data-volume": PVC_NAME_POOL,
    # services
    "api-svc": ["frontend-svc", "payment-svc", "search-svc", "auth-svc", "media-svc"],
    "healthcheck-svc": ["api-svc", "metrics-svc", "probes-svc", "health-svc"],
    # quota / policy / SA names
    "zero-pods": ["max-zero", "no-pods-allowed", "strict-pod-cap"],
    "tight-cpu": ["frugal-cpu", "low-cpu-limit", "constrained-cpu"],
    "default-deny-ingress": ["default-deny", "isolate-ns", "block-all-ingress"],
    "api-runner": ["app-runner", "service-runner", "workload-runner"],
    "no-perms-sa": ["read-only-sa", "limited-sa", "scoped-sa"],
}


# ---------- variant table generation ----------


def _det_pick(
    pool: list[str],
    used: set[str],
    scenario_id: str,
    variant_idx: int,
    key: str,
    avoid_substring: str | None = None,
) -> str:
    """Pick a name from `pool` deterministically and uniquely within `used`.

    If `avoid_substring` is set, candidates that contain it as a substring
    are skipped. Used for short source keys like `"db"` or `"api"` so the
    variant doesn't end up with the source as a prefix
    (`db` → `db-init` would otherwise leave `db` matchable at the
    start of `db-init` and confuse downstream consistency checks).
    """
    seed = f"{scenario_id}::{variant_idx}::{key}".encode()
    h = int(hashlib.sha256(seed).hexdigest(), 16)
    start = h % len(pool)
    for offset in range(len(pool)):
        candidate = pool[(start + offset) % len(pool)]
        if candidate in used:
            continue
        if avoid_substring and avoid_substring in candidate:
            continue
        used.add(candidate)
        return candidate
    # Pool exhausted: fall back with a deterministic suffix.
    return f"{pool[h % len(pool)]}-{variant_idx}"


def _detect_resource_names_in_messages(messages: list[dict[str, Any]]) -> set[str]:
    """Pluck 'name' / 'involvedObjectName' / 'podName' values from assistant tool_calls."""
    seen: set[str] = set()
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            args_raw = tc.get("function", {}).get("arguments") or "{}"
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                continue
            for k in ("name", "involvedObjectName", "podName"):
                v = args.get(k)
                if isinstance(v, str) and v:
                    seen.add(v)
    return seen


def build_variation_map(rec: dict[str, Any], variant_idx: int) -> dict[str, str]:
    """Return a {source_string: variant_string} mapping for one variant."""
    scenario_id = rec["scenario_id"]
    original_ns = f"scenario-{scenario_id}"

    used: set[str] = set()
    new_ns = _det_pick(NAMESPACE_POOL, used, scenario_id, variant_idx, "namespace")

    var_map: dict[str, str] = {original_ns: new_ns}

    # Resource names extracted from the trajectory plus any we know about
    # by pattern (the scenario's "main" pod/svc names listed in
    # PRIMARY_RESOURCE_POOLS).
    candidate_names = _detect_resource_names_in_messages(rec["messages"])
    for name in candidate_names:
        pool = PRIMARY_RESOURCE_POOLS.get(name)
        if pool is None:
            continue
        avoid = name if len(name) <= 4 else None
        var_map[name] = _det_pick(
            pool, used, scenario_id, variant_idx, f"name::{name}", avoid_substring=avoid
        )

    # Also vary names that PRIMARY_RESOURCE_POOLS knows about but the
    # model might reference only in the goal/conclusion, not in a tool
    # call. Build this set from a regex over the source `goal`.
    goal = rec.get("goal", "") or ""
    for known in PRIMARY_RESOURCE_POOLS:
        if known in var_map:
            continue
        if re.search(rf"\b{re.escape(known)}\b", goal):
            pool = PRIMARY_RESOURCE_POOLS[known]
            avoid = known if len(known) <= 4 else None
            var_map[known] = _det_pick(
                pool, used, scenario_id, variant_idx, f"goal::{known}", avoid_substring=avoid
            )

    return var_map


# ---------- substitution ----------


def apply_substitutions(text: str, var_map: dict[str, str]) -> str:
    """Substitute every key in var_map with its value in text.

    Substitutions are applied longest-key-first so a key cannot
    be corrupted by being a suffix of another key. Whole-word
    boundaries are honored for short keys (≤ 4 chars) to avoid
    over-matching; longer keys substitute as plain substrings
    since they're already specific enough not to collide.
    """
    if not text:
        return text
    keys = sorted(var_map.keys(), key=len, reverse=True)
    for k in keys:
        v = var_map[k]
        text = re.sub(rf"\b{re.escape(k)}\b", v, text) if len(k) <= 4 else text.replace(k, v)
    return text


def _substitute_in_value(value: Any, var_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        return apply_substitutions(value, var_map)
    if isinstance(value, list):
        return [_substitute_in_value(v, var_map) for v in value]
    if isinstance(value, dict):
        return {k: _substitute_in_value(v, var_map) for k, v in value.items()}
    return value


def vary_record(rec: dict[str, Any], variant_idx: int) -> dict[str, Any]:
    """Return a new trajectory record with substitutions applied."""
    var_map = build_variation_map(rec, variant_idx)
    out: dict[str, Any] = json.loads(json.dumps(rec, default=str))  # deep copy via JSON roundtrip

    out["trajectory_id"] = str(uuid.uuid4())
    out["system_prompt"] = apply_substitutions(out.get("system_prompt", ""), var_map)
    out["goal"] = apply_substitutions(out.get("goal", ""), var_map)

    # Walk messages; substitute string fields and tool_call arg JSON.
    for m in out.get("messages", []):
        if "content" in m and isinstance(m["content"], str):
            m["content"] = apply_substitutions(m["content"], var_map)
        for tc in m.get("tool_calls") or []:
            args_raw = tc.get("function", {}).get("arguments") or "{}"
            try:
                args = json.loads(args_raw)
                args = _substitute_in_value(args, var_map)
                tc["function"]["arguments"] = json.dumps(args)
            except json.JSONDecodeError:
                tc["function"]["arguments"] = apply_substitutions(args_raw, var_map)

    # Provenance update.
    src_id = rec["trajectory_id"]
    out["provenance"] = {
        **rec["provenance"],
        "source": "eval_bench_variation",
        "variation_of": src_id,
        "variation_idx": variant_idx,
        "variation_map": var_map,
        "review_status": rec["provenance"].get("review_status", "unreviewed"),
    }
    return out


# ---------- entry point ----------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument(
        "--variants",
        type=int,
        default=5,
        help="How many variants to produce per source trajectory (default 5).",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    source_records = [
        json.loads(line) for line in args.in_path.read_text().splitlines() if line.strip()
    ]

    out_records: list[dict[str, Any]] = []
    for rec in source_records:
        for i in range(args.variants):
            out_records.append(vary_record(rec, variant_idx=i))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, default=str) + "\n")

    print(
        f"Read {len(source_records)} source trajectories from {args.in_path}",
        file=sys.stderr,
    )
    print(
        f"Wrote {len(out_records)} variant trajectories to {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
