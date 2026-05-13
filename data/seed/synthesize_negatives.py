"""Synthesize negative-example trajectories from clean seeds.

Per ROADMAP Phase 4 construction step 3, negative examples teach the
model to recover from a bad first call. v0 covers two injection
patterns; both surgically transplant a wrong call + K8sGPT-shape
error + brief recovery message at the front of an already-correct
trajectory.

Pattern A — wrong `resourceType` on `list-resources`:
  Pre-call:  list-resources(resourceType="<typo>", namespace=...)
  Result:    {"isError": true, "content":[{"type":"text",
              "text":"unsupported resource type: <typo>. Supported
              types: [...]"}]}
  Recovery:  short assistant message acknowledging the typo, then
              the original (correct) trajectory begins.

  Applies to any seed whose first list-resources call uses one of
  the supported plural resourceTypes; we corrupt the value with a
  plausible typo (`pod` → `pod_name`, `services` → `service_list`,
  etc.).

Pattern B — hallucinated tool name:
  Pre-call:  get-pod(name=..., namespace=...)
            (a plausible-looking but nonexistent tool name; the real
             tool is `get-resource`)
  Result:    {"isError": true, "content":[{"type":"text",
              "text":"unknown tool: get-pod"}]}
  Recovery:  short assistant message, then the original trajectory.

  Applies to every seed: a generic "hallucinated tool" call doesn't
  depend on scenario specifics.

The K8sGPT error string for Pattern A is reproduced verbatim from
what we observed in the network-policy drill-in (2026-05-12 audit) —
not invented. The Pattern B error string is a plausible MCP-level
"unknown tool" response. If a future audit shows K8sGPT phrases this
differently, regenerate.

Negatives are deterministic given (scenario_id, pattern). Output
provenance: source=negative_synthetic, with the pattern name in
provenance.negative_pattern.

Usage:
    uv run python data/seed/synthesize_negatives.py \
        --in data/seed/v0/gpt-5.4-2026-05-12.jsonl \
        --out data/seed/varied/v0/negatives-2026-05-12.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

K8SGPT_SUPPORTED_TYPES = (
    "ingress persistentvolumeclaim persistentvolume pod deployment service "
    "cronjob daemonset configmap secret node job statefulset replicaset"
)

# Map a real resourceType to a plausible typo a small model might emit.
RESOURCE_TYPE_TYPOS: dict[str, str] = {
    "pods": "pod_name",
    "deployments": "deployment_list",
    "services": "service_list",
    "cronjobs": "cron_jobs",
    "jobs": "job_list",
    "persistentvolumeclaims": "pvcs",
    "persistentvolumes": "pvs",
    "resourcequotas": "quotas",
    "horizontalpodautoscalers": "hpas",
    "nodes": "node_list",
    "networkpolicies": "netpolicies",
}


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _make_tool_result_content(text: str) -> str:
    return json.dumps({"content": [{"type": "text", "text": text}], "isError": True})


def _find_first_list_resources_call(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first list-resources tool_call dict found, or None."""
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            if tc.get("function", {}).get("name") == "list-resources":
                return tc
    return None


def pattern_a_wrong_resource_type(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Inject a wrong-resourceType pre-call before the trajectory's first list-resources."""
    target = _find_first_list_resources_call(rec["messages"])
    if target is None:
        return None
    args = json.loads(target["function"]["arguments"] or "{}")
    real_rt = args.get("resourceType")
    if not real_rt:
        return None
    typo = RESOURCE_TYPE_TYPOS.get(real_rt)
    if not typo:
        return None

    bad_args = dict(args)
    bad_args["resourceType"] = typo
    bad_call_id = _new_call_id()

    bad_assistant: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": bad_call_id,
                "type": "function",
                "function": {
                    "name": "list-resources",
                    "arguments": json.dumps(bad_args),
                },
            }
        ],
    }
    bad_tool_result: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": bad_call_id,
        "content": _make_tool_result_content(
            f"unsupported resource type: {typo}. Supported types: [{K8SGPT_SUPPORTED_TYPES}]"
        ),
    }
    recovery: dict[str, Any] = {
        "role": "assistant",
        "content": (
            f"That resourceType isn't supported. The right value for this lookup is "
            f"`{real_rt}` — retrying with the correct type."
        ),
    }

    out = json.loads(json.dumps(rec, default=str))  # deep copy
    msgs: list[dict[str, Any]] = out["messages"]
    # Insert before the first assistant turn that has tool_calls (right after
    # the leading system + user messages and any prior assistant text).
    insertion_idx = next(
        (i for i, m in enumerate(msgs) if m.get("role") == "assistant" and m.get("tool_calls")),
        2,
    )
    msgs[insertion_idx:insertion_idx] = [bad_assistant, bad_tool_result, recovery]

    out["trajectory_id"] = str(uuid.uuid4())
    out["provenance"] = {
        **rec["provenance"],
        "source": "negative_synthetic",
        "negative_pattern": "wrong_resource_type",
        "negative_of": rec["trajectory_id"],
        "review_status": "unreviewed",  # synthetic; needs human eye before training
    }
    out["quality"] = {
        **rec["quality"],
        "step_count": rec["quality"]["step_count"] + 1,
    }
    return out


def pattern_b_hallucinated_tool(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Inject a hallucinated tool-name pre-call before the first real tool call."""
    msgs = rec["messages"]
    target = next(
        (m for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")),
        None,
    )
    if target is None:
        return None
    first_tc = target["tool_calls"][0]
    args = json.loads(first_tc["function"]["arguments"] or "{}")
    ns = args.get("namespace", "")
    # Pick a "natural-looking" hallucinated name. `get-pod` is plausible — it's
    # the rough shape of `get-resource` but with the type fused into the name,
    # which is a common small-model error mode.
    bad_name = "get-pod"
    bad_args = {"namespace": ns} if ns else {}
    bad_call_id = _new_call_id()

    bad_assistant: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": bad_call_id,
                "type": "function",
                "function": {
                    "name": bad_name,
                    "arguments": json.dumps(bad_args),
                },
            }
        ],
    }
    bad_tool_result: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": bad_call_id,
        "content": _make_tool_result_content(f"unknown tool: {bad_name}"),
    }
    recovery: dict[str, Any] = {
        "role": "assistant",
        "content": (
            f"`{bad_name}` isn't a tool on this MCP surface. The right way to inspect "
            f"a pod is `get-resource` with `resourceType: pod`. Continuing with the "
            f"intended investigation."
        ),
    }

    out = json.loads(json.dumps(rec, default=str))
    out_msgs: list[dict[str, Any]] = out["messages"]
    insertion_idx = next(
        (i for i, m in enumerate(out_msgs) if m.get("role") == "assistant" and m.get("tool_calls")),
        2,
    )
    out_msgs[insertion_idx:insertion_idx] = [bad_assistant, bad_tool_result, recovery]

    out["trajectory_id"] = str(uuid.uuid4())
    out["provenance"] = {
        **rec["provenance"],
        "source": "negative_synthetic",
        "negative_pattern": "hallucinated_tool_name",
        "negative_of": rec["trajectory_id"],
        "review_status": "unreviewed",
    }
    out["quality"] = {
        **rec["quality"],
        "step_count": rec["quality"]["step_count"] + 1,
    }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    source_records = [
        json.loads(line) for line in args.in_path.read_text().splitlines() if line.strip()
    ]

    out_records: list[dict[str, Any]] = []
    skipped_a = 0
    skipped_b = 0
    for rec in source_records:
        a = pattern_a_wrong_resource_type(rec)
        if a is not None:
            out_records.append(a)
        else:
            skipped_a += 1
        b = pattern_b_hallucinated_tool(rec)
        if b is not None:
            out_records.append(b)
        else:
            skipped_b += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, default=str) + "\n")

    print(f"Read {len(source_records)} source trajectories from {args.in_path}", file=sys.stderr)
    print(f"Wrote {len(out_records)} negative trajectories to {args.out}", file=sys.stderr)
    print(
        f"  pattern A (wrong resource type): {len(source_records) - skipped_a} produced, "
        f"{skipped_a} skipped (no list-resources call or no typo defined)",
        file=sys.stderr,
    )
    print(
        f"  pattern B (hallucinated tool name): {len(source_records) - skipped_b} produced, "
        f"{skipped_b} skipped (no tool_calls in source)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
