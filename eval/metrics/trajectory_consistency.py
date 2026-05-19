"""Trajectory consistency analyzer (v0.1).

Detects "agent-narrative hallucination" — the failure mode where the
final assistant conclusion claims to have *done* something (called a
specific MCP tool, checked some specific data source) but the trajectory
log doesn't show the corresponding tool call.

Distinct from the grounding metric, which checks whether *facts cited*
in the conclusion appear in tool *results*. This one checks whether
*actions claimed* in the conclusion appear in tool *calls*.

v0.1 scope:
  - Strong-precision claim-detection patterns only. We catch the obvious
    cases ("events show", "the analyzer reported", "I checked the pod
    spec") and ignore the rest. False positives are worse than false
    negatives here — a noisy metric becomes ignored.
  - Each pattern maps to a *set* of acceptable tool names. K8sGPT MCP's
    `analyze` tool internally calls events+resources lookups, so a
    claim like "events show CrashLoopBackOff" is consistent if EITHER
    `list-events` OR `analyze` appears in the trajectory. Without
    this fan-out we'd false-positive every gpt-5.4 trajectory (heavy
    analyze-user).
  - A claim with no matching tool kind in the trajectory is flagged as
    inconsistent and the report's `passed` flips to False.

Out of scope for v0.1 (real narrative-hallucination patterns this
won't catch, slated for v0.2):
  - Sloppy phrasings that don't match any pattern ("the cluster's
    state suggests...", "looking at the data...")
  - Claims about *which* resource was inspected when the model called
    get-resource on a different one (cross-name verification needs the
    args, not just the call kind)
  - Implicit claims via causal phrasing ("the pod is failing because
    its image isn't pullable" — implies the model fetched the image
    field, but no surface marker says so)
  - Loop / repetition behavior (model issues the same call twice
    without learning — visible in trajectory but no metric flags it)

Public API mirrors grounding.py + reference_calls.py: a dataclass
`TrajectoryConsistencyReport` and an analyzer function
`analyze_trajectory_consistency(events)` that returns it. Wired in via
`eval/runner/results.py` and aggregated by `eval/scenarios/bench.py`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

# Resource kinds that show up in K8sGPT MCP's `list-resources` /
# `get-resource` tools. Keep singular forms here; plural detection is
# handled by adding `s|es` in the regex. Sourced from the K8sGPT
# v0.4.32 `list-resources` resourceType enum (see CLAUDE.md).
RESOURCE_KINDS: frozenset[str] = frozenset(
    {
        "pod",
        "deployment",
        "service",
        "configmap",
        "secret",
        "cronjob",
        "daemonset",
        "statefulset",
        "replicaset",
        "job",
        "node",
        "persistentvolume",
        "persistentvolumeclaim",
        "ingress",
    }
)

_RESOURCE_ALTERNATION = "|".join(sorted(RESOURCE_KINDS, key=len, reverse=True))


@dataclass(frozen=True)
class _ClaimPattern:
    """A regex that detects a claim + the tool kinds that would
    satisfy it.

    `name` is for error messages and report output (so a human
    reading the inconsistent-claims list can tell which heuristic
    fired). `accepts` is the set of MCP tool names that, if any
    of them appears in the trajectory, makes the claim consistent.
    """

    name: str
    pattern: re.Pattern[str]
    accepts: frozenset[str]


_CLAIM_PATTERNS: tuple[_ClaimPattern, ...] = (
    # Events-related claims. The K8sGPT analyzer internally consults
    # events, so `analyze` satisfies them too.
    _ClaimPattern(
        name="events_show",
        pattern=re.compile(
            r"\bevents? (?:show|showed|reveal|reveals|indicate|indicated|confirm|confirmed)\b",
            re.IGNORECASE,
        ),
        accepts=frozenset({"list-events", "analyze"}),
    ),
    _ClaimPattern(
        name="based_on_events",
        pattern=re.compile(r"\bbased on (?:the )?events\b", re.IGNORECASE),
        accepts=frozenset({"list-events", "analyze"}),
    ),
    _ClaimPattern(
        name="checked_events",
        pattern=re.compile(
            r"\b(?:I |we )?(?:checked|looked at|reviewed) (?:the )?events\b",
            re.IGNORECASE,
        ),
        accepts=frozenset({"list-events", "analyze"}),
    ),
    # Analyzer claims — must have been `analyze`.
    _ClaimPattern(
        name="analyzer_reported",
        pattern=re.compile(
            r"\bthe analyzer (?:said|reported|flagged|found|indicated)\b",
            re.IGNORECASE,
        ),
        accepts=frozenset({"analyze"}),
    ),
    _ClaimPattern(
        name="ran_analyzer",
        pattern=re.compile(
            r"\b(?:I |we )?ran (?:the )?analyzer?\b",
            re.IGNORECASE,
        ),
        accepts=frozenset({"analyze"}),
    ),
    # Resource-inspection claims. Either get-resource (single) or
    # list-resources (set) would satisfy.
    _ClaimPattern(
        name="examined_resource",
        pattern=re.compile(
            rf"\b(?:I |we )?(?:fetched|examined|inspected|retrieved) "
            rf"(?:the )?(?:{_RESOURCE_ALTERNATION})(?:s|es)?(?: spec)?\b",
            re.IGNORECASE,
        ),
        accepts=frozenset({"get-resource", "list-resources"}),
    ),
    # Listing claims — must be list-resources specifically.
    _ClaimPattern(
        name="listed_resources",
        pattern=re.compile(
            rf"\b(?:I |we )?listed (?:all )?(?:the )?(?:{_RESOURCE_ALTERNATION})(?:s|es)\b",
            re.IGNORECASE,
        ),
        accepts=frozenset({"list-resources"}),
    ),
    # Describe-style claims.
    _ClaimPattern(
        name="described_resource",
        pattern=re.compile(
            rf"\b(?:I |we )?described (?:the )?(?:{_RESOURCE_ALTERNATION})\b",
            re.IGNORECASE,
        ),
        accepts=frozenset({"describe-resource"}),
    ),
    # Explicit tool-name mention — "from the analyze output", etc.
    # Captures the tool name so we can require that exact tool.
    _ClaimPattern(
        name="from_tool_output",
        pattern=re.compile(
            r"\bfrom (?:the )?(analyze|list-events|get-resource|list-resources|describe-resource)"
            r" (?:output|result|response|call)\b",
            re.IGNORECASE,
        ),
        accepts=frozenset({"__capture__"}),  # special: handled below
    ),
)


@dataclass
class ClaimMatch:
    """One detected self-reference claim and whether the trajectory
    backs it up."""

    pattern_name: str
    matched_text: str  # the exact regex match from the conclusion
    accepted_tools: frozenset[str]
    consistent: bool
    actual_calls_seen: tuple[str, ...]  # tool names that DID appear in trajectory


@dataclass
class TrajectoryConsistencyReport:
    total_claims: int
    consistent_claims: int
    inconsistent_claims: list[ClaimMatch] = field(default_factory=list)
    passed: bool = True


def _last_assistant_text(events: Iterable[Mapping[str, Any]]) -> str:
    """The last assistant turn with non-empty text is the conclusion.

    Same convention as grounding.py — keep them in lockstep so a
    trajectory's "conclusion" is one well-defined string across all
    metrics.
    """
    last = ""
    for e in events:
        if e.get("kind") == "assistant" and e.get("text"):
            last = e["text"]
    return last


def _called_tool_names(events: Iterable[Mapping[str, Any]]) -> frozenset[str]:
    """The set of distinct tool names actually invoked.

    We only count *outgoing* tool calls from the model, not
    `tool_result` events (which would double-count for paired
    call/result entries). Excludes erroring tool_result responses
    from satisfying claims — if the analyzer call errored, the model
    didn't *actually* see analyzer output, so a "the analyzer said"
    claim against an errored analyze should still fail.
    """
    erroring_call_ids: set[str] = set()
    for e in events:
        if e.get("kind") == "tool_result" and e.get("is_error"):
            cid = e.get("tool_call_id")
            if cid:
                erroring_call_ids.add(cid)

    names: set[str] = set()
    for e in events:
        if e.get("kind") != "assistant":
            continue
        for call in e.get("tool_calls") or []:
            name = call.get("name")
            cid = call.get("id")
            if name and cid not in erroring_call_ids:
                names.add(name)
    return frozenset(names)


def analyze_trajectory_consistency(
    events: Iterable[Mapping[str, Any]],
) -> TrajectoryConsistencyReport:
    """Score the trajectory for narrative consistency.

    Walks the conclusion text, finds every claim that matches a
    detection pattern, and verifies the trajectory contains a tool
    call of an acceptable kind for each claim.
    """
    events_list = list(events)
    conclusion = _last_assistant_text(events_list)
    called = _called_tool_names(events_list)

    claims: list[ClaimMatch] = []
    for pat in _CLAIM_PATTERNS:
        for m in pat.pattern.finditer(conclusion):
            if pat.accepts == frozenset({"__capture__"}):
                # Capture group is the exact tool name required.
                required = frozenset({m.group(1).lower()})
            else:
                required = pat.accepts

            actual = required & called
            consistent = bool(actual)
            claims.append(
                ClaimMatch(
                    pattern_name=pat.name,
                    matched_text=m.group(0),
                    accepted_tools=required,
                    consistent=consistent,
                    actual_calls_seen=tuple(sorted(actual)),
                )
            )

    inconsistent = [c for c in claims if not c.consistent]
    return TrajectoryConsistencyReport(
        total_claims=len(claims),
        consistent_claims=len(claims) - len(inconsistent),
        inconsistent_claims=inconsistent,
        passed=not inconsistent,
    )
