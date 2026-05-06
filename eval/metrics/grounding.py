"""Rule-based grounding analyzer (v0.1).

Checks whether the final assistant conclusion makes assertions about
cluster state that are grounded in either the user's goal or some prior
tool_result. Verbatim substring match (lowercased, whitespace-collapsed)
— no paraphrase detection.

v0.1 limitations (real grounding failures this won't catch):
  - paraphrase ("killed for memory" vs `OOMKilled`)
  - negation ("there are no events" vs an empty events array)
  - quantity claims ("3 pods" vs an actual count)
  - causal claims ("the pod is failing because X")
  - exit codes (the number rarely appears verbatim — model paraphrases
    `exitCode: 137` as "exit code 137")

These are scoped for the v0.2 LLM-judge variant in ROADMAP.md.

False positives are bounded by NON_FACT_HYPHENATED — English compound
hyphenations the model uses descriptively, not as cluster identifiers.
Extend as real trajectories surface new ones.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

# K8s status reasons a model might assert verbatim. Maintained manually;
# extend when new ones surface in real trajectories.
STATUS_REASONS: frozenset[str] = frozenset(
    {
        "CrashLoopBackOff",
        "ImagePullBackOff",
        "ErrImagePull",
        "OOMKilled",
        "Evicted",
        "Pending",
        "ContainerCreating",
        "Running",
        "Succeeded",
        "Failed",
        "Terminating",
        "Unknown",
        "Completed",
        "BackOff",
        "DeadlineExceeded",
        "NodeLost",
        "NodeNotReady",
        "InvalidImageName",
        "CreateContainerConfigError",
        "RunContainerError",
        "PostStartHookError",
        "PreStopHookError",
    }
)

# English compound hyphenations a model uses descriptively, not as
# cluster identifiers. Filtered out before grounding checks to avoid
# false-positive ungrounded reports.
NON_FACT_HYPHENATED: frozenset[str] = frozenset(
    {
        "in-memory",
        "real-time",
        "cross-cluster",
        "out-of-memory",
        "long-running",
        "non-existent",
        "well-known",
        "high-availability",
        "load-balancing",
        "read-only",
        "write-once",
    }
)

_KEBAB_RE = re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+)+\b")
_IMAGE_RE = re.compile(r"\b[a-z0-9][a-z0-9._/\-]*:[a-z0-9][a-z0-9._\-]*\b")
_BACKTICK_RE = re.compile(r"`([^`\n]+)`")
_DQUOTE_RE = re.compile(r'"([^"\n]+)"')
_WS_RE = re.compile(r"\s+")


@dataclass
class FactCheck:
    fact: str
    grounded: bool
    source: str | None = None


@dataclass
class GroundingReport:
    conclusion_text: str
    facts: list[FactCheck] = field(default_factory=list)

    @property
    def total_facts(self) -> int:
        return len(self.facts)

    @property
    def ungrounded_facts(self) -> int:
        return sum(1 for f in self.facts if not f.grounded)

    @property
    def has_grounding_failure(self) -> bool:
        return self.ungrounded_facts > 0


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text.lower()).strip()


def _stringify(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _extract_facts(text: str) -> list[str]:
    facts: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        v = value.strip()
        if len(v) <= 2:
            return
        key = v.lower()
        if key in seen or key in NON_FACT_HYPHENATED:
            return
        seen.add(key)
        facts.append(v)

    for m in _KEBAB_RE.finditer(text):
        _add(m.group(0))
    for m in _IMAGE_RE.finditer(text):
        _add(m.group(0))
    for reason in STATUS_REASONS:
        if re.search(rf"\b{re.escape(reason)}\b", text):
            _add(reason)
    for m in _BACKTICK_RE.finditer(text):
        _add(m.group(1))
    for m in _DQUOTE_RE.finditer(text):
        _add(m.group(1))
    return facts


def _last_assistant_text(events: list[Mapping[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("kind") == "assistant":
            return event.get("text") or ""
    return ""


def analyze_grounding(events: Iterable[Mapping[str, Any]]) -> GroundingReport:
    events_list = list(events)
    conclusion = _last_assistant_text(events_list)
    if not conclusion.strip():
        return GroundingReport(conclusion_text="")

    sources: list[tuple[str, str]] = []
    for event in events_list:
        kind = event.get("kind")
        if kind == "meta":
            goal = event.get("goal") or ""
            if goal:
                sources.append(("goal", _normalize(goal)))
        elif kind == "tool_result":
            step = event.get("step", -1)
            body = _normalize(_stringify(event.get("content")))
            sources.append((f"tool_result:step={step}", body))

    report = GroundingReport(conclusion_text=conclusion)
    for fact in _extract_facts(conclusion):
        needle = _normalize(fact)
        source = next((label for label, body in sources if needle in body), None)
        report.facts.append(FactCheck(fact=fact, grounded=source is not None, source=source))
    return report
