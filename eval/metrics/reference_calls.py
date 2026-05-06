"""Reference-call coverage metric.

Walks the recorded tool_call events and checks them against a scenario's
``ReferenceCalls`` expectation:

  - ``must_include``: each matcher must hit at least one recorded call
  - ``forbidden``:    no matcher may hit any recorded call

A call matches a matcher when the recorded ``name`` equals the matcher's
``name`` AND the recorded ``arguments`` is a superset of the matcher's
``args_match`` (every key/value in the matcher must also be in the
arguments). Extra keys in the recorded arguments are fine — multiple
valid investigation paths exist, and the matcher should describe the
*minimum* a competent investigation includes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from eval.scenarios.spec import ReferenceCall, ReferenceCalls


@dataclass
class ReferenceCallMatch:
    name: str
    args_match: dict[str, Any]
    matched: bool
    matched_step: int | None = None


@dataclass
class ReferenceCallsReport:
    must_include: list[ReferenceCallMatch] = field(default_factory=list)
    forbidden: list[ReferenceCallMatch] = field(default_factory=list)

    @property
    def must_include_hits(self) -> int:
        return sum(1 for m in self.must_include if m.matched)

    @property
    def must_include_misses(self) -> int:
        return sum(1 for m in self.must_include if not m.matched)

    @property
    def forbidden_hits(self) -> int:
        return sum(1 for m in self.forbidden if m.matched)

    @property
    def passed(self) -> bool:
        return self.must_include_misses == 0 and self.forbidden_hits == 0


def _is_subset(matcher: Mapping[str, Any], arguments: Mapping[str, Any]) -> bool:
    return all(k in arguments and arguments[k] == v for k, v in matcher.items())


def _find_match(rc: ReferenceCall, events: list[Mapping[str, Any]]) -> int | None:
    for event in events:
        if event.get("kind") != "assistant":
            continue
        for call in event.get("tool_calls") or []:
            if call.get("name") != rc.name:
                continue
            if _is_subset(rc.args_match or {}, call.get("arguments") or {}):
                return event.get("step", -1)
    return None


def evaluate_reference_calls(
    events: Iterable[Mapping[str, Any]],
    expected: ReferenceCalls,
) -> ReferenceCallsReport:
    events_list = list(events)
    report = ReferenceCallsReport()
    for rc in expected.must_include:
        step = _find_match(rc, events_list)
        report.must_include.append(
            ReferenceCallMatch(
                name=rc.name,
                args_match=dict(rc.args_match),
                matched=step is not None,
                matched_step=step,
            )
        )
    for rc in expected.forbidden:
        step = _find_match(rc, events_list)
        report.forbidden.append(
            ReferenceCallMatch(
                name=rc.name,
                args_match=dict(rc.args_match),
                matched=step is not None,
                matched_step=step,
            )
        )
    return report
