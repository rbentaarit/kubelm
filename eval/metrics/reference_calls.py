"""Reference-call coverage metric.

Walks the recorded tool_call events and checks them against a scenario's
``ReferenceCalls`` expectation:

  - ``must_include``: each matcher must hit at least one recorded call
  - ``any_of``:       at least one matcher must hit a recorded call
                      (when the list is non-empty). Captures "multiple
                      valid investigation paths" — e.g. ``get-logs`` OR
                      ``list-events`` is acceptable for surfacing a
                      pod failure cause.
  - ``forbidden``:    no matcher may hit any recorded call

A call matches a matcher when ALL of:
  - the recorded ``name`` equals the matcher's ``name``
  - the recorded ``arguments`` is a superset of the matcher's
    ``args_match`` (every key/value in the matcher must also be in the
    arguments)
  - the corresponding ``tool_result`` event did not carry
    ``is_error: true``. A call that the MCP server rejected
    (unsupported resource type, schema error, server fault) is not a
    successful reference call even when the arguments looked right —
    the model didn't actually obtain the evidence.

``forbidden`` matchers still hit on errored calls: if a scenario
declares a tool call forbidden, the model attempting it is the
violation, regardless of whether the server accepted it.
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
    any_of: list[ReferenceCallMatch] = field(default_factory=list)
    forbidden: list[ReferenceCallMatch] = field(default_factory=list)

    @property
    def must_include_hits(self) -> int:
        return sum(1 for m in self.must_include if m.matched)

    @property
    def must_include_misses(self) -> int:
        return sum(1 for m in self.must_include if not m.matched)

    @property
    def any_of_hits(self) -> int:
        return sum(1 for m in self.any_of if m.matched)

    @property
    def any_of_satisfied(self) -> bool:
        return not self.any_of or self.any_of_hits >= 1

    @property
    def forbidden_hits(self) -> int:
        return sum(1 for m in self.forbidden if m.matched)

    @property
    def passed(self) -> bool:
        return self.must_include_misses == 0 and self.any_of_satisfied and self.forbidden_hits == 0


def _is_subset(matcher: Mapping[str, Any], arguments: Mapping[str, Any]) -> bool:
    return all(k in arguments and arguments[k] == v for k, v in matcher.items())


def _errored_call_ids(events: list[Mapping[str, Any]]) -> set[str]:
    return {
        str(e["tool_call_id"])
        for e in events
        if e.get("kind") == "tool_result" and e.get("is_error") and e.get("tool_call_id")
    }


def _find_match(
    rc: ReferenceCall,
    events: list[Mapping[str, Any]],
    errored: set[str],
    *,
    skip_errored: bool,
) -> int | None:
    for event in events:
        if event.get("kind") != "assistant":
            continue
        for call in event.get("tool_calls") or []:
            if call.get("name") != rc.name:
                continue
            if skip_errored and call.get("id") in errored:
                continue
            if _is_subset(rc.args_match or {}, call.get("arguments") or {}):
                return event.get("step", -1)
    return None


def evaluate_reference_calls(
    events: Iterable[Mapping[str, Any]],
    expected: ReferenceCalls,
) -> ReferenceCallsReport:
    events_list = list(events)
    errored = _errored_call_ids(events_list)
    report = ReferenceCallsReport()
    for rc in expected.must_include:
        step = _find_match(rc, events_list, errored, skip_errored=True)
        report.must_include.append(
            ReferenceCallMatch(
                name=rc.name,
                args_match=dict(rc.args_match),
                matched=step is not None,
                matched_step=step,
            )
        )
    for rc in expected.any_of:
        step = _find_match(rc, events_list, errored, skip_errored=True)
        report.any_of.append(
            ReferenceCallMatch(
                name=rc.name,
                args_match=dict(rc.args_match),
                matched=step is not None,
                matched_step=step,
            )
        )
    for rc in expected.forbidden:
        step = _find_match(rc, events_list, errored, skip_errored=False)
        report.forbidden.append(
            ReferenceCallMatch(
                name=rc.name,
                args_match=dict(rc.args_match),
                matched=step is not None,
                matched_step=step,
            )
        )
    return report
