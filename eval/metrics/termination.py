"""Termination classifier (v0.1).

Categorizes a recorded trajectory into one of five labels:

  errored        — explicit error events, or end.status == "error"
  no_conclusion  — no synthesizing assistant turn (last assistant has
                   empty text, or pending tool_calls)
  looping        — same (tool_name, arguments) signature appears at
                   least duplicate_threshold times
  premature      — concluded with fewer than min_successful_tool_calls
                   successful tool_results
  complete       — none of the above

Detection is shape-based and post-hoc: the recorder's `end.status` is preserved
on the report, but the label is derived from the event stream so it's not at
the mercy of the runner's optimism.

Priority ordering when multiple flags fire:
    errored > no_conclusion > looping > premature > complete

v0.1 limitations:
  - "looping" only catches verbatim repetition of (name, arguments). Semantic
    loops (slightly different args, same investigation) need an LLM judge.
  - "premature" only catches the egregious zero-evidence case. Higher
    thresholds would need per-scenario calibration we don't have yet.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

DEFAULT_DUPLICATE_THRESHOLD = 3
DEFAULT_MIN_SUCCESSFUL_TOOL_CALLS = 1


@dataclass
class TerminationReport:
    end_status: str = ""
    has_conclusion: bool = False
    successful_tool_calls: int = 0
    error_events: int = 0
    duplicate_call_groups: list[list[int]] = field(default_factory=list)

    errored: bool = False
    no_conclusion: bool = False
    looping: bool = False
    premature: bool = False

    @property
    def label(self) -> str:
        if self.errored:
            return "errored"
        if self.no_conclusion:
            return "no_conclusion"
        if self.looping:
            return "looping"
        if self.premature:
            return "premature"
        return "complete"

    @property
    def is_failure(self) -> bool:
        return self.label != "complete"


def _is_conclusion(event: Mapping[str, Any]) -> bool:
    text = (event.get("text") or "").strip()
    tool_calls = event.get("tool_calls") or []
    return bool(text) and not tool_calls


def classify_termination(
    events: Iterable[Mapping[str, Any]],
    *,
    duplicate_threshold: int = DEFAULT_DUPLICATE_THRESHOLD,
    min_successful_tool_calls: int = DEFAULT_MIN_SUCCESSFUL_TOOL_CALLS,
) -> TerminationReport:
    events_list = list(events)

    end_status = ""
    error_events = 0
    successful_tool_calls = 0
    last_assistant: Mapping[str, Any] | None = None
    sig_to_steps: dict[tuple[str, str], list[int]] = {}

    for event in events_list:
        kind = event.get("kind")
        if kind == "end":
            end_status = event.get("status") or ""
        elif kind == "error":
            error_events += 1
        elif kind == "tool_result":
            if not event.get("is_error", False):
                successful_tool_calls += 1
        elif kind == "assistant":
            last_assistant = event
            step = event.get("step", -1)
            for call in event.get("tool_calls") or []:
                name = call.get("name", "")
                args_key = json.dumps(call.get("arguments") or {}, sort_keys=True)
                sig_to_steps.setdefault((name, args_key), []).append(step)

    duplicate_groups = [
        steps for steps in sig_to_steps.values() if len(steps) >= duplicate_threshold
    ]
    has_conclusion = last_assistant is not None and _is_conclusion(last_assistant)

    report = TerminationReport(
        end_status=end_status,
        has_conclusion=has_conclusion,
        successful_tool_calls=successful_tool_calls,
        error_events=error_events,
        duplicate_call_groups=duplicate_groups,
    )
    report.errored = end_status == "error" or error_events > 0
    report.no_conclusion = not has_conclusion
    report.looping = len(duplicate_groups) > 0
    report.premature = has_conclusion and successful_tool_calls < min_successful_tool_calls
    return report
