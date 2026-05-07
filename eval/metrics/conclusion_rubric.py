"""Conclusion rubric metric (v0.1, substring-based).

Checks the *final* assistant text against a scenario's
``ConclusionRubric``:

  - ``must_mention``: each slot must be satisfied (case-insensitive
    substring). A slot is either a string (exact match required) or a
    list of strings (any one of them satisfies the slot — synonym set).
  - ``must_not_mention``: no entry may appear.

``semantic_intent`` is preserved on the report but not auto-evaluated;
v0.2 adds an LLM-judge variant for that, parallel to the grounding
v0.1 -> v0.2 split.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from eval.scenarios.spec import ConclusionRubric


@dataclass
class ConclusionRubricReport:
    conclusion_text: str = ""
    missing_mentions: list[str] = field(default_factory=list)
    forbidden_mentions: list[str] = field(default_factory=list)
    semantic_intent: str = ""

    @property
    def passed(self) -> bool:
        return not self.missing_mentions and not self.forbidden_mentions


def _last_assistant_text(events: Iterable[Mapping[str, object]]) -> str:
    last_text = ""
    for event in events:
        if event.get("kind") == "assistant":
            text = event.get("text") or ""
            last_text = text if isinstance(text, str) else ""
    return last_text


def _slot_satisfied(slot: str | list[str], text_lower: str) -> bool:
    if isinstance(slot, str):
        return slot.lower() in text_lower
    return any(s.lower() in text_lower for s in slot)


def _format_slot(slot: str | list[str]) -> str:
    if isinstance(slot, str):
        return slot
    return " | ".join(slot)


def evaluate_conclusion_rubric(
    events: Iterable[Mapping[str, object]],
    rubric: ConclusionRubric,
) -> ConclusionRubricReport:
    text = _last_assistant_text(events)
    text_lower = text.lower()
    missing = [
        _format_slot(slot) for slot in rubric.must_mention if not _slot_satisfied(slot, text_lower)
    ]
    forbidden_hits = [m for m in rubric.must_not_mention if m.lower() in text_lower]
    return ConclusionRubricReport(
        conclusion_text=text,
        missing_mentions=missing,
        forbidden_mentions=forbidden_hits,
        semantic_intent=rubric.semantic_intent,
    )
