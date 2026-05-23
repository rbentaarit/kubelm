"""Scenario YAML format: dataclasses + loader.

A scenario is one YAML file describing a failing K8s cluster state and
the investigation that's expected to surface from a competent SRE
working through K8sGPT's MCP tools. The runner (slice 2.4) consumes
these to drive end-to-end benchmark runs; the metrics modules (slice
2.5) consume the `expected` block to grade the model's behavior.

This module is pure data + validation. It deliberately knows nothing
about kind, kubectl, or how scenarios are executed — that boundary is
what lets Phase 4 ingest the same files for training-data construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SETTLE_TIMEOUT = 60


@dataclass
class WaitForStatus:
    """A single settle condition the runner waits for after setup."""

    kind: str
    namespace: str
    name: str
    reason: str | None = None
    phase: str | None = None
    condition: str | None = None
    message_contains: str | None = None
    timeout_seconds: int = DEFAULT_SETTLE_TIMEOUT


@dataclass
class SetupStep:
    """One setup action: exactly one of apply_inline or apply_file is set."""

    apply_inline: str | None = None
    apply_file: str | None = None


@dataclass
class ReferenceCall:
    name: str
    args_match: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceCalls:
    must_include: list[ReferenceCall] = field(default_factory=list)
    any_of: list[ReferenceCall] = field(default_factory=list)
    forbidden: list[ReferenceCall] = field(default_factory=list)


@dataclass
class ConclusionRubric:
    """Substring rubric for the final assistant text.

    ``must_mention`` is a list of *slots*. Each slot is either:

      - a string: the exact (case-insensitive) substring must appear
      - a list of strings: at least one of them must appear
        (synonym set — captures phrasings like "Pending" vs.
        "unschedulable" that mean the same thing for grading)

    ``must_not_mention`` is a flat list of strings that may not appear.
    """

    must_mention: list[str | list[str]] = field(default_factory=list)
    must_not_mention: list[str] = field(default_factory=list)
    semantic_intent: str = ""


@dataclass
class ExpectedOutcome:
    reference_calls: ReferenceCalls = field(default_factory=ReferenceCalls)
    conclusion_rubric: ConclusionRubric = field(default_factory=ConclusionRubric)


@dataclass
class Scenario:
    id: str
    profile: str
    description: str = ""
    goal: str = ""
    setup: list[SetupStep] = field(default_factory=list)
    settle: list[WaitForStatus] = field(default_factory=list)
    expected: ExpectedOutcome = field(default_factory=ExpectedOutcome)
    source_path: Path | None = None


class ScenarioParseError(ValueError):
    """Raised when a scenario YAML is structurally invalid."""


def _parse_duration(value: int | str | None, *, default: int) -> int:
    """Accept ``60``, ``"60"``, ``"60s"``, ``"5m"``, ``"1h"``."""
    if value is None:
        return default
    if isinstance(value, int):
        return value
    s = str(value).strip().lower()
    if not s:
        return default
    multipliers = {"s": 1, "m": 60, "h": 3600}
    if s[-1] in multipliers:
        return int(s[:-1]) * multipliers[s[-1]]
    return int(s)


def _require(data: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in data:
        raise ScenarioParseError(f"{ctx}: missing required field {key!r}")
    return data[key]


def _parse_setup_step(raw: dict[str, Any], ctx: str) -> SetupStep:
    inline = raw.get("apply_inline")
    path = raw.get("apply_file")
    if (inline is None) == (path is None):
        raise ScenarioParseError(
            f"{ctx}: each setup step must set exactly one of 'apply_inline' or 'apply_file'"
        )
    return SetupStep(apply_inline=inline, apply_file=path)


def _parse_settle_step(raw: dict[str, Any], ctx: str) -> WaitForStatus:
    if "wait_for_status" not in raw:
        raise ScenarioParseError(f"{ctx}: only 'wait_for_status' is supported in v0.1")
    body = raw["wait_for_status"]
    if not isinstance(body, dict):
        raise ScenarioParseError(f"{ctx}: 'wait_for_status' must be a mapping")
    return WaitForStatus(
        kind=_require(body, "kind", ctx),
        namespace=_require(body, "namespace", ctx),
        name=_require(body, "name", ctx),
        reason=body.get("reason"),
        phase=body.get("phase"),
        condition=body.get("condition"),
        message_contains=body.get("message_contains"),
        timeout_seconds=_parse_duration(body.get("timeout"), default=DEFAULT_SETTLE_TIMEOUT),
    )


def _parse_reference_call(raw: dict[str, Any], ctx: str) -> ReferenceCall:
    return ReferenceCall(
        name=_require(raw, "name", ctx),
        args_match=raw.get("args_match") or {},
    )


def _parse_expected(raw: dict[str, Any], ctx: str) -> ExpectedOutcome:
    rc_raw = raw.get("reference_calls") or {}
    cr_raw = raw.get("conclusion_rubric") or {}
    return ExpectedOutcome(
        reference_calls=ReferenceCalls(
            must_include=[
                _parse_reference_call(c, f"{ctx}.reference_calls.must_include[{i}]")
                for i, c in enumerate(rc_raw.get("must_include") or [])
            ],
            any_of=[
                _parse_reference_call(c, f"{ctx}.reference_calls.any_of[{i}]")
                for i, c in enumerate(rc_raw.get("any_of") or [])
            ],
            forbidden=[
                _parse_reference_call(c, f"{ctx}.reference_calls.forbidden[{i}]")
                for i, c in enumerate(rc_raw.get("forbidden") or [])
            ],
        ),
        conclusion_rubric=ConclusionRubric(
            must_mention=[
                item if isinstance(item, str) else list(item)
                for item in (cr_raw.get("must_mention") or [])
            ],
            must_not_mention=list(cr_raw.get("must_not_mention") or []),
            semantic_intent=cr_raw.get("semantic_intent") or "",
        ),
    )


def load_scenario(path: Path | str) -> Scenario:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScenarioParseError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ScenarioParseError(f"{path}: top level must be a mapping")
    ctx = str(path)
    return Scenario(
        id=_require(raw, "id", ctx),
        profile=_require(raw, "profile", ctx),
        description=raw.get("description") or "",
        goal=_require(raw, "goal", ctx),
        setup=[
            _parse_setup_step(s, f"{ctx}.setup[{i}]") for i, s in enumerate(raw.get("setup") or [])
        ],
        settle=[
            _parse_settle_step(s, f"{ctx}.settle[{i}]")
            for i, s in enumerate(raw.get("settle") or [])
        ],
        expected=_parse_expected(raw.get("expected") or {}, ctx),
        source_path=path,
    )


def load_scenarios(directory: Path | str) -> list[Scenario]:
    directory = Path(directory)
    if not directory.is_dir():
        raise ScenarioParseError(f"{directory}: not a directory")
    scenarios = [load_scenario(p) for p in sorted(directory.glob("*.yaml"))]
    seen_ids: set[str] = set()
    for scn in scenarios:
        if scn.id in seen_ids:
            raise ScenarioParseError(f"{scn.source_path}: duplicate scenario id {scn.id!r}")
        seen_ids.add(scn.id)
    return scenarios
