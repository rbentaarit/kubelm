"""Argument-schema validator for recorded trajectories.

Reads the tool_call events emitted by the assistant and checks each one
against the tool's advertised JSON Schema (`inputSchema` from MCP
`tools/list`). Produces two of PROJECT.md's primary metrics as distinct
counts:

  - tool-name hallucinations: calls whose `name` is not in the catalog
  - argument hallucinations:  known tool, arguments fail schema

A name-hallucinated call is *not* also counted as an argument
hallucination — we have no schema to validate against, so the failure
mode is unambiguous.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


@dataclass
class ToolCallValidation:
    step: int
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    name_known: bool
    schema_errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.name_known and not self.schema_errors


@dataclass
class TrajectorySchemaReport:
    calls: list[ToolCallValidation] = field(default_factory=list)

    @property
    def total_calls(self) -> int:
        return len(self.calls)

    @property
    def name_hallucinations(self) -> int:
        return sum(1 for c in self.calls if not c.name_known)

    @property
    def argument_hallucinations(self) -> int:
        return sum(1 for c in self.calls if c.name_known and c.schema_errors)

    @property
    def valid_calls(self) -> int:
        return sum(1 for c in self.calls if c.valid)


def _format_error(err: ValidationError) -> str:
    path = "/".join(str(p) for p in err.absolute_path) or "<root>"
    return f"{path}: {err.message}"


def _validate_one(
    name: str,
    arguments: dict[str, Any],
    schemas: Mapping[str, dict[str, Any]],
) -> tuple[bool, list[str]]:
    if name not in schemas:
        return False, []
    schema = schemas[name] or {}
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(arguments), key=lambda e: list(e.absolute_path))
    return True, [_format_error(e) for e in errors]


def validate_trajectory(
    events: Iterable[Mapping[str, Any]],
    schemas: Mapping[str, dict[str, Any]],
) -> TrajectorySchemaReport:
    report = TrajectorySchemaReport()
    for event in events:
        if event.get("kind") != "assistant":
            continue
        for call in event.get("tool_calls") or []:
            name = call.get("name", "")
            arguments = call.get("arguments") or {}
            name_known, errors = _validate_one(name, arguments, schemas)
            report.calls.append(
                ToolCallValidation(
                    step=event.get("step", -1),
                    tool_call_id=call.get("id", ""),
                    name=name,
                    arguments=arguments,
                    name_known=name_known,
                    schema_errors=errors,
                )
            )
    return report
