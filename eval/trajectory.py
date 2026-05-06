"""Append-only JSONL recorder for a single model trajectory.

A trajectory file is a JSONL stream:

    {kind: "meta", ...}            # exactly one, first line
    {kind: "assistant", ...}       # zero or more, interleaved
    {kind: "tool_result", ...}     # zero or more, interleaved
    {kind: "error", ...}           # zero or more
    {kind: "end", ...}             # exactly one, last line

This file is the input every Phase 1 metric calculator reads. The format
is deliberately flat and append-friendly: one event per line, machine-
diffable, streamable, easy to grep.
"""

from __future__ import annotations

import json
import uuid
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self, TextIO

from eval import K8SGPT_VERSION, MCP_PROTOCOL_VERSION

SCHEMA_VERSION = 1


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolResult:
    tool_call_id: str
    name: str
    content: Any
    is_error: bool = False


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


@dataclass
class TrajectoryRecorder(AbstractContextManager["TrajectoryRecorder"]):
    path: Path
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model: str = ""
    scenario_id: str = ""
    goal: str = ""
    extra_meta: dict[str, Any] = field(default_factory=dict)
    _fh: TextIO | None = field(default=None, init=False, repr=False)
    _step: int = field(default=0, init=False, repr=False)
    _ended: bool = field(default=False, init=False, repr=False)

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._write(
            {
                "kind": "meta",
                "ts": _now(),
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "model": self.model,
                "scenario_id": self.scenario_id,
                "goal": self.goal,
                "k8sgpt_version": K8SGPT_VERSION,
                "mcp_protocol_version": MCP_PROTOCOL_VERSION,
                **self.extra_meta,
            }
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if not self._ended:
            self.end("error" if exc else "incomplete", str(exc) if exc else None)
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _write(self, event: dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("recorder used outside its context manager")
        self._fh.write(json.dumps(event, separators=(",", ":")) + "\n")
        self._fh.flush()

    def assistant(
        self,
        text: str = "",
        tool_calls: list[ToolCall] | None = None,
        latency_ms: float | None = None,
    ) -> None:
        self._write(
            {
                "kind": "assistant",
                "step": self._step,
                "ts": _now(),
                "latency_ms": latency_ms,
                "text": text,
                "tool_calls": [asdict(c) for c in (tool_calls or [])],
            }
        )
        self._step += 1

    def tool_result(self, result: ToolResult, latency_ms: float | None = None) -> None:
        self._write(
            {
                "kind": "tool_result",
                "step": self._step,
                "ts": _now(),
                "latency_ms": latency_ms,
                **asdict(result),
            }
        )
        self._step += 1

    def error(self, where: str, message: str) -> None:
        self._write(
            {
                "kind": "error",
                "step": self._step,
                "ts": _now(),
                "where": where,
                "message": message,
            }
        )
        self._step += 1

    def end(self, status: str, message: str | None = None) -> None:
        self._write(
            {
                "kind": "end",
                "ts": _now(),
                "status": status,
                "message": message,
                "steps": self._step,
            }
        )
        self._ended = True


def load_trajectory(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
