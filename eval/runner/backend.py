"""Model-backend interface for the eval runner.

A Backend accepts OpenAI-style chat messages plus the MCP tool catalog
and returns one AssistantTurn (text + structured tool calls + latency).
The run loop is backend-agnostic; concrete implementations translate
to/from their underlying API as needed.

MockBackend lives here intentionally. Methodology #2 in PROJECT.md
forbids mocking the *MCP surface* (the thing under evaluation), not the
model — which is what we're evaluating, not what we're depending on.
A scripted backend lets the harness validate its own plumbing without a
network or a model in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from eval.client import Tool
from eval.trajectory import ToolCall


@dataclass
class AssistantTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    latency_ms: float = 0.0
    raw: dict[str, Any] | None = None


class Backend(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool],
    ) -> AssistantTurn: ...


@dataclass
class MockBackend:
    """Replays a scripted sequence of AssistantTurns. For tests only."""

    script: list[AssistantTurn]
    calls: list[tuple[list[dict[str, Any]], list[Tool]]] = field(default_factory=list, init=False)
    _index: int = field(default=0, init=False)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool],
    ) -> AssistantTurn:
        if self._index >= len(self.script):
            raise RuntimeError(f"MockBackend script exhausted after {self._index} turns")
        self.calls.append((list(messages), list(tools)))
        turn = self.script[self._index]
        self._index += 1
        return turn
