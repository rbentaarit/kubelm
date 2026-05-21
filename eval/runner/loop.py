"""Run loop: drive a Backend through a multi-turn investigation.

Sequence per step:
  1. Backend.chat -> AssistantTurn
  2. Recorder logs the assistant event (text, tool_calls, latency_ms)
  3. If no tool_calls: terminate (assumed conclusion or empty turn)
  4. For each tool_call: invoke call_tool, record tool_result with latency
  5. Append the assistant message and tool results to chat history
  6. Repeat until conclusion or max_steps exhausted

Termination outcomes:
  - Loop body breaks (no tool_calls) -> recorder.end("complete")
  - Loop exhausts max_steps          -> recorder.end("incomplete", "hit step budget ...")
  - Exception escapes                -> recorder.__exit__ writes end("error", ...)

The runner does not categorize trajectories — that's the termination
classifier's job. recorder.end("complete") here just means "the loop
ran to its natural end without exhausting the budget."
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from eval.client import Tool
from eval.runner.backend import Backend
from eval.trajectory import ToolResult, TrajectoryRecorder

DEFAULT_SYSTEM_PROMPT = (
    "You are an SRE investigating a Kubernetes cluster via K8sGPT's MCP tools.\n"
    "Investigate the specific resource named in the question. Use the tools to "
    "gather evidence and trace symptoms to their ROOT CAUSE: when a workload "
    "(Deployment, StatefulSet, Job, or Pod) is unhealthy, inspect the affected "
    "Pods and their container statuses to find WHY — do not stop at a top-level "
    "status such as 'replicas unavailable'.\n"
    "Cite specific cluster state from tool results in your conclusions.\n"
    "Once you have identified the root cause, stop calling tools and write a "
    "concise conclusion that names the failing resource and its root cause."
)


def _serialize_tool_calls(tool_calls: list) -> list[dict[str, Any]]:
    return [
        {
            "id": c.id,
            "type": "function",
            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
        }
        for c in tool_calls
    ]


def _tool_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def run_trajectory(
    *,
    goal: str,
    backend: Backend,
    tools: list[Tool],
    call_tool: Callable[[str, dict[str, Any]], Any],
    recorder: TrajectoryRecorder,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_steps: int = 16,
) -> None:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": goal},
    ]

    for _ in range(max_steps):
        turn = backend.chat(messages, tools)
        recorder.assistant(
            text=turn.text,
            tool_calls=turn.tool_calls,
            latency_ms=turn.latency_ms,
        )
        if not turn.tool_calls:
            recorder.end("complete")
            return

        messages.append(
            {
                "role": "assistant",
                "content": turn.text or None,
                "tool_calls": _serialize_tool_calls(turn.tool_calls),
            }
        )

        for call in turn.tool_calls:
            t0 = time.monotonic()
            is_error = False
            try:
                result = call_tool(call.name, call.arguments)
                if isinstance(result, dict) and result.get("isError"):
                    is_error = True
                content = result
            except Exception as exc:
                content = str(exc)
                is_error = True
            latency_ms = (time.monotonic() - t0) * 1000
            recorder.tool_result(
                ToolResult(call.id, call.name, content, is_error=is_error),
                latency_ms=latency_ms,
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": _tool_message_content(content),
                }
            )

    recorder.end("incomplete", message=f"hit step budget ({max_steps})")
