"""OpenAI-compatible Chat Completions backend.

Speaks the OpenAI Chat Completions API with tool-use. Same shape works
against the real OpenAI endpoint, Ollama (http://localhost:11434/v1),
llama.cpp's HTTP server, vLLM, LocalAI, and anything else implementing
the de facto standard.

Malformed `arguments` JSON (small models routinely emit it) is folded
to an empty dict so the schema validator can flag the call as an
argument hallucination via "missing required fields". The raw bad
string is preserved on AssistantTurn.raw for in-memory inspection but
is not written to the trajectory file.

HTTP errors (4xx/5xx, including rate limits) propagate as
requests.HTTPError; the runner does not catch them, so the recorder's
context-manager exit writes end("error", ...). No retries by design.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from eval.client import Tool
from eval.runner.backend import AssistantTurn
from eval.trajectory import ToolCall

log = logging.getLogger(__name__)


def _uses_max_completion_tokens(model: str) -> bool:
    """OpenAI's gpt-5 family and o1/o3 reasoning models reject `max_tokens`.

    They require `max_completion_tokens` instead. Local backends (Ollama,
    llama.cpp) and older OpenAI models still want `max_tokens`. Detection
    is by model name since that matches the actual API constraint.
    """
    name = model.lower()
    return name.startswith("gpt-5") or name.startswith("o1") or name.startswith("o3")


def _tool_to_openai(tool: Tool) -> dict[str, Any]:
    parameters = tool.input_schema or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        },
    }


@dataclass
class OpenAICompatBackend:
    base_url: str
    model: str
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2048
    reasoning_effort: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    timeout: float = 120.0
    http: requests.Session = field(default_factory=requests.Session)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool],
    ) -> AssistantTurn:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if _uses_max_completion_tokens(self.model):
            payload["max_completion_tokens"] = self.max_tokens
        else:
            payload["max_tokens"] = self.max_tokens
        # reasoning_effort="none" disables thinking on reasoning-capable
        # backends (e.g. ollama's Qwen3.5 renderer, which ignores the
        # legacy /no_think token over /v1). Left unset for non-reasoning
        # models so it has no effect on them.
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        # chat_template_kwargs is the llama.cpp /v1 lever for Qwen3.5-style
        # templates that take template variables (e.g. enable_thinking=false).
        # Ollama exposes the same control via reasoning_effort over /v1.
        if self.chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        if tools:
            payload["tools"] = [_tool_to_openai(t) for t in tools]
            payload["tool_choice"] = "auto"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = self.base_url.rstrip("/") + "/chat/completions"
        t0 = time.monotonic()
        resp = self.http.post(url, json=payload, headers=headers, timeout=self.timeout)
        latency_ms = (time.monotonic() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]["message"]
        text = choice.get("content") or ""
        tool_calls: list[ToolCall] = []
        for tc in choice.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                log.warning("malformed tool arguments from model: %r", raw_args)
                args = {}
            tool_calls.append(
                ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args)
            )

        return AssistantTurn(
            text=text,
            tool_calls=tool_calls,
            latency_ms=latency_ms,
            raw=data,
        )
