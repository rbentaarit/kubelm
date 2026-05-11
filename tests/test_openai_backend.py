from __future__ import annotations

from typing import Any

import pytest
import requests

from eval.client import Tool
from eval.runner import OpenAICompatBackend


class _FakeResponse:
    def __init__(self, body: dict[str, Any], status: int = 200) -> None:
        self._body = body
        self.status_code = status
        self.headers: dict[str, str] = {"Content-Type": "application/json"}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._body


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def post(self, url, json=None, headers=None, timeout=None) -> _FakeResponse:  # noqa: A002
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return self._response


def _tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"description of {name}",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        annotations={},
    )


def _resp_text(text: str) -> _FakeResponse:
    return _FakeResponse(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ]
        }
    )


def _resp_tool_call(call_id: str, name: str, args: str) -> _FakeResponse:
    return _FakeResponse(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": args},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )


def _backend(session: _FakeSession, **kwargs: Any) -> OpenAICompatBackend:
    return OpenAICompatBackend(
        base_url="http://localhost:1234/v1", model="test-model", http=session, **kwargs
    )


def test_chat_returns_text_only_turn() -> None:
    session = _FakeSession(_resp_text("hello world"))
    turn = _backend(session).chat([{"role": "user", "content": "hi"}], [])
    assert turn.text == "hello world"
    assert turn.tool_calls == []
    assert turn.latency_ms >= 0
    assert turn.raw is not None


def test_chat_parses_tool_calls() -> None:
    session = _FakeSession(_resp_tool_call("c1", "list-namespaces", '{"x": "y"}'))
    turn = _backend(session).chat([], [_tool("list-namespaces")])
    assert turn.text == ""
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].id == "c1"
    assert turn.tool_calls[0].name == "list-namespaces"
    assert turn.tool_calls[0].arguments == {"x": "y"}


def test_malformed_arguments_become_empty_dict() -> None:
    session = _FakeSession(_resp_tool_call("c1", "list-namespaces", "{not valid"))
    turn = _backend(session).chat([], [_tool("list-namespaces")])
    assert turn.tool_calls[0].arguments == {}


def test_non_object_arguments_become_empty_dict() -> None:
    # Some models emit a JSON array or null in arguments.
    session = _FakeSession(_resp_tool_call("c1", "list-namespaces", "[1, 2, 3]"))
    turn = _backend(session).chat([], [_tool("list-namespaces")])
    assert turn.tool_calls[0].arguments == {}


def test_tools_are_translated_to_openai_format() -> None:
    session = _FakeSession(_resp_text("ok"))
    _backend(session).chat([{"role": "user", "content": "hi"}], [_tool("list-namespaces")])
    payload = session.calls[0]["json"]
    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["function"]["name"] == "list-namespaces"
    assert payload["tools"][0]["function"]["parameters"]["type"] == "object"
    assert payload["tool_choice"] == "auto"


def test_no_tools_omits_tools_field() -> None:
    session = _FakeSession(_resp_text("ok"))
    _backend(session).chat([{"role": "user", "content": "hi"}], [])
    payload = session.calls[0]["json"]
    assert "tools" not in payload
    assert "tool_choice" not in payload


def test_api_key_sets_authorization_header() -> None:
    session = _FakeSession(_resp_text("ok"))
    _backend(session, api_key="sk-abc").chat([], [])
    assert session.calls[0]["headers"]["Authorization"] == "Bearer sk-abc"


def test_no_api_key_omits_authorization_header() -> None:
    session = _FakeSession(_resp_text("ok"))
    _backend(session).chat([], [])
    assert "Authorization" not in session.calls[0]["headers"]


def test_url_is_chat_completions_path() -> None:
    session = _FakeSession(_resp_text("ok"))
    OpenAICompatBackend(
        base_url="http://localhost:1234/v1/", model="test-model", http=session
    ).chat([], [])
    assert session.calls[0]["url"] == "http://localhost:1234/v1/chat/completions"


def test_http_error_raises() -> None:
    session = _FakeSession(_FakeResponse({}, status=500))
    with pytest.raises(requests.HTTPError):
        _backend(session).chat([], [])


def test_payload_includes_temperature_and_max_tokens() -> None:
    session = _FakeSession(_resp_text("ok"))
    _backend(session, temperature=0.7, max_tokens=512).chat([], [])
    payload = session.calls[0]["json"]
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 512
    assert "max_completion_tokens" not in payload
    assert payload["model"] == "test-model"


@pytest.mark.parametrize(
    "model",
    ["gpt-5", "gpt-5.1", "gpt-5.4", "gpt-5.4-mini", "GPT-5.5", "o1", "o1-mini", "o3", "o3-pro"],
)
def test_gpt5_and_reasoning_models_use_max_completion_tokens(model: str) -> None:
    session = _FakeSession(_resp_text("ok"))
    OpenAICompatBackend(
        base_url="https://api.openai.com/v1", model=model, http=session, max_tokens=2048
    ).chat([], [])
    payload = session.calls[0]["json"]
    assert payload["max_completion_tokens"] == 2048
    assert "max_tokens" not in payload


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4.1", "gpt-3.5-turbo", "qwen2.5:7b", "llama3.2"])
def test_legacy_models_use_max_tokens(model: str) -> None:
    session = _FakeSession(_resp_text("ok"))
    OpenAICompatBackend(
        base_url="https://api.openai.com/v1", model=model, http=session, max_tokens=2048
    ).chat([], [])
    payload = session.calls[0]["json"]
    assert payload["max_tokens"] == 2048
    assert "max_completion_tokens" not in payload
