"""Tests for the Ollama lifecycle helpers in eval/scenarios/cluster.py."""

from __future__ import annotations

import socket
import subprocess

import pytest

from eval.scenarios import cluster


def test_is_local_ollama_matches_default_url() -> None:
    assert cluster.is_local_ollama("http://localhost:11434/v1")
    assert cluster.is_local_ollama("https://127.0.0.1:11434/v1")
    assert not cluster.is_local_ollama("https://api.openai.com/v1")
    assert not cluster.is_local_ollama("http://localhost:8080/v1")
    assert not cluster.is_local_ollama("http://example.com:11434/v1")


def test_health_check_true_when_socket_connects(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(socket, "create_connection", lambda *_a, **_kw: _Conn())
    assert cluster._ollama_health_check() is True


def test_health_check_false_when_socket_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*_a, **_kw):
        raise OSError("refused")

    monkeypatch.setattr(socket, "create_connection", _fail)
    assert cluster._ollama_health_check() is False


def test_manage_ollama_skips_start_when_already_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cluster, "_ollama_health_check", lambda: True)
    popen_calls: list = []
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **kw: popen_calls.append((a, kw)) or object()
    )
    monkeypatch.setattr(cluster, "_ollama_unload_model", lambda *_a, **_kw: None)

    with cluster.manage_ollama(models_to_unload=["m1"]):
        pass
    assert popen_calls == []  # never started a new daemon


def test_manage_ollama_starts_and_stops_when_daemon_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    health = [False, True]  # first call says down, then up after wait
    monkeypatch.setattr(cluster, "_ollama_health_check", lambda: health.pop(0) if health else True)

    class _FakeProc:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False
            self.waited = False

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            self.waited = True
            return 0

        def kill(self) -> None:
            self.killed = True

    fake_proc = _FakeProc()
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_kw: fake_proc)
    monkeypatch.setattr(cluster, "_ollama_unload_model", lambda *_a, **_kw: None)

    with cluster.manage_ollama(models_to_unload=["m1"]):
        pass

    assert fake_proc.terminated  # we started it, so we stopped it
    assert fake_proc.waited


def test_manage_ollama_unloads_models_on_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cluster, "_ollama_health_check", lambda: True)
    unloaded: list[str] = []
    monkeypatch.setattr(cluster, "_ollama_unload_model", lambda m: unloaded.append(m))

    with cluster.manage_ollama(models_to_unload=["m1", "m2"]):
        pass

    assert unloaded == ["m1", "m2"]


def test_manage_ollama_unloads_even_when_body_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cluster, "_ollama_health_check", lambda: True)
    unloaded: list[str] = []
    monkeypatch.setattr(cluster, "_ollama_unload_model", lambda m: unloaded.append(m))

    with (
        pytest.raises(RuntimeError, match="boom"),
        cluster.manage_ollama(models_to_unload=["m1"]),
    ):
        raise RuntimeError("boom")

    assert unloaded == ["m1"]
