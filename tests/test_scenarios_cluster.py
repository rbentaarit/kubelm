from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from eval.scenarios.cluster import (
    CommandError,
    helm_install,
    helm_repo_add,
    kind_create_cluster,
    kind_delete_cluster,
    kind_list_clusters,
    kubectl_apply,
    kubectl_apply_file,
    kubectl_delete_namespace,
    kubectl_get,
    wait_for_status,
)


class _FakeRun:
    """Replaces subprocess.run for tests; records calls, returns scripted output."""

    def __init__(
        self,
        responses: list[tuple[int, str, str]] | None = None,
        *,
        responder: Callable[[list[str]], tuple[int, str, str]] | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._responder = responder
        self.calls: list[dict[str, Any]] = []

    def __call__(self, cmd, **kwargs) -> subprocess.CompletedProcess[str]:  # noqa: ARG002
        self.calls.append(
            {
                "cmd": cmd,
                "env": kwargs.get("env"),
                "input": kwargs.get("input"),
                "timeout": kwargs.get("timeout"),
            }
        )
        if self._responder is not None:
            rc, stdout, stderr = self._responder(cmd)
        elif self._responses:
            rc, stdout, stderr = self._responses.pop(0)
        else:
            rc, stdout, stderr = 0, "", ""
        return subprocess.CompletedProcess(cmd, rc, stdout, stderr)


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch) -> _FakeRun:
    fake = _FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    return fake


# ---------- kind ----------


def test_kind_create_cluster_constructs_command(fake_run: _FakeRun, tmp_path: Path) -> None:
    kc_path = tmp_path / "kc"
    kc = kind_create_cluster("kubelm-x", kubeconfig_path=kc_path, image="kindest/node:v1")

    cmd = fake_run.calls[0]["cmd"]
    assert cmd[:3] == ["kind", "create", "cluster"]
    assert "--name" in cmd and "kubelm-x" in cmd
    assert "--kubeconfig" in cmd and str(kc_path) in cmd
    assert "--image" in cmd and "kindest/node:v1" in cmd
    assert kc.name == "kubelm-x"
    assert kc.kubeconfig_path == kc_path


def test_kind_create_cluster_omits_image_when_none(fake_run: _FakeRun, tmp_path: Path) -> None:
    kind_create_cluster("foo", kubeconfig_path=tmp_path / "kc")
    assert "--image" not in fake_run.calls[0]["cmd"]


def test_kind_create_cluster_raises_command_error_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subprocess, "run", _FakeRun([(1, "", "boom")]))
    with pytest.raises(CommandError) as excinfo:
        kind_create_cluster("foo", kubeconfig_path=tmp_path / "kc")
    assert "boom" in str(excinfo.value)
    assert excinfo.value.returncode == 1


def test_kind_delete_cluster(fake_run: _FakeRun) -> None:
    kind_delete_cluster("kubelm-x")
    assert fake_run.calls[0]["cmd"] == ["kind", "delete", "cluster", "--name", "kubelm-x"]


def test_kind_list_clusters_parses_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", _FakeRun([(0, "alpha\nbeta\n\n", "")]))
    assert kind_list_clusters() == ["alpha", "beta"]


# ---------- kubectl ----------


def test_kubectl_apply_pipes_manifest_to_stdin(fake_run: _FakeRun, tmp_path: Path) -> None:
    kubectl_apply("kind: Pod\n", kubeconfig_path=tmp_path / "kc")
    call = fake_run.calls[0]
    assert call["cmd"][:5] == ["kubectl", "--kubeconfig", str(tmp_path / "kc"), "apply", "-f"]
    assert call["cmd"][5] == "-"
    assert call["input"] == "kind: Pod\n"


def test_kubectl_apply_file_passes_path(fake_run: _FakeRun, tmp_path: Path) -> None:
    f = tmp_path / "m.yaml"
    f.write_text("kind: Pod\n")
    kubectl_apply_file(f, kubeconfig_path=tmp_path / "kc")
    cmd = fake_run.calls[0]["cmd"]
    assert cmd[-2:] == ["-f", str(f)]


def test_kubectl_get_returns_parsed_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    body = {"status": {"phase": "Running"}}
    monkeypatch.setattr(subprocess, "run", _FakeRun([(0, json.dumps(body), "")]))
    obj = kubectl_get("Pod", "p", "ns", kubeconfig_path=tmp_path / "kc")
    assert obj == body


def test_kubectl_delete_namespace_uses_ignore_not_found(fake_run: _FakeRun, tmp_path: Path) -> None:
    kubectl_delete_namespace("scenario-x", kubeconfig_path=tmp_path / "kc")
    cmd = fake_run.calls[0]["cmd"]
    assert "delete" in cmd and "namespace" in cmd and "scenario-x" in cmd
    assert "--ignore-not-found" in cmd


# ---------- wait_for_status ----------


class _Clock:
    """Deterministic clock that advances `step` per `now()` call."""

    def __init__(self, step: float = 1.0) -> None:
        self.t = 0.0
        self.step = step
        self.sleeps: list[float] = []

    def now(self) -> float:
        self.t += self.step
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)


def test_wait_for_status_returns_when_phase_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    states = [
        json.dumps({"status": {"phase": "Pending"}}),
        json.dumps({"status": {"phase": "Pending"}}),
        json.dumps({"status": {"phase": "Running"}}),
    ]
    monkeypatch.setattr(subprocess, "run", _FakeRun(responder=lambda _cmd: (0, states.pop(0), "")))
    clock = _Clock(step=1.0)
    wait_for_status(
        kind="Pod",
        name="p",
        namespace="ns",
        kubeconfig_path=tmp_path / "kc",
        phase="Running",
        timeout_seconds=10,
        sleep=clock.sleep,
        now=clock.now,
    )
    assert states == []  # consumed all three responses
    assert clock.sleeps  # slept between attempts


def test_wait_for_status_matches_container_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    body = {
        "status": {
            "phase": "Running",
            "containerStatuses": [{"state": {"waiting": {"reason": "CrashLoopBackOff"}}}],
        }
    }
    monkeypatch.setattr(subprocess, "run", _FakeRun([(0, json.dumps(body), "")]))
    clock = _Clock()
    wait_for_status(
        kind="Pod",
        name="p",
        namespace="ns",
        kubeconfig_path=tmp_path / "kc",
        reason="CrashLoopBackOff",
        timeout_seconds=10,
        sleep=clock.sleep,
        now=clock.now,
    )


def test_wait_for_status_matches_condition_only_when_true(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    body_false = {"status": {"conditions": [{"type": "Available", "status": "False"}]}}
    body_true = {"status": {"conditions": [{"type": "Available", "status": "True"}]}}
    seq = [json.dumps(body_false), json.dumps(body_true)]
    monkeypatch.setattr(subprocess, "run", _FakeRun(responder=lambda _cmd: (0, seq.pop(0), "")))
    clock = _Clock()
    wait_for_status(
        kind="Deployment",
        name="d",
        namespace="ns",
        kubeconfig_path=tmp_path / "kc",
        condition="Available",
        timeout_seconds=10,
        sleep=clock.sleep,
        now=clock.now,
    )
    assert seq == []


def test_wait_for_status_times_out(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    body = json.dumps({"status": {"phase": "Pending"}})
    monkeypatch.setattr(subprocess, "run", _FakeRun(responder=lambda _cmd: (0, body, "")))
    clock = _Clock(step=2.0)  # advance 2s per now() call
    with pytest.raises(TimeoutError, match="timed out"):
        wait_for_status(
            kind="Pod",
            name="p",
            namespace="ns",
            kubeconfig_path=tmp_path / "kc",
            phase="Running",
            timeout_seconds=4,
            sleep=clock.sleep,
            now=clock.now,
        )


def test_wait_for_status_retries_on_kubectl_get_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    body = json.dumps({"status": {"phase": "Running"}})
    seq: list[tuple[int, str, str]] = [
        (1, "", "not found yet"),
        (0, body, ""),
    ]
    monkeypatch.setattr(subprocess, "run", _FakeRun(responder=lambda _cmd: seq.pop(0)))
    clock = _Clock()
    wait_for_status(
        kind="Pod",
        name="p",
        namespace="ns",
        kubeconfig_path=tmp_path / "kc",
        phase="Running",
        timeout_seconds=10,
        sleep=clock.sleep,
        now=clock.now,
    )


def test_wait_for_status_requires_at_least_one_predicate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        wait_for_status(
            kind="Pod",
            name="p",
            namespace="ns",
            kubeconfig_path=tmp_path / "kc",
            timeout_seconds=10,
        )


# ---------- helm ----------


def test_helm_repo_add_runs_add_then_update(fake_run: _FakeRun, tmp_path: Path) -> None:
    helm_repo_add("argo", "https://argoproj.github.io/argo-helm", helm_home=tmp_path / "helm")
    assert [c["cmd"][:3] for c in fake_run.calls] == [
        ["helm", "repo", "add"],
        ["helm", "repo", "update"],
    ]


def test_helm_install_uses_per_run_env(fake_run: _FakeRun, tmp_path: Path) -> None:
    helm_home = tmp_path / "helm"
    helm_install(
        name="argocd",
        chart="argo/argo-cd",
        namespace="argocd",
        kubeconfig_path=tmp_path / "kc",
        helm_home=helm_home,
        version="7.6.12",
    )
    cmd = fake_run.calls[0]["cmd"]
    env = fake_run.calls[0]["env"]
    assert cmd[:4] == ["helm", "install", "argocd", "argo/argo-cd"]
    assert "--namespace" in cmd and "argocd" in cmd
    assert "--create-namespace" in cmd
    assert "--version" in cmd and "7.6.12" in cmd
    assert env["HELM_CACHE_HOME"] == str(helm_home / "cache")
    assert env["HELM_CONFIG_HOME"] == str(helm_home / "config")
    assert env["HELM_DATA_HOME"] == str(helm_home / "data")


def test_helm_install_omits_version_when_none(fake_run: _FakeRun, tmp_path: Path) -> None:
    helm_install(
        name="x",
        chart="r/c",
        namespace="ns",
        kubeconfig_path=tmp_path / "kc",
        helm_home=tmp_path / "helm",
    )
    assert "--version" not in fake_run.calls[0]["cmd"]
