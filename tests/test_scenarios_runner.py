from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from eval.scenarios.profile import (
    HelmInstall,
    HelmRepo,
    InstallStep,
    Profile,
)
from eval.scenarios.runner import cluster_name_for, scenario_context
from eval.scenarios.spec import (
    ConclusionRubric,
    ExpectedOutcome,
    ReferenceCalls,
    Scenario,
    SetupStep,
    WaitForStatus,
)


class _Recorder:
    """Replaces subprocess.run; records cmd/env/input and returns scripted output."""

    def __init__(
        self, responder: Callable[[list[str]], tuple[int, str, str]] | None = None
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responder = responder or self._default

    @staticmethod
    def _default(cmd: list[str]) -> tuple[int, str, str]:
        # kubectl get -o json: return a status that matches phase=Running and
        # condition Available=True. Sufficient for any settle that uses those.
        if cmd[:1] == ["kubectl"] and "get" in cmd:
            body = {
                "status": {
                    "phase": "Running",
                    "conditions": [{"type": "Available", "status": "True"}],
                }
            }
            return 0, json.dumps(body), ""
        return 0, "", ""

    def __call__(self, cmd, **kwargs) -> subprocess.CompletedProcess[str]:  # noqa: ARG002
        self.calls.append(
            {
                "cmd": cmd,
                "env": kwargs.get("env"),
                "input": kwargs.get("input"),
                "timeout": kwargs.get("timeout"),
            }
        )
        rc, stdout, stderr = self._responder(cmd)
        return subprocess.CompletedProcess(cmd, rc, stdout, stderr)


@pytest.fixture
def fake_subprocess(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(subprocess, "run", rec)
    return rec


def _scenario(
    *,
    setup: list[SetupStep] | None = None,
    settle: list[WaitForStatus] | None = None,
    source_path: Path | None = None,
) -> Scenario:
    return Scenario(
        id="pod-crashloop-001",
        profile="base",
        goal="Why?",
        setup=setup or [],
        settle=settle or [],
        expected=ExpectedOutcome(
            reference_calls=ReferenceCalls(),
            conclusion_rubric=ConclusionRubric(),
        ),
        source_path=source_path,
    )


def _profile(
    *,
    install: list[InstallStep] | None = None,
    wait_for: list[WaitForStatus] | None = None,
    prebuilt_image: str | None = None,
    node_image: str | None = "kindest/node:v1.31.4",
) -> Profile:
    return Profile(
        name="base",
        node_image=node_image,
        install=install or [],
        wait_for=wait_for or [],
        prebuilt_image=prebuilt_image,
    )


def test_cluster_name_for_is_deterministic_and_dns_compliant() -> None:
    n = cluster_name_for("abcdef0123456789", "pod-crashloop-001")
    assert n == "kubelm-abcdef01-pod-crashloop-001"
    assert len(n) <= 63
    assert n == n.lower()


def test_cluster_name_for_truncates_long_ids() -> None:
    n = cluster_name_for("0123456789", "very-long-scenario-" + ("x" * 80))
    assert len(n) <= 63


def test_yields_context_and_tears_down_on_success(
    fake_subprocess: _Recorder, tmp_path: Path
) -> None:
    scn = _scenario(
        setup=[SetupStep(apply_inline="kind: Pod\n")],
        settle=[WaitForStatus(kind="Pod", namespace="ns", name="p", phase="Running")],
    )
    prof = _profile()

    with scenario_context(
        scenario=scn,
        profile=prof,
        run_id="abc12345",
        output_root=tmp_path,
    ) as ctx:
        assert ctx.cluster_name.startswith("kubelm-abc12345-")
        assert ctx.kubeconfig_path.parent.name == "_workspace"
        assert ctx.output_dir == tmp_path / "abc12345" / "pod-crashloop-001"

    cmds = [c["cmd"] for c in fake_subprocess.calls]
    cmd_signatures = [tuple(c[:3]) for c in cmds]
    assert ("kind", "create", "cluster") in cmd_signatures
    assert ("kubectl", "--kubeconfig", str(ctx.kubeconfig_path)) in cmd_signatures or any(
        c[0] == "kubectl" and "apply" in c for c in cmds
    )
    assert ("kind", "delete", "cluster") in cmd_signatures


def test_tears_down_cluster_when_body_raises(fake_subprocess: _Recorder, tmp_path: Path) -> None:
    scn = _scenario()
    prof = _profile()

    with (
        pytest.raises(RuntimeError, match="boom"),
        scenario_context(scenario=scn, profile=prof, run_id="abc12345", output_root=tmp_path),
    ):
        raise RuntimeError("boom")

    delete_calls = [
        c for c in fake_subprocess.calls if c["cmd"][:3] == ["kind", "delete", "cluster"]
    ]
    assert len(delete_calls) == 1


def test_install_profile_steps_run_in_order(fake_subprocess: _Recorder, tmp_path: Path) -> None:
    scn = _scenario()
    prof = _profile(
        install=[
            InstallStep(helm_repo=HelmRepo(name="argo", url="https://example/argo")),
            InstallStep(
                helm_install=HelmInstall(name="argocd", chart="argo/argo-cd", namespace="argocd")
            ),
        ]
    )

    with scenario_context(scenario=scn, profile=prof, run_id="abc12345", output_root=tmp_path):
        pass

    helm_cmds = [c["cmd"] for c in fake_subprocess.calls if c["cmd"][0] == "helm"]
    # Expect: helm repo add, helm repo update, helm install (in that order).
    actions = [tuple(c[:3]) for c in helm_cmds]
    assert actions[0] == ("helm", "repo", "add")
    assert actions[1] == ("helm", "repo", "update")
    assert tuple(helm_cmds[2][:3]) == ("helm", "install", "argocd")


def test_prebuilt_image_skips_install_steps(fake_subprocess: _Recorder, tmp_path: Path) -> None:
    scn = _scenario()
    prof = _profile(
        install=[InstallStep(helm_repo=HelmRepo(name="argo", url="https://x"))],
        prebuilt_image="kubelm-argocd-base:v1",
    )

    with scenario_context(scenario=scn, profile=prof, run_id="abc12345", output_root=tmp_path):
        pass

    helm_cmds = [c for c in fake_subprocess.calls if c["cmd"][0] == "helm"]
    assert helm_cmds == []
    create_cmd = next(
        c for c in fake_subprocess.calls if c["cmd"][:3] == ["kind", "create", "cluster"]
    )
    assert "--image" in create_cmd["cmd"]
    assert "kubelm-argocd-base:v1" in create_cmd["cmd"]


def test_apply_inline_pipes_manifest(fake_subprocess: _Recorder, tmp_path: Path) -> None:
    scn = _scenario(setup=[SetupStep(apply_inline="kind: Pod\nname: foo\n")])
    prof = _profile()

    with scenario_context(scenario=scn, profile=prof, run_id="abc12345", output_root=tmp_path):
        pass

    apply_calls = [
        c for c in fake_subprocess.calls if c["cmd"][0] == "kubectl" and "apply" in c["cmd"]
    ]
    assert any(c["input"] == "kind: Pod\nname: foo\n" for c in apply_calls)


def test_apply_file_resolves_relative_to_scenario_path(
    fake_subprocess: _Recorder, tmp_path: Path
) -> None:
    scenario_dir = tmp_path / "specs"
    scenario_dir.mkdir()
    scenario_yaml = scenario_dir / "x.yaml"
    scenario_yaml.write_text("placeholder")

    manifests_dir = scenario_dir / "manifests"
    manifests_dir.mkdir()
    manifest = manifests_dir / "foo.yaml"
    manifest.write_text("kind: Pod\n")

    scn = _scenario(
        setup=[SetupStep(apply_file="manifests/foo.yaml")],
        source_path=scenario_yaml,
    )
    prof = _profile()
    out_root = tmp_path / "out"

    with scenario_context(scenario=scn, profile=prof, run_id="abc12345", output_root=out_root):
        pass

    apply_calls = [
        c for c in fake_subprocess.calls if c["cmd"][0] == "kubectl" and "apply" in c["cmd"]
    ]
    assert any(str(manifest) in c["cmd"] for c in apply_calls)


def test_apply_file_relative_without_source_path_raises(
    fake_subprocess: _Recorder, tmp_path: Path
) -> None:
    scn = _scenario(setup=[SetupStep(apply_file="manifests/foo.yaml")], source_path=None)
    prof = _profile()
    with (
        pytest.raises(ValueError, match="not loaded from a file"),
        scenario_context(scenario=scn, profile=prof, run_id="abc12345", output_root=tmp_path),
    ):
        pass
    # Cluster was created and torn down.
    delete_calls = [
        c for c in fake_subprocess.calls if c["cmd"][:3] == ["kind", "delete", "cluster"]
    ]
    assert len(delete_calls) == 1


def test_settle_polls_kubectl_get(fake_subprocess: _Recorder, tmp_path: Path) -> None:
    scn = _scenario(
        settle=[
            WaitForStatus(
                kind="Pod",
                namespace="scenario-pod-crashloop-001",
                name="crash-pod",
                reason="CrashLoopBackOff",
                timeout_seconds=10,
            )
        ]
    )
    prof = _profile()

    crash_status = {
        "status": {
            "phase": "Running",
            "containerStatuses": [{"state": {"waiting": {"reason": "CrashLoopBackOff"}}}],
        }
    }

    def responder(cmd: list[str]) -> tuple[int, str, str]:
        if cmd[:1] == ["kubectl"] and "get" in cmd:
            return 0, json.dumps(crash_status), ""
        return 0, "", ""

    fake_subprocess._responder = responder  # type: ignore[attr-defined]

    with scenario_context(scenario=scn, profile=prof, run_id="abc12345", output_root=tmp_path):
        pass

    get_calls = [c for c in fake_subprocess.calls if c["cmd"][0] == "kubectl" and "get" in c["cmd"]]
    assert any("crash-pod" in c["cmd"] for c in get_calls)


def test_kind_create_uses_node_image_when_no_prebuilt(
    fake_subprocess: _Recorder, tmp_path: Path
) -> None:
    scn = _scenario()
    prof = _profile(node_image="kindest/node:v1.31.4")

    with scenario_context(scenario=scn, profile=prof, run_id="abc12345", output_root=tmp_path):
        pass

    create_cmd = next(
        c for c in fake_subprocess.calls if c["cmd"][:3] == ["kind", "create", "cluster"]
    )
    assert "--image" in create_cmd["cmd"]
    assert "kindest/node:v1.31.4" in create_cmd["cmd"]
