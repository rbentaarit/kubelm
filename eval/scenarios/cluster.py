"""Subprocess wrappers around kind, kubectl, and helm.

Each wrapper takes an explicit kubeconfig path (and helm cache home,
where applicable) so parallel scenarios never collide on shared global
state. The runner (slice 2.4) hands every scenario its own per-run
tmpdir for these.

Failures raise CommandError with the exact command, exit code, and
captured stdout/stderr so debugging is concrete. Wrappers do not retry
or recover; that's the runner's job.

`wait_for_status` is the only non-trivial helper: kubectl wait can
match conditions but not container-state reason fields like
"CrashLoopBackOff", which is what scenarios actually need to settle on.
We poll `kubectl get -o json` and inspect the parsed status manually.

`k8sgpt_mcp_server` is a context manager that spawns
`k8sgpt serve --mcp --mcp-http` on an ephemeral port pointed at a
specific kubeconfig, yields the MCP URL, and tears the process down on
exit. The scenario CLI and the live integration test share it.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class CommandError(RuntimeError):
    """A subprocess invocation returned a non-zero exit code."""

    def __init__(self, cmd: list[str], returncode: int, stdout: str, stderr: str) -> None:
        super().__init__(
            f"command failed (exit {returncode}): {' '.join(cmd)}\n"
            f"stdout: {stdout}\nstderr: {stderr}"
        )
        self.cmd = cmd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _run(
    cmd: list[str],
    *,
    env: Mapping[str, str] | None = None,
    input_text: str | None = None,
    timeout: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    log.debug("exec: %s", " ".join(cmd))
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=full_env,
        input=input_text,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        raise CommandError(cmd, result.returncode, result.stdout, result.stderr)
    return result


# ---------- kind ----------


@dataclass
class KindCluster:
    name: str
    kubeconfig_path: Path


def kind_create_cluster(
    name: str,
    *,
    kubeconfig_path: Path,
    image: str | None = None,
    timeout: float = 180,
) -> KindCluster:
    cmd = [
        "kind",
        "create",
        "cluster",
        "--name",
        name,
        "--kubeconfig",
        str(kubeconfig_path),
    ]
    if image:
        cmd += ["--image", image]
    _run(cmd, timeout=timeout)
    return KindCluster(name=name, kubeconfig_path=kubeconfig_path)


def kind_delete_cluster(
    name: str,
    *,
    kubeconfig_path: Path | None = None,
    timeout: float = 60,
) -> None:
    cmd = ["kind", "delete", "cluster", "--name", name]
    if kubeconfig_path is not None:
        cmd += ["--kubeconfig", str(kubeconfig_path)]
    _run(cmd, timeout=timeout)


def kind_list_clusters(timeout: float = 10) -> list[str]:
    result = _run(["kind", "get", "clusters"], timeout=timeout)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


# ---------- kubectl ----------


# When a namespace is freshly created, its default ServiceAccount is created
# asynchronously by the SA controller. Pods scheduled into the namespace before
# that controller catches up are rejected with these stderr patterns. We retry
# specifically on these to avoid masking real errors. Same applies to the
# "no endpoints available" race against the Service controller.
_APPLY_TRANSIENT_PATTERNS = (
    'serviceaccount "default" not found',
    "no endpoints available",
)


def _apply_with_transient_retry(
    cmd: list[str],
    *,
    input_text: str | None,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> None:
    for attempt in range(retries):
        try:
            _run(cmd, input_text=input_text, timeout=timeout)
            return
        except CommandError as exc:
            transient = any(p in (exc.stderr or "") for p in _APPLY_TRANSIENT_PATTERNS)
            if not transient or attempt == retries - 1:
                raise
            time.sleep(retry_delay)


def kubectl_apply(
    manifest_yaml: str,
    *,
    kubeconfig_path: Path,
    timeout: float = 60,
    retries: int = 6,
    retry_delay: float = 1.0,
) -> None:
    _apply_with_transient_retry(
        ["kubectl", "--kubeconfig", str(kubeconfig_path), "apply", "-f", "-"],
        input_text=manifest_yaml,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
    )


def kubectl_apply_file(
    path: Path,
    *,
    kubeconfig_path: Path,
    timeout: float = 60,
    retries: int = 6,
    retry_delay: float = 1.0,
) -> None:
    _apply_with_transient_retry(
        ["kubectl", "--kubeconfig", str(kubeconfig_path), "apply", "-f", str(path)],
        input_text=None,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
    )


def kubectl_get(
    kind: str,
    name: str,
    namespace: str,
    *,
    kubeconfig_path: Path,
    timeout: float = 30,
) -> dict[str, Any]:
    result = _run(
        [
            "kubectl",
            "--kubeconfig",
            str(kubeconfig_path),
            "get",
            kind,
            name,
            "-n",
            namespace,
            "-o",
            "json",
        ],
        timeout=timeout,
    )
    return json.loads(result.stdout)


def kubectl_delete_namespace(
    namespace: str,
    *,
    kubeconfig_path: Path,
    timeout: float = 120,
) -> None:
    _run(
        [
            "kubectl",
            "--kubeconfig",
            str(kubeconfig_path),
            "delete",
            "namespace",
            namespace,
            "--ignore-not-found",
        ],
        timeout=timeout,
    )


def _has_reason(status: Mapping[str, Any], reason: str) -> bool:
    """Search Pod status for a matching reason in container or top-level state."""
    if status.get("reason") == reason:
        return True
    for source_key in ("containerStatuses", "initContainerStatuses"):
        for cs in status.get(source_key) or []:
            for state_key in ("state", "lastState"):
                state = cs.get(state_key) or {}
                for phase_key in ("waiting", "terminated"):
                    if (state.get(phase_key) or {}).get("reason") == reason:
                        return True
    return False


def _matches(
    obj: Mapping[str, Any],
    *,
    reason: str | None,
    phase: str | None,
    condition: str | None,
) -> bool:
    status = obj.get("status") or {}
    if phase is not None and status.get("phase") != phase:
        return False
    if reason is not None and not _has_reason(status, reason):
        return False
    if condition is not None:
        conds = status.get("conditions") or []
        if not any(c.get("type") == condition and c.get("status") == "True" for c in conds):
            return False
    return True


def wait_for_status(
    *,
    kind: str,
    name: str,
    namespace: str,
    kubeconfig_path: Path,
    reason: str | None = None,
    phase: str | None = None,
    condition: str | None = None,
    timeout_seconds: int,
    poll_interval_seconds: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> None:
    """Poll kubectl get until the resource matches; raise TimeoutError otherwise."""
    if reason is None and phase is None and condition is None:
        raise ValueError("at least one of reason/phase/condition must be set")
    deadline = now() + timeout_seconds
    last_state: dict[str, Any] | None = None
    while now() < deadline:
        try:
            obj = kubectl_get(kind, name, namespace, kubeconfig_path=kubeconfig_path)
        except CommandError:
            sleep(poll_interval_seconds)
            continue
        last_state = obj
        if _matches(obj, reason=reason, phase=phase, condition=condition):
            return
        sleep(poll_interval_seconds)
    raise TimeoutError(
        f"timed out after {timeout_seconds}s waiting for "
        f"{kind}/{name} in {namespace} (reason={reason} phase={phase} "
        f"condition={condition}); last state: {last_state}"
    )


# ---------- helm ----------


def _helm_env(helm_home: Path) -> dict[str, str]:
    return {
        "HELM_CACHE_HOME": str(helm_home / "cache"),
        "HELM_CONFIG_HOME": str(helm_home / "config"),
        "HELM_DATA_HOME": str(helm_home / "data"),
    }


def helm_repo_add(
    name: str,
    url: str,
    *,
    helm_home: Path,
    timeout: float = 60,
) -> None:
    _run(["helm", "repo", "add", name, url], env=_helm_env(helm_home), timeout=timeout)
    _run(["helm", "repo", "update", name], env=_helm_env(helm_home), timeout=timeout)


def helm_install(
    *,
    name: str,
    chart: str,
    namespace: str,
    kubeconfig_path: Path,
    helm_home: Path,
    version: str | None = None,
    timeout: float = 300,
) -> None:
    cmd = [
        "helm",
        "install",
        name,
        chart,
        "--namespace",
        namespace,
        "--create-namespace",
        "--kubeconfig",
        str(kubeconfig_path),
        "--wait",
        f"--timeout={int(timeout)}s",
    ]
    if version:
        cmd += ["--version", version]
    _run(cmd, env=_helm_env(helm_home), timeout=timeout + 30)


# ---------- k8sgpt MCP server ----------


def find_free_port() -> int:
    """Bind, read, release. Inherits the SO_REUSEADDR window."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(port: int, *, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"port {port} did not accept connections within {timeout}s")


@contextmanager
def k8sgpt_mcp_server(kubeconfig_path: Path) -> Iterator[str]:
    """Spawn `k8sgpt serve --mcp --mcp-http` against a kubeconfig.

    Yields the MCP HTTP URL. Tears the process down on exit, even if
    the body raises. Uses an ephemeral port so multiple scenarios can
    run in parallel without colliding.
    """
    port = find_free_port()
    env = dict(os.environ)
    env["KUBECONFIG"] = str(kubeconfig_path)
    proc = subprocess.Popen(
        ["k8sgpt", "serve", "--mcp", "--mcp-http", "--mcp-port", str(port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_port(port)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
