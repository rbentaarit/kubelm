"""Profile YAML format: dataclasses + loader + composition.

A profile describes the cluster-level state (kind base node image plus
operators / CRDs) that scenarios run against. Profiles compose via
single-inheritance `extends:`; `compose_profile()` flattens the chain
into a single executable spec at run time.

If `prebuilt_image` is set, the runner (slice 2.4) uses that kind node
image directly and skips `install:` entirely. v0.1 ships the install
path; image baking is documented for Phase 3 as an optimization layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from eval.scenarios.spec import WaitForStatus

DEFAULT_PROFILE_WAIT_TIMEOUT = 180


@dataclass
class HelmRepo:
    name: str
    url: str


@dataclass
class HelmInstall:
    name: str
    chart: str
    namespace: str
    version: str | None = None


@dataclass
class InstallStep:
    """One install action: exactly one of helm_repo / helm_install is set."""

    helm_repo: HelmRepo | None = None
    helm_install: HelmInstall | None = None


@dataclass
class Profile:
    name: str
    extends: str | None = None
    node_image: str | None = None
    install: list[InstallStep] = field(default_factory=list)
    wait_for: list[WaitForStatus] = field(default_factory=list)
    prebuilt_image: str | None = None
    source_path: Path | None = None


class ProfileParseError(ValueError):
    """Raised when a profile YAML is structurally invalid."""


class ProfileCycleError(ValueError):
    """Raised when 'extends:' forms a cycle or names an unknown profile."""


def _require(data: Mapping[str, Any], key: str, ctx: str) -> Any:
    if key not in data:
        raise ProfileParseError(f"{ctx}: missing required field {key!r}")
    return data[key]


def _parse_duration(value: int | str | None, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    s = str(value).strip().lower()
    if not s:
        return default
    multipliers = {"s": 1, "m": 60, "h": 3600}
    if s[-1] in multipliers:
        return int(s[:-1]) * multipliers[s[-1]]
    return int(s)


def _parse_helm_repo(raw: Mapping[str, Any], ctx: str) -> HelmRepo:
    return HelmRepo(name=_require(raw, "name", ctx), url=_require(raw, "url", ctx))


def _parse_helm_install(raw: Mapping[str, Any], ctx: str) -> HelmInstall:
    return HelmInstall(
        name=_require(raw, "name", ctx),
        chart=_require(raw, "chart", ctx),
        namespace=_require(raw, "namespace", ctx),
        version=raw.get("version"),
    )


def _parse_install_step(raw: Mapping[str, Any], ctx: str) -> InstallStep:
    helm_repo = raw.get("helm_repo")
    helm_install = raw.get("helm_install")
    set_count = sum(1 for v in (helm_repo, helm_install) if v is not None)
    if set_count != 1:
        raise ProfileParseError(
            f"{ctx}: each install step must set exactly one of 'helm_repo' or 'helm_install'"
        )
    if helm_repo is not None:
        return InstallStep(helm_repo=_parse_helm_repo(helm_repo, ctx))
    return InstallStep(helm_install=_parse_helm_install(helm_install, ctx))


def _parse_wait_for(raw: Mapping[str, Any], ctx: str) -> WaitForStatus:
    return WaitForStatus(
        kind=_require(raw, "kind", ctx),
        namespace=_require(raw, "namespace", ctx),
        name=_require(raw, "name", ctx),
        reason=raw.get("reason"),
        phase=raw.get("phase"),
        condition=raw.get("condition") or "Available",
        timeout_seconds=_parse_duration(raw.get("timeout"), default=DEFAULT_PROFILE_WAIT_TIMEOUT),
    )


def load_profile(path: Path | str) -> Profile:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ProfileParseError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProfileParseError(f"{path}: top level must be a mapping")
    ctx = str(path)
    return Profile(
        name=_require(raw, "name", ctx),
        extends=raw.get("extends"),
        node_image=raw.get("node_image"),
        install=[
            _parse_install_step(s, f"{ctx}.install[{i}]")
            for i, s in enumerate(raw.get("install") or [])
        ],
        wait_for=[
            _parse_wait_for(s, f"{ctx}.wait_for[{i}]")
            for i, s in enumerate(raw.get("wait_for") or [])
        ],
        prebuilt_image=raw.get("prebuilt_image"),
        source_path=path,
    )


def load_profiles(directory: Path | str) -> dict[str, Profile]:
    directory = Path(directory)
    if not directory.is_dir():
        raise ProfileParseError(f"{directory}: not a directory")
    profiles: dict[str, Profile] = {}
    for p in sorted(directory.glob("*.yaml")):
        prof = load_profile(p)
        if prof.name in profiles:
            raise ProfileParseError(f"{p}: duplicate profile name {prof.name!r}")
        profiles[prof.name] = prof
    return profiles


def compose_profile(name: str, profiles: Mapping[str, Profile]) -> Profile:
    """Flatten the extends chain into a single Profile.

    Walks leaf -> root, then composes root-first so parent install steps
    run before child install steps. Child wins for scalar fields
    (node_image, prebuilt_image); list fields (install, wait_for) are
    concatenated in chain order.
    """
    chain_names: list[str] = []
    chain: list[Profile] = []
    current: str | None = name
    while current is not None:
        if current in chain_names:
            cycle = " -> ".join([*chain_names, current])
            raise ProfileCycleError(f"profile extends cycle: {cycle}")
        if current not in profiles:
            raise ProfileCycleError(f"unknown profile {current!r}")
        chain_names.append(current)
        chain.append(profiles[current])
        current = profiles[current].extends
    chain.reverse()

    leaf = chain[-1]
    composed = Profile(name=name, source_path=leaf.source_path)
    for prof in chain:
        if prof.node_image is not None:
            composed.node_image = prof.node_image
        if prof.prebuilt_image is not None:
            composed.prebuilt_image = prof.prebuilt_image
        composed.install.extend(prof.install)
        composed.wait_for.extend(prof.wait_for)
    return composed
