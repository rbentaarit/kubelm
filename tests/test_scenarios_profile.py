from __future__ import annotations

from pathlib import Path

import pytest

from eval.scenarios.profile import (
    Profile,
    ProfileCycleError,
    ProfileParseError,
    compose_profile,
    load_profile,
    load_profiles,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


FULL_PROFILE = """\
name: argocd
extends: base
node_image: kindest/node:v1.31.4
install:
  - helm_repo:
      name: argo
      url: https://argoproj.github.io/argo-helm
  - helm_install:
      name: argocd
      chart: argo/argo-cd
      namespace: argocd
      version: 7.6.12
wait_for:
  - kind: Deployment
    namespace: argocd
    name: argocd-server
    condition: Available
    timeout: 3m
prebuilt_image: kubelm-argocd-base:v1
"""


def test_load_full_profile(tmp_path: Path) -> None:
    p = _write(tmp_path / "argocd.yaml", FULL_PROFILE)
    prof = load_profile(p)

    assert prof.name == "argocd"
    assert prof.extends == "base"
    assert prof.node_image == "kindest/node:v1.31.4"
    assert prof.prebuilt_image == "kubelm-argocd-base:v1"
    assert prof.source_path == p

    assert len(prof.install) == 2
    assert prof.install[0].helm_repo is not None
    assert prof.install[0].helm_repo.name == "argo"
    assert prof.install[1].helm_install is not None
    assert prof.install[1].helm_install.chart == "argo/argo-cd"
    assert prof.install[1].helm_install.version == "7.6.12"

    assert len(prof.wait_for) == 1
    wf = prof.wait_for[0]
    assert wf.kind == "Deployment"
    assert wf.condition == "Available"
    assert wf.timeout_seconds == 180


def test_minimal_profile(tmp_path: Path) -> None:
    p = _write(tmp_path / "p.yaml", "name: tiny\n")
    prof = load_profile(p)
    assert prof.name == "tiny"
    assert prof.extends is None
    assert prof.install == []
    assert prof.wait_for == []
    assert prof.node_image is None
    assert prof.prebuilt_image is None


def test_install_step_must_set_exactly_one_action(tmp_path: Path) -> None:
    both = _write(
        tmp_path / "both.yaml",
        """\
name: bad
install:
  - helm_repo: { name: r, url: https://example.com }
    helm_install: { name: x, chart: c, namespace: n }
""",
    )
    with pytest.raises(ProfileParseError, match="exactly one"):
        load_profile(both)

    neither = _write(tmp_path / "neither.yaml", "name: bad\ninstall:\n  - {}\n")
    with pytest.raises(ProfileParseError, match="exactly one"):
        load_profile(neither)


def test_missing_required_field_raises(tmp_path: Path) -> None:
    p = _write(tmp_path / "p.yaml", "extends: base\n")
    with pytest.raises(ProfileParseError, match="name"):
        load_profile(p)


def test_invalid_yaml_raises_with_path(tmp_path: Path) -> None:
    p = _write(tmp_path / "bad.yaml", "name: [unclosed\n")
    with pytest.raises(ProfileParseError) as excinfo:
        load_profile(p)
    assert "bad.yaml" in str(excinfo.value)


def test_load_profiles_directory_returns_dict(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", "name: base\n")
    _write(tmp_path / "argocd.yaml", "name: argocd\nextends: base\n")
    _write(tmp_path / "ignored.txt", "not a profile")

    profs = load_profiles(tmp_path)
    assert set(profs) == {"base", "argocd"}
    assert profs["argocd"].extends == "base"


def test_load_profiles_rejects_duplicate_names(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "name: dup\n")
    _write(tmp_path / "b.yaml", "name: dup\n")
    with pytest.raises(ProfileParseError, match="duplicate"):
        load_profiles(tmp_path)


def test_compose_no_extends_is_identity(tmp_path: Path) -> None:
    base = Profile(name="base", node_image="img:1")
    composed = compose_profile("base", {"base": base})
    assert composed.name == "base"
    assert composed.node_image == "img:1"


def test_compose_inherits_parent_node_image() -> None:
    base = Profile(name="base", node_image="kindest/node:v1.31")
    child = Profile(name="argocd", extends="base")
    composed = compose_profile("argocd", {"base": base, "argocd": child})
    assert composed.node_image == "kindest/node:v1.31"


def test_compose_child_overrides_node_image() -> None:
    base = Profile(name="base", node_image="kindest/node:v1.30")
    child = Profile(name="argocd", extends="base", node_image="kindest/node:v1.31")
    composed = compose_profile("argocd", {"base": base, "argocd": child})
    assert composed.node_image == "kindest/node:v1.31"


def test_compose_appends_install_parent_first(tmp_path: Path) -> None:
    _write(
        tmp_path / "base.yaml",
        """\
name: base
install:
  - helm_repo: { name: stable, url: https://stable.example/charts }
""",
    )
    _write(
        tmp_path / "argocd.yaml",
        """\
name: argocd
extends: base
install:
  - helm_repo:    { name: argo, url: https://argoproj.github.io/argo-helm }
  - helm_install: { name: argocd, chart: argo/argo-cd, namespace: argocd }
""",
    )
    profs = load_profiles(tmp_path)
    composed = compose_profile("argocd", profs)
    names = []
    for step in composed.install:
        if step.helm_repo:
            names.append(("repo", step.helm_repo.name))
        elif step.helm_install:
            names.append(("install", step.helm_install.name))
    assert names == [("repo", "stable"), ("repo", "argo"), ("install", "argocd")]


def test_compose_appends_wait_for(tmp_path: Path) -> None:
    _write(
        tmp_path / "base.yaml",
        """\
name: base
wait_for:
  - { kind: Deployment, namespace: kube-system, name: coredns }
""",
    )
    _write(
        tmp_path / "argocd.yaml",
        """\
name: argocd
extends: base
wait_for:
  - { kind: Deployment, namespace: argocd, name: argocd-server }
""",
    )
    profs = load_profiles(tmp_path)
    composed = compose_profile("argocd", profs)
    names = [(w.namespace, w.name) for w in composed.wait_for]
    assert names == [("kube-system", "coredns"), ("argocd", "argocd-server")]


def test_compose_three_level_chain() -> None:
    a = Profile(name="a", node_image="img-a")
    b = Profile(name="b", extends="a")
    c = Profile(name="c", extends="b", node_image="img-c")
    composed = compose_profile("c", {"a": a, "b": b, "c": c})
    assert composed.node_image == "img-c"
    assert composed.name == "c"


def test_compose_unknown_extends_raises() -> None:
    child = Profile(name="argocd", extends="missing")
    with pytest.raises(ProfileCycleError, match="missing"):
        compose_profile("argocd", {"argocd": child})


def test_compose_cycle_raises() -> None:
    a = Profile(name="a", extends="b")
    b = Profile(name="b", extends="a")
    with pytest.raises(ProfileCycleError, match="cycle"):
        compose_profile("a", {"a": a, "b": b})


def test_compose_prebuilt_image_inheritance() -> None:
    parent = Profile(name="base", prebuilt_image="img:base")
    child_no_override = Profile(name="extender", extends="base")
    composed = compose_profile("extender", {"base": parent, "extender": child_no_override})
    assert composed.prebuilt_image == "img:base"

    child_override = Profile(name="extender2", extends="base", prebuilt_image="img:extender")
    composed2 = compose_profile("extender2", {"base": parent, "extender2": child_override})
    assert composed2.prebuilt_image == "img:extender"
