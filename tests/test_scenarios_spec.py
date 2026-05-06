from __future__ import annotations

from pathlib import Path

import pytest

from eval.scenarios.spec import (
    DEFAULT_SETTLE_TIMEOUT,
    ScenarioParseError,
    load_scenario,
    load_scenarios,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


FULL_SCENARIO = """\
id: pod-crashloop-001
profile: base
description: Pod crashes on startup due to a missing required env var.
goal: "Why is my pod crashing?"
setup:
  - apply_inline: |
      apiVersion: v1
      kind: Pod
      metadata:
        name: crash-pod
        namespace: scenario-pod-crashloop-001
      spec:
        containers:
          - name: app
            image: busybox
            command: ["false"]
settle:
  - wait_for_status:
      kind: Pod
      namespace: scenario-pod-crashloop-001
      name: crash-pod
      reason: CrashLoopBackOff
      timeout: 90s
expected:
  reference_calls:
    must_include:
      - { name: list-resources, args_match: { resourceType: pods } }
      - { name: get-logs,       args_match: { podName: crash-pod } }
    forbidden:
      - { name: add-filters }
  conclusion_rubric:
    must_mention: ["CrashLoopBackOff", "crash-pod"]
    must_not_mention: ["DeadlineExceeded"]
    semantic_intent: "Identifies the pod, the failure mode, and the cause."
"""


def test_load_full_scenario(tmp_path: Path) -> None:
    p = _write(tmp_path / "s.yaml", FULL_SCENARIO)
    scn = load_scenario(p)

    assert scn.id == "pod-crashloop-001"
    assert scn.profile == "base"
    assert "missing required env var" in scn.description
    assert scn.goal.startswith("Why")
    assert scn.source_path == p

    assert len(scn.setup) == 1
    assert scn.setup[0].apply_inline is not None
    assert "kind: Pod" in scn.setup[0].apply_inline
    assert scn.setup[0].apply_file is None

    assert len(scn.settle) == 1
    s0 = scn.settle[0]
    assert s0.kind == "Pod"
    assert s0.reason == "CrashLoopBackOff"
    assert s0.timeout_seconds == 90

    rc = scn.expected.reference_calls
    assert [c.name for c in rc.must_include] == ["list-resources", "get-logs"]
    assert rc.must_include[0].args_match == {"resourceType": "pods"}
    assert rc.must_include[1].args_match == {"podName": "crash-pod"}
    assert [c.name for c in rc.forbidden] == ["add-filters"]

    cr = scn.expected.conclusion_rubric
    assert cr.must_mention == ["CrashLoopBackOff", "crash-pod"]
    assert cr.must_not_mention == ["DeadlineExceeded"]
    assert "failure mode" in cr.semantic_intent


def test_minimal_scenario_uses_defaults(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "s.yaml",
        """\
id: minimal-001
profile: base
goal: "Anything?"
""",
    )
    scn = load_scenario(p)
    assert scn.setup == []
    assert scn.settle == []
    assert scn.expected.reference_calls.must_include == []
    assert scn.expected.conclusion_rubric.must_mention == []


def test_missing_required_field_raises(tmp_path: Path) -> None:
    p = _write(tmp_path / "s.yaml", "id: only-id\nprofile: base\n")
    with pytest.raises(ScenarioParseError, match="goal"):
        load_scenario(p)


def test_setup_step_must_have_exactly_one_action(tmp_path: Path) -> None:
    both = _write(
        tmp_path / "both.yaml",
        """\
id: x
profile: base
goal: g
setup:
  - apply_inline: "..."
    apply_file: manifests/foo.yaml
""",
    )
    with pytest.raises(ScenarioParseError, match="exactly one"):
        load_scenario(both)

    neither = _write(
        tmp_path / "neither.yaml",
        """\
id: x
profile: base
goal: g
setup:
  - {}
""",
    )
    with pytest.raises(ScenarioParseError, match="exactly one"):
        load_scenario(neither)


def test_unknown_settle_step_kind_raises(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "s.yaml",
        """\
id: x
profile: base
goal: g
settle:
  - unknown_step: {}
""",
    )
    with pytest.raises(ScenarioParseError, match="wait_for_status"):
        load_scenario(p)


def test_duration_parsing_accepts_seconds_minutes_hours_and_int(tmp_path: Path) -> None:
    template = """\
id: x
profile: base
goal: g
settle:
  - wait_for_status:
      kind: Pod
      namespace: ns
      name: pod
      timeout: {timeout}
"""
    cases = {"60s": 60, "5m": 300, "1h": 3600, "45": 45, 30: 30}
    for raw, expected in cases.items():
        p = _write(tmp_path / f"s-{raw}.yaml", template.format(timeout=raw))
        assert load_scenario(p).settle[0].timeout_seconds == expected


def test_default_settle_timeout_when_omitted(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "s.yaml",
        """\
id: x
profile: base
goal: g
settle:
  - wait_for_status:
      kind: Pod
      namespace: ns
      name: pod
""",
    )
    assert load_scenario(p).settle[0].timeout_seconds == DEFAULT_SETTLE_TIMEOUT


def test_invalid_yaml_raises_with_path(tmp_path: Path) -> None:
    p = _write(tmp_path / "bad.yaml", "id: [unclosed\n")
    with pytest.raises(ScenarioParseError) as excinfo:
        load_scenario(p)
    assert "bad.yaml" in str(excinfo.value)


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    p = _write(tmp_path / "list.yaml", "- one\n- two\n")
    with pytest.raises(ScenarioParseError, match="mapping"):
        load_scenario(p)


def test_load_scenarios_from_directory(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "id: a-1\nprofile: base\ngoal: ga\n")
    _write(tmp_path / "b.yaml", "id: b-1\nprofile: base\ngoal: gb\n")
    _write(tmp_path / "ignored.txt", "not a scenario")

    scenarios = load_scenarios(tmp_path)
    assert [s.id for s in scenarios] == ["a-1", "b-1"]


def test_load_scenarios_rejects_duplicate_ids(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "id: dup\nprofile: base\ngoal: a\n")
    _write(tmp_path / "b.yaml", "id: dup\nprofile: base\ngoal: b\n")
    with pytest.raises(ScenarioParseError, match="duplicate"):
        load_scenarios(tmp_path)


def test_load_scenarios_rejects_non_directory(tmp_path: Path) -> None:
    p = _write(tmp_path / "f.yaml", "id: x\nprofile: base\ngoal: g\n")
    with pytest.raises(ScenarioParseError, match="not a directory"):
        load_scenarios(p)
