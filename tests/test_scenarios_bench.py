from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.scenarios.bench import (
    BENCH_SCHEMA_VERSION,
    BenchParseError,
    ModelConfig,
    RunRecord,
    _model_summaries,
    _record_from_results,
    format_summary_table,
    load_models,
)
from eval.scenarios.spec import (
    ConclusionRubric,
    ExpectedOutcome,
    ReferenceCall,
    ReferenceCalls,
    Scenario,
)

SHAPE_A_YAML = """\
- name: llama3.2-3b
  backend_url: http://localhost:11434/v1
  model: llama3.2:3b

- name: gpt-4o-mini
  backend_url: https://api.openai.com/v1
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY
  temperature: 0.5
  max_tokens: 1024
"""


def test_load_models_parses_full_entry(tmp_path: Path) -> None:
    p = tmp_path / "m.yaml"
    p.write_text(SHAPE_A_YAML)
    models = load_models(p)

    assert len(models) == 2
    assert models[0].name == "llama3.2-3b"
    assert models[0].backend_url == "http://localhost:11434/v1"
    assert models[0].model == "llama3.2:3b"
    assert models[0].api_key_env is None
    assert models[0].temperature == 0.0
    assert models[0].max_tokens == 2048

    assert models[1].name == "gpt-4o-mini"
    assert models[1].api_key_env == "OPENAI_API_KEY"
    assert models[1].temperature == 0.5
    assert models[1].max_tokens == 1024


def test_load_models_rejects_non_list(tmp_path: Path) -> None:
    p = tmp_path / "m.yaml"
    p.write_text("name: oops\n")
    with pytest.raises(BenchParseError, match="YAML list"):
        load_models(p)


def test_load_models_rejects_missing_required_field(tmp_path: Path) -> None:
    p = tmp_path / "m.yaml"
    p.write_text("- name: foo\n")
    with pytest.raises(BenchParseError, match="backend_url"):
        load_models(p)


def test_load_models_rejects_duplicate_names(tmp_path: Path) -> None:
    p = tmp_path / "m.yaml"
    p.write_text(
        "- name: dup\n  backend_url: http://x\n  model: a\n"
        "- name: dup\n  backend_url: http://y\n  model: b\n"
    )
    with pytest.raises(BenchParseError, match="duplicate"):
        load_models(p)


def test_resolve_api_key_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_KEY", "sk-abc")
    cfg = ModelConfig(name="x", backend_url="http://x", model="x", api_key_env="MY_KEY")
    assert cfg.resolve_api_key() == "sk-abc"


def test_resolve_api_key_returns_none_when_env_var_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MY_KEY", raising=False)
    cfg = ModelConfig(name="x", backend_url="http://x", model="x", api_key_env="MY_KEY")
    assert cfg.resolve_api_key() is None


def _scenario(scn_id: str = "s-1") -> Scenario:
    return Scenario(
        id=scn_id,
        profile="base",
        goal="g",
        expected=ExpectedOutcome(
            reference_calls=ReferenceCalls(must_include=[ReferenceCall(name="x")]),
            conclusion_rubric=ConclusionRubric(must_mention=["x"]),
        ),
    )


def _passing_results() -> dict:
    return {
        "schema_report": {
            "total_calls": 2,
            "valid_calls": 2,
            "name_hallucinations": 0,
            "argument_hallucinations": 0,
        },
        "grounding_report": {"has_grounding_failure": False},
        "termination_report": {"label": "complete"},
        "reference_calls_report": {"passed": True},
        "conclusion_rubric_report": {"passed": True},
        "totals": {"model_latency_ms": 250.0},
    }


def _failing_results() -> dict:
    return {
        "schema_report": {
            "total_calls": 3,
            "valid_calls": 1,
            "name_hallucinations": 1,
            "argument_hallucinations": 1,
        },
        "grounding_report": {"has_grounding_failure": True},
        "termination_report": {"label": "premature"},
        "reference_calls_report": {"passed": False},
        "conclusion_rubric_report": {"passed": False},
        "totals": {"model_latency_ms": 100.0},
    }


def test_record_from_results_extracts_passing(tmp_path: Path) -> None:
    cfg = ModelConfig(name="m", backend_url="u", model="m1")
    rec = _record_from_results(cfg, _scenario(), "run-1", _passing_results(), tmp_path, 12.5)
    assert rec.termination_label == "complete"
    assert rec.schema_passed is True
    assert rec.schema_name_halluc == 0
    assert rec.grounding_failed is False
    assert rec.reference_calls_passed is True
    assert rec.conclusion_rubric_passed is True
    assert rec.model_latency_ms == 250.0
    assert rec.duration_seconds == 12.5


def test_record_from_results_extracts_failures(tmp_path: Path) -> None:
    cfg = ModelConfig(name="m", backend_url="u", model="m1")
    rec = _record_from_results(cfg, _scenario(), "run-2", _failing_results(), tmp_path, 7.0)
    assert rec.termination_label == "premature"
    assert rec.schema_passed is False
    assert rec.schema_name_halluc == 1
    assert rec.schema_arg_halluc == 1
    assert rec.grounding_failed is True


def test_model_summaries_aggregates_per_model() -> None:
    models = [
        ModelConfig(name="a", backend_url="u", model="m"),
        ModelConfig(name="b", backend_url="u", model="m"),
    ]
    runs = [
        RunRecord(
            model="a",
            scenario="s1",
            run_id="r1",
            termination_label="complete",
            schema_passed=True,
            schema_name_halluc=0,
            schema_arg_halluc=0,
            grounding_failed=False,
            reference_calls_passed=True,
            conclusion_rubric_passed=True,
            model_latency_ms=100.0,
            duration_seconds=5.0,
        ),
        RunRecord(
            model="a",
            scenario="s2",
            run_id="r2",
            termination_label="premature",
            schema_passed=False,
            schema_name_halluc=2,
            schema_arg_halluc=1,
            grounding_failed=True,
            reference_calls_passed=False,
            conclusion_rubric_passed=False,
            model_latency_ms=50.0,
            duration_seconds=3.0,
        ),
        RunRecord(
            model="b",
            scenario="s1",
            run_id="r3",
            error="boom",
            duration_seconds=1.0,
        ),
    ]
    s = _model_summaries(runs, models)
    assert s["a"]["scenarios_attempted"] == 2
    assert s["a"]["termination_complete"] == 1
    assert s["a"]["schema_passed"] == 1
    assert s["a"]["name_hallucinations_total"] == 2
    assert s["a"]["argument_hallucinations_total"] == 1
    assert s["a"]["grounding_failures"] == 1
    assert s["a"]["reference_calls_passed"] == 1
    assert s["a"]["scenarios_errored"] == 0
    assert s["a"]["total_model_latency_ms"] == 150.0

    assert s["b"]["scenarios_attempted"] == 1
    assert s["b"]["scenarios_errored"] == 1
    assert s["b"]["termination_complete"] == 0


def test_format_summary_table_renders() -> None:
    summary = {
        "model_summaries": {
            "alpha": {
                "scenarios_attempted": 3,
                "termination_complete": 2,
                "schema_passed": 3,
                "name_hallucinations_total": 0,
                "argument_hallucinations_total": 1,
                "grounding_failures": 0,
                "reference_calls_passed": 3,
                "conclusion_rubric_passed": 2,
                "scenarios_errored": 0,
                "total_duration_seconds": 42.0,
            }
        }
    }
    out = format_summary_table(summary)
    assert "alpha" in out
    assert "2/3" in out
    assert "model" in out
    assert "complete" in out


def test_shape_a_models_file_loads() -> None:
    """Ship-time validation of the bundled Shape A config."""
    from eval.scenarios.bench import load_models as _load

    repo_models = (
        Path(__file__).resolve().parent.parent
        / "eval"
        / "scenarios"
        / "benchmarks"
        / "shape-a.yaml"
    )
    models = _load(repo_models)
    names = {m.name for m in models}
    assert {"llama3.2-3b", "gpt-4o-mini"} <= names


def test_shape_b_models_file_loads() -> None:
    """Ship-time validation of the bundled Shape B (4-model) config."""
    from eval.scenarios.bench import load_models as _load

    repo_models = (
        Path(__file__).resolve().parent.parent
        / "eval"
        / "scenarios"
        / "benchmarks"
        / "shape-b.yaml"
    )
    models = _load(repo_models)
    names = {m.name for m in models}
    assert {"llama3.2-3b", "qwen2.5-7b", "qwen2.5-32b", "gpt-4o"} <= names


def test_bench_schema_version_constant() -> None:
    assert BENCH_SCHEMA_VERSION == 1


def test_runrecord_serializes_to_json() -> None:
    """run_bench writes summary.json via dataclass asdict; ensure RunRecord is friendly."""
    from dataclasses import asdict

    rec = RunRecord(model="a", scenario="s", run_id="r", duration_seconds=1.0)
    assert json.dumps(asdict(rec)) is not None
