from __future__ import annotations

from pathlib import Path

from eval.metrics import evaluate_conclusion_rubric
from eval.scenarios.spec import ConclusionRubric
from eval.trajectory import TrajectoryRecorder, load_trajectory


def _record(path: Path, *texts: str) -> list[dict]:
    with TrajectoryRecorder(path=path) as rec:
        for t in texts:
            rec.assistant(text=t)
        rec.end("complete")
    return load_trajectory(path)


def test_all_must_mention_present(tmp_path: Path) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        "The pod crash-pod hit CrashLoopBackOff repeatedly.",
    )
    rubric = ConclusionRubric(must_mention=["CrashLoopBackOff", "crash-pod"])
    report = evaluate_conclusion_rubric(events, rubric)
    assert report.passed
    assert report.missing_mentions == []


def test_missing_mention_fails(tmp_path: Path) -> None:
    events = _record(tmp_path / "t.jsonl", "Just CrashLoopBackOff, no name.")
    rubric = ConclusionRubric(must_mention=["CrashLoopBackOff", "crash-pod"])
    report = evaluate_conclusion_rubric(events, rubric)
    assert not report.passed
    assert report.missing_mentions == ["crash-pod"]


def test_must_not_mention_present_fails(tmp_path: Path) -> None:
    events = _record(tmp_path / "t.jsonl", "DeadlineExceeded happened.")
    rubric = ConclusionRubric(must_not_mention=["DeadlineExceeded"])
    report = evaluate_conclusion_rubric(events, rubric)
    assert not report.passed
    assert report.forbidden_mentions == ["DeadlineExceeded"]


def test_case_insensitive_substring(tmp_path: Path) -> None:
    events = _record(tmp_path / "t.jsonl", "the pod crashed with crashloopbackoff")
    rubric = ConclusionRubric(must_mention=["CrashLoopBackOff"])
    assert evaluate_conclusion_rubric(events, rubric).passed


def test_uses_only_last_assistant_text(tmp_path: Path) -> None:
    events = _record(
        tmp_path / "t.jsonl",
        "Initial guess: maybe foo.",
        "After investigation: the pod failed with CrashLoopBackOff.",
    )
    rubric = ConclusionRubric(must_mention=["CrashLoopBackOff"], must_not_mention=["Initial guess"])
    report = evaluate_conclusion_rubric(events, rubric)
    assert report.passed
    assert "After investigation" in report.conclusion_text


def test_no_assistant_event_returns_empty_conclusion(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    with TrajectoryRecorder(path=path) as rec:
        rec.end("incomplete")
    events = load_trajectory(path)
    rubric = ConclusionRubric(must_mention=["x"])
    report = evaluate_conclusion_rubric(events, rubric)
    assert report.conclusion_text == ""
    assert report.missing_mentions == ["x"]
    assert not report.passed


def test_semantic_intent_preserved_in_report(tmp_path: Path) -> None:
    events = _record(tmp_path / "t.jsonl", "ok")
    rubric = ConclusionRubric(semantic_intent="Identifies the root cause")
    report = evaluate_conclusion_rubric(events, rubric)
    assert report.semantic_intent == "Identifies the root cause"


def test_empty_rubric_passes(tmp_path: Path) -> None:
    events = _record(tmp_path / "t.jsonl", "anything")
    assert evaluate_conclusion_rubric(events, ConclusionRubric()).passed
