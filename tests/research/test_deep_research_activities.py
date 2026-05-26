"""Tests for COCO-backed deep research activities."""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest
import temporalio.exceptions

import research
import research.coco_cli
import research.deep_research_activities as activities


def test_generate_subtopics_with_coco_parses_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_coco(
        *,
        prompt: str,
        phase: str,
        worktree: str,
        workspace: object | None,
        **kwargs: object,
    ) -> SimpleNamespace:
        captured["phase"] = phase
        captured["worktree"] = worktree
        captured["workspace"] = workspace
        captured["prompt"] = prompt
        return SimpleNamespace(stdout='{"subtopics": ["a", "b"], "rationale": "r"}')

    monkeypatch.setattr(activities.research.coco_cli, "run_coco", fake_run_coco)

    result = activities.generate_subtopics_with_coco(
        {"topic": "Temporal workflows", "max_subtopics": 4, "workspace": tmp_path},
    )

    assert result == {"subtopics": ["a", "b"], "rationale": "r"}
    assert captured["phase"] == "subtopics"
    assert captured["worktree"] == "deep-research-subtopics"
    assert captured["workspace"] == tmp_path


def test_generate_search_queries_with_coco_uses_subtopics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_coco(
        *,
        prompt: str,
        phase: str,
        worktree: str,
        workspace: object | None,
        **kwargs: object,
    ) -> SimpleNamespace:
        captured["phase"] = phase
        captured["prompt"] = prompt
        return SimpleNamespace(stdout='{"queries": ["query-one", "query-two"]}')

    monkeypatch.setattr(activities.research.coco_cli, "run_coco", fake_run_coco)

    result = activities.generate_search_queries_with_coco(
        {"subtopics": ["subtopic-a", "subtopic-b"]}
    )

    assert result == {"queries": ["query-one", "query-two"]}
    assert captured["phase"] == "search_queries"
    assert "subtopic-a" in str(captured["prompt"])


def test_search_web_with_coco_parses_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_coco(
        *,
        prompt: str,
        phase: str,
        worktree: str,
        workspace: object | None,
        **kwargs: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            stdout=(
                '{"query": "Temporal", "summary": "Temporal has durable execution", '
                '"sources": ["https://example.com"]}'
            )
        )

    monkeypatch.setattr(activities.research.coco_cli, "run_coco", fake_run_coco)

    result = activities.search_web_with_coco({"query": "Temporal"})

    assert result["query"] == "Temporal"
    assert result["summary"] == "Temporal has durable execution"
    assert result["sources"] == ["https://example.com"]


def test_synthesize_report_with_coco_wraps_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    markdown = "\n  \n# Topic Report\n\nThis is the report body\n  \n"

    def fake_run_coco(
        *,
        prompt: str,
        phase: str,
        worktree: str,
        workspace: object | None,
        **kwargs: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(stdout=markdown)

    monkeypatch.setattr(activities.research.coco_cli, "run_coco", fake_run_coco)

    result = activities.synthesize_report_with_coco(
        {
            "topic": "Temporal",
            "research": [
                {"query": "query-1", "summary": "s", "sources": []},
                {"query": "query-2", "summary": "s", "sources": []},
            ],
            "failed_queries": [{"query": "bad", "error": "timeout"}],
        }
    )

    assert result["topic"] == "Temporal"
    assert result["markdown"] == markdown
    assert result["successful_queries"] == ["query-1", "query-2"]
    assert result["failed_queries"] == [{"query": "bad", "error": "timeout"}]


def test_synthesize_report_with_coco_empty_markdown_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_coco(
        *,
        prompt: str,
        phase: str,
        worktree: str,
        workspace: object | None,
        **kwargs: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(stdout="\n   \n\t")

    monkeypatch.setattr(activities.research.coco_cli, "run_coco", fake_run_coco)

    with pytest.raises(temporalio.exceptions.ApplicationError) as exc_info:
        activities.synthesize_report_with_coco(
            {
                "topic": "Temporal",
                "research": [{"query": "q", "summary": "s", "sources": []}],
            }
        )

    assert exc_info.value.type == "deep_research_empty_report"
    assert exc_info.value.non_retryable is False


def test_parse_failure_raises_application_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_coco(
        *,
        prompt: str,
        phase: str,
        worktree: str,
        workspace: object | None,
        **kwargs: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(stdout='{"queries": []}')

    monkeypatch.setattr(activities.research.coco_cli, "run_coco", fake_run_coco)

    with pytest.raises(temporalio.exceptions.ApplicationError) as exc_info:
        activities.generate_search_queries_with_coco({"subtopics": ["a"]})

    assert exc_info.value.type == "deep_research_parse_failed"
    assert exc_info.value.non_retryable is False


def test_invalid_payload_missing_topic_is_non_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_coco(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout='{"subtopics": ["a"]}')

    monkeypatch.setattr(activities.research.coco_cli, "run_coco", fake_run_coco)

    with pytest.raises(temporalio.exceptions.ApplicationError) as exc_info:
        activities.generate_subtopics_with_coco({})

    assert exc_info.value.type == "deep_research_invalid_payload"
    assert exc_info.value.non_retryable is True


def test_coco_execution_error_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_coco(
        *,
        prompt: str,
        phase: str,
        worktree: str,
        workspace: object | None,
        **kwargs: object,
    ) -> SimpleNamespace:
        raise research.coco_cli.CocoExecutionError(
            phase="subtopics",
            returncode=2,
            stdout="",
            stderr="boom",
            message="coco failed",
        )

    monkeypatch.setattr(activities.research.coco_cli, "run_coco", fake_run_coco)

    with pytest.raises(temporalio.exceptions.ApplicationError) as exc_info:
        activities.generate_subtopics_with_coco({"topic": "Temporal"})

    assert exc_info.value.type == "deep_research_coco_failed"


def test_synthesize_empty_research_is_non_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_coco(
        *,
        prompt: str,
        phase: str,
        worktree: str,
        workspace: object | None,
        **kwargs: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(stdout="# Report\nbody")

    monkeypatch.setattr(activities.research.coco_cli, "run_coco", fake_run_coco)

    with pytest.raises(temporalio.exceptions.ApplicationError) as exc_info:
        activities.synthesize_report_with_coco({"topic": "Temporal", "research": []})

    assert exc_info.value.type == "deep_research_empty_research"
    assert exc_info.value.non_retryable is True
