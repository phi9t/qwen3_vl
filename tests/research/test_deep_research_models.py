"""Tests for deep-research model parsing."""

from __future__ import annotations

import pytest

import research.deep_research_models as models


def test_deep_research_request_defaults() -> None:
    request = models.DeepResearchRequest(topic="Temporal AI workflows")

    assert request.topic == "Temporal AI workflows"
    assert request.max_subtopics == 5
    assert request.queries_per_subtopic == 3
    assert request.max_search_results == 5


def test_parse_subtopic_plan_accepts_compact_json() -> None:
    plan = models.parse_subtopic_plan(
        '{"subtopics": ["durable execution", "activity retries"],'
        ' "rationale": "covers Temporal basics"}'
    )

    assert plan.subtopics == ["durable execution", "activity retries"]
    assert plan.rationale == "covers Temporal basics"


def test_parse_search_query_plan_rejects_empty_queries() -> None:
    with pytest.raises(models.DeepResearchParseError, match="queries"):
        models.parse_search_query_plan('{"queries": []}')


def test_parse_search_result_requires_summary() -> None:
    with pytest.raises(models.DeepResearchParseError, match="summary"):
        models.parse_search_result('{"query": "Temporal", "sources": []}')


def test_report_as_temporal_payload_uses_plain_dicts() -> None:
    report = models.DeepResearchReport(
        topic="Temporal",
        markdown="# Temporal",
        successful_queries=["Temporal activity retries"],
        failed_queries=[{"query": "bad", "error": "timeout"}],
    )

    assert report.as_dict() == {
        "topic": "Temporal",
        "markdown": "# Temporal",
        "successful_queries": ["Temporal activity retries"],
        "failed_queries": [{"query": "bad", "error": "timeout"}],
    }
