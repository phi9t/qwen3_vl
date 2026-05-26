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


def test_parse_subtopic_plan_rejects_malformed_json() -> None:
    with pytest.raises(models.DeepResearchParseError, match="invalid JSON"):
        models.parse_subtopic_plan('{subtopics: ["Temporal"]}')


def test_parse_search_query_plan_rejects_missing_queries_key() -> None:
    with pytest.raises(models.DeepResearchParseError, match="queries"):
        models.parse_search_query_plan("{}")


def test_parse_search_query_plan_rejects_non_string_elements() -> None:
    with pytest.raises(models.DeepResearchParseError, match="queries"):
        models.parse_search_query_plan('{"queries": ["Temporal", 123]}')


def test_parse_search_result_rejects_whitespace_query() -> None:
    with pytest.raises(models.DeepResearchParseError, match="query"):
        models.parse_search_result(
            '{"query": "   ", "summary": "A short summary", "sources": []}'
        )


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


def test_deep_research_request_invalid_max_subtopics() -> None:
    with pytest.raises(ValueError, match="max_subtopics"):
        models.DeepResearchRequest(topic="Temporal", max_subtopics=0)
