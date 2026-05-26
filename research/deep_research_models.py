"""Data contracts for COCO-backed deep research."""

from __future__ import annotations

import dataclasses
import json
from typing import Any


JsonDict = dict[str, Any]


class DeepResearchParseError(ValueError):
    """Raised when a COCO response does not match the expected shape."""


@dataclasses.dataclass(frozen=True)
class DeepResearchRequest:
    """Input for one deep-research workflow run."""

    topic: str
    max_subtopics: int = 5
    queries_per_subtopic: int = 3
    max_search_results: int = 5

    def as_dict(self) -> JsonDict:
        """Return a Temporal payload-safe representation."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SubtopicPlan:
    """COCO-generated subtopic plan."""

    subtopics: list[str]
    rationale: str = ""

    def as_dict(self) -> JsonDict:
        """Return a Temporal payload-safe representation."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SearchQueryPlan:
    """COCO-generated search query plan."""

    queries: list[str]

    def as_dict(self) -> JsonDict:
        """Return a Temporal payload-safe representation."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SearchResult:
    """Summarized result for one COCO search query."""

    query: str
    summary: str
    sources: list[str]
    raw_output_path: str = ""

    def as_dict(self) -> JsonDict:
        """Return a Temporal payload-safe representation."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DeepResearchReport:
    """Final Markdown report and workflow result metadata."""

    topic: str
    markdown: str
    successful_queries: list[str]
    failed_queries: list[dict[str, str]]

    def as_dict(self) -> JsonDict:
        """Return a Temporal payload-safe representation."""
        return dataclasses.asdict(self)


def parse_subtopic_plan(raw_output: str) -> SubtopicPlan:
    """Parse COCO JSON into a subtopic plan."""
    payload = _json_object(raw_output)
    subtopics = _string_list(payload, "subtopics")
    rationale = _optional_string(payload, "rationale")
    return SubtopicPlan(subtopics=subtopics, rationale=rationale)


def parse_search_query_plan(raw_output: str) -> SearchQueryPlan:
    """Parse COCO JSON into search queries."""
    payload = _json_object(raw_output)
    return SearchQueryPlan(queries=_string_list(payload, "queries"))


def parse_search_result(raw_output: str) -> SearchResult:
    """Parse COCO JSON into one search result."""
    payload = _json_object(raw_output)
    query = _required_string(payload, "query")
    summary = _required_string(payload, "summary")
    sources = _string_list(payload, "sources", allow_empty=True)
    raw_output_path = _optional_string(payload, "raw_output_path")
    return SearchResult(
        query=query,
        summary=summary,
        sources=sources,
        raw_output_path=raw_output_path,
    )


def _json_object(raw_output: str) -> JsonDict:
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise DeepResearchParseError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise DeepResearchParseError("expected JSON object")
    return payload


def _required_string(payload: JsonDict, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DeepResearchParseError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_string(payload: JsonDict, key: str) -> str:
    value = payload.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise DeepResearchParseError(f"{key} must be a string")
    return value.strip()


def _string_list(
    payload: JsonDict,
    key: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise DeepResearchParseError(f"{key} must be a list of strings")
    strings = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if len(strings) != len(value):
        raise DeepResearchParseError(f"{key} must contain only non-empty strings")
    if not strings and not allow_empty:
        raise DeepResearchParseError(f"{key} must not be empty")
    return strings
