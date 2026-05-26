"""Temporal activities for COCO-backed deep research phases."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import temporalio.activity
from temporalio.exceptions import ApplicationError

import research
import research.coco_cli
import research.deep_research_models as models


_SUBTOPICS_WORKTREE = "deep-research-subtopics"
_SEARCH_QUERIES_WORKTREE = "deep-research-search-queries"
_SEARCH_WEB_WORKTREE = "deep-research-search"
_SYNTHESIZE_WORKTREE = "deep-research-synthesize"


@temporalio.activity.defn
def generate_subtopics_with_coco(request: dict[str, Any]) -> dict[str, Any]:
    """Generate subtopics for a research topic using COCO."""
    payload = _coerce_payload(request)
    topic = _require_non_empty_str(payload, "topic")
    max_subtopics = _require_positive_int(payload, "max_subtopics", 5)
    workspace = _optional_workspace(payload, "workspace")

    prompt = _build_subtopic_prompt(topic, max_subtopics)

    result = _run_coco(
        prompt=prompt,
        phase="subtopics",
        worktree=_SUBTOPICS_WORKTREE,
        workspace=workspace,
    )

    try:
        return models.parse_subtopic_plan(result.stdout).as_dict()
    except models.DeepResearchParseError as exc:
        _raise_parse_error(exc)


@temporalio.activity.defn
def generate_search_queries_with_coco(plan: dict[str, Any]) -> dict[str, Any]:
    """Generate search queries for a set of subtopics using COCO."""
    payload = _coerce_payload(plan)
    subtopics = _require_non_empty_string_list(payload, "subtopics")
    queries_per_subtopic = _require_positive_int(
        payload,
        "queries_per_subtopic",
        3,
    )
    workspace = _optional_workspace(payload, "workspace")

    prompt = _build_search_queries_prompt(subtopics, queries_per_subtopic)
    result = _run_coco(
        prompt=prompt,
        phase="search_queries",
        worktree=_SEARCH_QUERIES_WORKTREE,
        workspace=workspace,
    )

    try:
        return models.parse_search_query_plan(result.stdout).as_dict()
    except models.DeepResearchParseError as exc:
        _raise_parse_error(exc)


@temporalio.activity.defn
def search_web_with_coco(query_payload: dict[str, Any]) -> dict[str, Any]:
    """Search the web for one COCO-generated query and summarize the result."""
    payload = _coerce_payload(query_payload)
    query = _require_non_empty_str(payload, "query")
    max_search_results = _require_positive_int(payload, "max_search_results", 5)
    workspace = _optional_workspace(payload, "workspace")

    prompt = _build_search_web_prompt(query, max_search_results)
    result = _run_coco(
        prompt=prompt,
        phase="search",
        worktree=_SEARCH_WEB_WORKTREE,
        workspace=workspace,
    )

    try:
        return models.parse_search_result(result.stdout).as_dict()
    except models.DeepResearchParseError as exc:
        _raise_parse_error(exc)


@temporalio.activity.defn
def synthesize_report_with_coco(payload: dict[str, Any]) -> dict[str, Any]:
    """Generate a final report from successful and failed research results."""
    payload_data = _coerce_payload(payload)
    topic = _require_non_empty_str(payload_data, "topic")
    research_items = _require_list(payload_data, "research", allow_empty=True)
    failed_queries = _optional_failed_queries(payload_data)
    workspace = _optional_workspace(payload_data, "workspace")

    if not research_items:
        raise ApplicationError(
            "deep research synthesis requires at least one successful search result",
            type="deep_research_empty_research",
            non_retryable=True,
        )

    successful_queries = []
    for item in research_items:
        if not isinstance(item, Mapping):
            _raise_invalid_payload("research item must be an object")
        successful_queries.append(_require_non_empty_str(item, "query"))

    prompt_payload = {
        "topic": topic,
        "research": research_items,
        "failed_queries": failed_queries,
    }
    prompt = _build_synthesize_prompt(topic, prompt_payload)

    result = _run_coco(
        prompt=prompt,
        phase="synthesize",
        worktree=_SYNTHESIZE_WORKTREE,
        workspace=workspace,
    )

    markdown = result.stdout
    if not markdown.strip():
        raise ApplicationError(
            "synthesis produced empty report markdown",
            type="deep_research_empty_report",
            non_retryable=False,
        )

    return models.DeepResearchReport(
        topic=topic,
        markdown=markdown,
        successful_queries=successful_queries,
        failed_queries=failed_queries,
    ).as_dict()


def _coerce_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Ensure payload is a mutable string-keyed mapping."""
    if not isinstance(payload, Mapping):
        _raise_invalid_payload("request payload must be an object")
    return dict(payload)


def _require_non_empty_str(payload: Mapping[str, Any], key: str) -> str:
    """Read a required non-empty string field or fail with invalid payload."""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        _raise_invalid_payload(f"{key} must be a non-empty string")
    return value.strip()


def _require_non_empty_list(payload: Mapping[str, Any], key: str) -> list[Any]:
    """Read a required non-empty list field or fail with invalid payload."""
    return _require_list(payload, key, allow_empty=False)


def _require_list(
    payload: Mapping[str, Any], key: str, *, allow_empty: bool
) -> list[Any]:
    """Read a list field, optionally allowing empty values."""
    value = payload.get(key)
    if not isinstance(value, list):
        _raise_invalid_payload(f"{key} must be a list")
    if not allow_empty and not value:
        _raise_invalid_payload(f"{key} must be a non-empty list")
    return value


def _require_non_empty_string_list(payload: Mapping[str, Any], key: str) -> list[str]:
    """Read a required list of non-empty strings or fail with invalid payload."""
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        _raise_invalid_payload(f"{key} must be a non-empty list of strings")

    string_values: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            _raise_invalid_payload(f"{key} must contain only non-empty strings")
        string_values.append(item.strip())
    return string_values


def _require_positive_int(payload: Mapping[str, Any], key: str, default: int) -> int:
    """Read and validate a positive integer field, or use a default."""
    value = payload.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        _raise_invalid_payload(f"{key} must be a positive integer")
    return value


def _optional_workspace(payload: Mapping[str, Any], key: str) -> Path | None:
    """Return optional workspace path from payload."""
    workspace = payload.get(key)
    if workspace is None:
        return None
    if isinstance(workspace, Path):
        return workspace
    if isinstance(workspace, str):
        if not workspace.strip():
            _raise_invalid_payload("workspace must be a non-empty string when provided")
        return Path(workspace)
    _raise_invalid_payload("workspace must be a path string or Path")


def _optional_failed_queries(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    """Read optional failed query list."""
    value = payload.get("failed_queries", [])
    if value is None:
        return []
    if not isinstance(value, list):
        _raise_invalid_payload("failed_queries must be a list")
    failed: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            _raise_invalid_payload("failed_queries entries must be objects")
        failed.append(dict(item))  # type: ignore[arg-type]
    return failed


def _run_coco(
    *,
    prompt: str,
    phase: str,
    worktree: str,
    workspace: Path | None,
) -> Any:
    """Execute a COCO phase and normalize exception mapping."""
    try:
        return research.coco_cli.run_coco(
            prompt=prompt,
            phase=phase,
            worktree=worktree,
            workspace=workspace,
        )
    except research.coco_cli.CocoExecutionError as exc:
        details = {
            "phase": exc.phase,
            "returncode": exc.returncode,
            "stderr": exc.stderr,
            "stdout": exc.stdout,
            "message": str(exc),
        }
        raise ApplicationError(
            f"COCO execution failed during phase {phase}: {details}",
            type="deep_research_coco_failed",
            non_retryable=False,
        ) from exc


def _raise_parse_error(exc: models.DeepResearchParseError) -> None:
    """Map parser failures to retryable workflow application errors."""
    raise ApplicationError(
        str(exc),
        type="deep_research_parse_failed",
    ) from exc


def _raise_invalid_payload(message: str) -> None:
    """Map payload validation failures to non-retryable errors."""
    raise ApplicationError(
        message,
        type="deep_research_invalid_payload",
        non_retryable=True,
    )


def _build_subtopic_prompt(topic: str, max_subtopics: int) -> str:
    """Build prompt requesting compact subtopic JSON from COCO."""
    return (
        f"Generate up to {max_subtopics} high-quality, distinct subtopics for the\n"
        f"topic: {json.dumps(topic)}\n"
        "Return *only* compact JSON with keys:\n"
        '{"subtopics": [...], "rationale": "..."}\n'
    )


def _build_search_queries_prompt(
    subtopics: list[str], queries_per_subtopic: int
) -> str:
    """Build prompt requesting compact query JSON from COCO."""
    return (
        f"Generate up to {queries_per_subtopic} search queries for each of these subtopics: \n"
        f"{json.dumps(subtopics)}\n"
        "Return *only* compact JSON with key:\n"
        '{"queries": [..]}\n'
    )


def _build_search_web_prompt(query: str, max_search_results: int) -> str:
    """Build prompt requesting a single web-search result JSON payload."""
    return (
        f"Search web for the query and summarize findings: {json.dumps(query)}\n"
        f"Return at most {max_search_results} sources and output compact JSON with:\n"
        '{"query": "...", "summary": "...", "sources": ["..."]}\n'
    )


def _build_synthesize_prompt(topic: str, synth_payload: dict[str, Any]) -> str:
    """Build prompt requesting a final markdown research report."""
    return (
        "Write a comprehensive markdown report for the topic and evidence below.\n"
        f"Topic: {json.dumps(topic)}\n"
        f"Research payload: {json.dumps(synth_payload)}\n"
        "Return only markdown text (no surrounding explanation)."
    )
