# COCO deep research workflow design

Date: 2026-05-26

## Goal

Adapt the Temporal deep-research workflow pattern so every model-driven phase
runs through the local COCO CLI instead of the OpenAI SDK or API. The workflow
should keep Temporal in charge of orchestration, timeouts, retries, partial
search failure handling, and worker visibility.

The user explicitly wants `search_web` delegated to COCO CLI as well. There is
no separate Python web-search activity in this design.

## Existing context

The repository already separates generic research mechanics from Qwen-specific
experiment code:

- `research/workflows.py` defines generic Temporal workflows.
- `research/activities.py` contains activity helpers and the generic trial
  runner.
- `research/temporal.py` owns worker registration and workflow start helpers.
- `qwen-vl-finetune/experiments/qwen_temporal.py` registers concrete Qwen
  activities on top of generic research helpers.

The COCO deep-research workflow should follow that split. Generic Temporal and
COCO plumbing belongs under `research/`. Qwen trial execution should not change.

## Recommended approach

Use one Temporal activity per deep-research phase:

1. `generate_subtopics_with_coco`
2. `generate_search_queries_with_coco`
3. `search_web_with_coco`
4. `synthesize_report_with_coco`

This matches the Temporal AI pipeline pattern while keeping subprocess work out
of workflow code. The workflow can run search activities in parallel and keep
going when a subset fails.

Rejected alternatives:

- A single generic `run_coco_task` activity would reduce code, but it would push
  prompt selection, parse rules, timeout choices, and error handling into
  conditionals.
- A single "COCO does everything" activity would hide the pipeline from
  Temporal. That would remove durable per-query search, partial failure
  handling, and useful phase-level status.

## Architecture

Add new generic modules under `research/`:

- `research/deep_research_models.py`
- `research/deep_research_activities.py`
- `research/deep_research_workflows.py`
- `research/coco_cli.py`

`research/deep_research_workflows.py` defines `DeepResearchWorkflow`. It
orchestrates only deterministic Temporal commands:

1. Execute `generate_subtopics_with_coco`.
2. Execute `generate_search_queries_with_coco`.
3. Execute `search_web_with_coco` once per query with `asyncio.gather(...,
   return_exceptions=True)`.
4. Execute `synthesize_report_with_coco` with successful search results.

`research/coco_cli.py` is the subprocess boundary. It discovers the `coco`
binary, validates any configured workspace path, builds the CLI command, runs
the subprocess, and returns stdout, stderr, and exit code to the activity.

The workflow must never call COCO, access the filesystem, read environment
variables, or parse wall-clock time directly. Those operations belong in
activities.

## Data contracts

Use small dataclasses in `research/deep_research_models.py`:

- `DeepResearchRequest`
  - `topic: str`
  - `max_subtopics: int = 5`
  - `queries_per_subtopic: int = 3`
  - `max_search_results: int = 5`
- `SubtopicPlan`
  - `subtopics: list[str]`
  - `rationale: str = ""`
- `SearchQueryPlan`
  - `queries: list[str]`
- `SearchResult`
  - `query: str`
  - `summary: str`
  - `sources: list[str]`
  - `raw_output_path: str = ""`
- `DeepResearchReport`
  - `topic: str`
  - `markdown: str`
  - `successful_queries: list[str]`
  - `failed_queries: list[dict[str, str]]`

Activities return plain dictionaries to Temporal. The dataclasses are for local
construction, validation, and test clarity.

## COCO prompts and output parsing

Planning and search activities should ask COCO for compact JSON:

- Subtopics: `{"subtopics": [...], "rationale": "..."}`
- Query generation: `{"queries": [...]}`
- Search: `{"query": "...", "summary": "...", "sources": [...]}`

Synthesis may return Markdown. The synthesis activity wraps that Markdown into a
`DeepResearchReport` dictionary.

Each activity validates the parsed shape before returning. JSON parse failures
raise a retryable activity failure. Empty required fields fail the activity
rather than returning ambiguous data.

The activity may write raw COCO output to artifacts and return
`raw_output_path`. Workflow state should hold summaries and paths, not large raw
outputs.

## Failure handling

COCO subprocess failures raise activity failures that include:

- phase name
- exit code
- stderr tail
- stdout tail when useful for debugging

`search_web_with_coco` uses both `start_to_close_timeout` and
`schedule_to_close_timeout`. This follows the referenced Temporal pattern where
one repeatedly failing search task should not block the whole report.

`DeepResearchWorkflow` collects search exceptions with
`asyncio.gather(..., return_exceptions=True)`. It passes successful results and
failed query metadata to synthesis.

If all search activities fail, the workflow fails before synthesis. Producing a
report with no research would be misleading.

## Worker and CLI integration

Add a worker registration path that includes `DeepResearchWorkflow` and the four
COCO activities. This can be either:

- a generic `research` CLI subcommand such as `research deep-research worker`,
  plus a start command, or
- helper functions in `research.temporal` that a later CLI can expose.

The design should not add COCO dependencies to Qwen trial registration. Qwen
manager behavior should remain unchanged.

COCO discovery should use `shutil.which("coco")`, with an optional environment
override if implementation needs one. Missing COCO is a runtime activity failure
or explicit CLI preflight failure. Tests should not skip imports or code paths
because COCO is absent.

## Testing

Unit tests:

- COCO command construction uses the expected command shape.
- COCO output parsing accepts valid JSON and rejects malformed or incomplete
  data.
- COCO subprocess failures include phase, exit code, and stderr context.

Workflow tests:

- Happy path runs all phases and returns a report.
- Partial search failure still runs synthesis with successful results.
- All search failure fails before synthesis.
- Worker registration includes the workflow and all four activities.

Tests should fake the COCO subprocess runner or register fake activities under
the real activity names. Normal tests should not execute the real COCO CLI.

## Acceptance criteria

- The deep-research workflow contains no OpenAI SDK/API dependency.
- Every model/search phase runs through a Temporal activity backed by COCO CLI.
- Search activities run in parallel and tolerate partial failure.
- All-search failure is explicit and does not produce a report.
- Qwen experiment workflows and activities keep their current behavior.
- Verification uses `uv run ... python -m pytest ...` and repo lint commands.
