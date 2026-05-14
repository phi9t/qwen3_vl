"""Generic preflight checks and adapter preflight dispatch."""

from __future__ import annotations

import shutil

import research.models


def run_preflight(
    adapter: research.models.ExperimentAdapter,
    intent: research.models.Intent,
    context: research.models.TrialContext,
) -> research.models.PreflightResult:
    """Run generic and adapter-specific preflight checks for a trial.

    Args:
        adapter: The experiment adapter providing adapter-specific checks.
        intent: The intent configuration to validate.
        context: The trial runtime context with paths and identifiers.

    Returns:
        A PreflightResult with combined checks, ok flag, and message.
    """
    checks: dict[str, object] = {
        "adapter": adapter.name,
        "uv": "ok" if shutil.which("uv") else "missing",
        "db": "ok" if context.db_path.exists() else "missing",
    }
    context.worktree.mkdir(parents=True, exist_ok=True)
    context.artifact_dir.mkdir(parents=True, exist_ok=True)
    checks["worktree"] = "ok" if context.worktree.exists() else "missing"
    checks["artifact_dir"] = "ok" if context.artifact_dir.exists() else "missing"

    adapter_result = adapter.preflight(intent, context)
    checks.update(adapter_result.checks)
    ok = (
        checks["uv"] == "ok"
        and checks["db"] == "ok"
        and checks["worktree"] == "ok"
        and checks["artifact_dir"] == "ok"
        and adapter_result.ok
    )
    message = adapter_result.message if ok else "preflight failed"
    return research.models.PreflightResult(ok=ok, checks=checks, message=message)
