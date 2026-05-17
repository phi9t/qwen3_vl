"""Generic preflight checks and adapter preflight dispatch."""

from __future__ import annotations

import os
import shutil

import research.models


def _uv_available() -> bool:
    configured = os.environ.get("UV")
    if configured:
        return shutil.which(configured) is not None or os.path.exists(configured)
    return shutil.which("uv") is not None


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
    generic_checks: dict[str, object] = {
        "adapter": adapter.name,
        "uv": "ok" if _uv_available() else "missing",
        "db": "ok" if context.db_path.exists() else "missing",
    }
    try:
        context.worktree.mkdir(parents=True, exist_ok=True)
        context.artifact_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        generic_checks["worktree"] = (
            "ok" if context.worktree.is_dir() else f"error: {exc}"
        )
        generic_checks["artifact_dir"] = (
            "ok" if context.artifact_dir.is_dir() else f"error: {exc}"
        )
        return research.models.PreflightResult(
            ok=False,
            checks=generic_checks,
            message="preflight failed",
        )
    generic_checks["worktree"] = "ok" if context.worktree.is_dir() else "missing"
    generic_checks["artifact_dir"] = (
        "ok" if context.artifact_dir.is_dir() else "missing"
    )

    adapter_result = adapter.preflight(intent, context)
    checks = {
        **adapter_result.checks,
        **generic_checks,
    }
    ok = (
        generic_checks["uv"] == "ok"
        and generic_checks["db"] == "ok"
        and generic_checks["worktree"] == "ok"
        and generic_checks["artifact_dir"] == "ok"
        and adapter_result.ok
    )
    message = adapter_result.message if ok else "preflight failed"
    return research.models.PreflightResult(ok=ok, checks=checks, message=message)
