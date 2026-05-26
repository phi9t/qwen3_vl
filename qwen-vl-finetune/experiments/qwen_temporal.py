"""Qwen-VL Temporal activities and worker registration helpers."""

from __future__ import annotations

import pathlib

import temporalio.activity
import temporalio.exceptions

import experiments.qwen_adapter
import research.activities


def qwen_worktree() -> pathlib.Path:
    """Return the stable Qwen-VL project root used by worker activities."""
    return pathlib.Path(__file__).resolve().parents[1]


@temporalio.activity.defn
def qwen_run_trial_activity(
    db_path_s: str,
    experiment_id: int,
    attempt: int = 1,
) -> dict[str, object]:
    """Run a Qwen-VL trial through the generic research trial helper."""
    result = research.activities.run_trial_with_adapter(
        db_path_s,
        experiment_id,
        experiments.qwen_adapter.QwenVlAdapter(),
        attempt,
        qwen_worktree(),
        research.activities.heartbeat_progress,
    )
    if result["status"] != "succeeded":
        raise temporalio.exceptions.ApplicationError(
            f"Qwen-VL trial {experiment_id} failed",
            result,
            type="qwen_trial_failed",
            non_retryable=True,
        )
    return result
