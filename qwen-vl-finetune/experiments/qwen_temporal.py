"""Qwen-VL Temporal activities and worker registration helpers."""

from __future__ import annotations

import pathlib

import temporalio.activity

import experiments.qwen_adapter
import research.activities


@temporalio.activity.defn
def qwen_run_trial_activity(
    db_path_s: str,
    experiment_id: int,
    attempt: int = 1,
) -> dict[str, object]:
    """Run a Qwen-VL trial through the generic research trial helper."""
    return research.activities.run_trial_with_adapter(
        db_path_s,
        experiment_id,
        experiments.qwen_adapter.QwenVlAdapter(),
        attempt,
        pathlib.Path.cwd(),
    )
