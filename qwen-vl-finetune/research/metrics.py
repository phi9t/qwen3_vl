from __future__ import annotations

import re

from research.models import FailureReason, TrialMetrics


_VAL_RE = re.compile(r"^val_loss:\s*([0-9.eE+-]+)\s*$", re.MULTILINE)
_VRAM_RE = re.compile(r"^peak_vram_mb:\s*([0-9.eE+-]+)\s*$", re.MULTILINE)
_THROUGHPUT_RE = re.compile(
    r"""["']train_steps_per_second["']\s*:\s*["']?([0-9.eE+-]+)["']?"""
)


def classify_failure(log_text: str, return_code: int) -> FailureReason:
    lower = log_text.lower()
    if "out of memory" in lower or "cuda oom" in lower:
        return FailureReason.OOM
    if "nccl" in lower and ("error" in lower or "timeout" in lower or "watchdog" in lower):
        return FailureReason.NCCL
    if return_code != 0:
        return FailureReason.LAUNCHER_ERROR
    return FailureReason.UNKNOWN


def _last_float(regex: re.Pattern[str], text: str) -> float | None:
    matches = regex.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def parse_trial_metrics(log_text: str, return_code: int) -> TrialMetrics:
    val_loss = _last_float(_VAL_RE, log_text)
    peak_vram_mb = _last_float(_VRAM_RE, log_text)
    throughput = _last_float(_THROUGHPUT_RE, log_text)

    if return_code == 0 and val_loss is not None and peak_vram_mb is not None:
        return TrialMetrics(
            status="ok",
            val_loss=val_loss,
            peak_vram_mb=peak_vram_mb,
            throughput_steps_per_sec=throughput,
            failure_reason=FailureReason.NONE,
        )

    reason = FailureReason.MISSING_FOOTER
    if return_code != 0:
        reason = classify_failure(log_text, return_code)
    return TrialMetrics(
        status="crash",
        val_loss=val_loss,
        peak_vram_mb=peak_vram_mb,
        throughput_steps_per_sec=throughput,
        failure_reason=reason,
    )
