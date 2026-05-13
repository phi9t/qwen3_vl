from __future__ import annotations

import re
from typing import Any

from research.models import FailureReason, TrialMetrics


_VAL_RE = re.compile(r"^val_loss:\s*([0-9.eE+-]+)\s*$", re.MULTILINE)
_VRAM_RE = re.compile(r"^peak_vram_mb:\s*([0-9.eE+-]+)\s*$", re.MULTILINE)
_THROUGHPUT_RE = re.compile(
    r"""["']train_steps_per_second["']\s*:\s*["']?([0-9.eE+-]+)["']?"""
)


def _last_float(regex: re.Pattern[str], text: str) -> float | None:
    matches = regex.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def _symptom(kind: str, message: str) -> dict[str, str]:
    return {"kind": kind, "message": message}


def _classify_failure(
    log_text: str,
    return_code: int,
) -> tuple[FailureReason, list[dict[str, Any]]]:
    lower = log_text.lower()
    symptoms: list[dict[str, Any]] = []

    if "out of memory" in lower or "cuda oom" in lower:
        symptoms.append(_symptom("oom", "CUDA out of memory detected"))
        return FailureReason.OOM, symptoms
    if "nccl" in lower and ("error" in lower or "timeout" in lower or "watchdog" in lower):
        symptoms.append(_symptom("nccl", "NCCL distributed transport failure detected"))
        return FailureReason.NCCL, symptoms
    if "cancelled" in lower or "canceled" in lower or "keyboardinterrupt" in lower or return_code == -15:
        symptoms.append(_symptom("cancelled", "Trial cancellation signature detected"))
        return FailureReason.CANCELLED, symptoms
    if "timed out" in lower or "timeout" in lower or "stalled" in lower or return_code == 124:
        symptoms.append(_symptom("timeout", "Trial timeout or stalled progress detected"))
        return FailureReason.TIMEOUT, symptoms
    if "modulenotfounderror" in lower or "importerror" in lower or "no module named" in lower:
        symptoms.append(_symptom("dependency", "Python dependency import failure detected"))
        return FailureReason.LAUNCHER_ERROR, symptoms
    if "keyerror" in lower and "dataset" in lower:
        symptoms.append(_symptom("dataset", "Dataset registry/config lookup failed"))
        return FailureReason.DATASET_ERROR, symptoms
    if "filenotfounderror" in lower and "dataset" in lower:
        symptoms.append(_symptom("dataset", "Dataset path was unavailable"))
        return FailureReason.DATASET_ERROR, symptoms
    if "model_load" in lower or "failed to load model" in lower or "could not load model" in lower:
        symptoms.append(_symptom("model_load", "Model load failure detected"))
        return FailureReason.MODEL_LOAD_ERROR, symptoms
    if (
        "snapshot_download" in lower
        or "hfhubhttperror" in lower
        or "huggingface" in lower
        or "hf_hub" in lower
        or ("cache" in lower and "model" in lower)
    ):
        symptoms.append(_symptom("hub_or_cache", "Hub/cache model snapshot failure detected"))
        return FailureReason.HUB_OR_CACHE_ERROR, symptoms
    if return_code != 0:
        symptoms.append(_symptom("launcher", "Launcher exited unsuccessfully"))
        return FailureReason.LAUNCHER_ERROR, symptoms
    return FailureReason.UNKNOWN, symptoms


def parse_metrics_and_symptoms(
    log_text: str,
    return_code: int,
) -> tuple[TrialMetrics, list[dict[str, Any]]]:
    val_loss = _last_float(_VAL_RE, log_text)
    peak_vram_mb = _last_float(_VRAM_RE, log_text)
    throughput = _last_float(_THROUGHPUT_RE, log_text)

    if return_code == 0 and val_loss is not None and peak_vram_mb is not None:
        return (
            TrialMetrics(
                status="ok",
                val_loss=val_loss,
                peak_vram_mb=peak_vram_mb,
                throughput_steps_per_sec=throughput,
                failure_reason=FailureReason.NONE,
            ),
            [],
        )

    if return_code == 0:
        return (
            TrialMetrics(
                status="crash",
                val_loss=val_loss,
                peak_vram_mb=peak_vram_mb,
                throughput_steps_per_sec=throughput,
                failure_reason=FailureReason.MISSING_FOOTER,
            ),
            [_symptom("missing_footer", "Training completed without metrics footer")],
        )

    failure_reason, symptoms = _classify_failure(log_text, return_code)
    return (
        TrialMetrics(
            status="crash",
            val_loss=val_loss,
            peak_vram_mb=peak_vram_mb,
            throughput_steps_per_sec=throughput,
            failure_reason=failure_reason,
        ),
        symptoms,
    )
