from research.metrics import classify_failure, parse_trial_metrics
from research.models import FailureReason


def test_parse_trial_metrics_extracts_footer_and_throughput() -> None:
    text = """
{'train_runtime': '159.9', 'train_steps_per_second': '0.313'}
val_loss: 0.5600338578224182
peak_vram_mb: 95280.6162109375
"""

    metrics = parse_trial_metrics(text, return_code=0)

    assert metrics.status == "ok"
    assert metrics.val_loss == 0.5600338578224182
    assert metrics.peak_vram_mb == 95280.6162109375
    assert metrics.throughput_steps_per_sec == 0.313


def test_classify_failure_detects_oom() -> None:
    assert classify_failure("CUDA out of memory", return_code=1) == FailureReason.OOM


def test_missing_footer_is_crash() -> None:
    metrics = parse_trial_metrics("training ended without footer", return_code=0)

    assert metrics.status == "crash"
    assert metrics.failure_reason == FailureReason.MISSING_FOOTER
