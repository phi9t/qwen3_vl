from pathlib import Path

from research.models import FailureReason
from research.observability.analyzer import analyze_trial


def test_analyze_expected_oom_capacity_probe(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "RuntimeError: CUDA out of memory. Tried to allocate 1.00 GiB\n",
        encoding="utf-8",
    )

    analysis = analyze_trial(
        trial_id="b200/probe/t1",
        attempt=1,
        artifact_dir=tmp_path,
        log_path=log,
        return_code=1,
        expected_failure_reason="oom",
    )

    assert analysis.failure_reason == FailureReason.OOM
    assert analysis.root_cause == "capacity_exceeded"
    assert analysis.expected_failure is True
    assert analysis.actions[0]["action"] == "mark_capacity_boundary"


def test_analyze_missing_footer_with_zero_return_code(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("{'train_steps_per_second': 0.2}\n", encoding="utf-8")

    analysis = analyze_trial(
        trial_id="b200/probe/t2",
        attempt=1,
        artifact_dir=tmp_path,
        log_path=log,
        return_code=0,
        expected_failure_reason=None,
    )

    assert analysis.failure_reason == FailureReason.MISSING_FOOTER
    assert analysis.root_cause == "training_incomplete"
    assert analysis.expected_failure is False


def test_analyze_dataset_error_maps_to_dataset_unavailable(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("KeyError: 'unknown_dataset'\n", encoding="utf-8")

    analysis = analyze_trial(
        trial_id="b200/sweep/t3",
        attempt=1,
        artifact_dir=tmp_path,
        log_path=log,
        return_code=1,
        expected_failure_reason=None,
    )

    assert analysis.failure_reason == FailureReason.DATASET_ERROR
    assert analysis.root_cause == "dataset_unavailable"


def test_analyze_nccl_refines_distributed_transport(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("NCCL watchdog timeout while waiting for rank 3\n", encoding="utf-8")

    analysis = analyze_trial(
        trial_id="b200/sweep/t4",
        attempt=1,
        artifact_dir=tmp_path,
        log_path=log,
        return_code=1,
        expected_failure_reason=None,
    )

    assert analysis.failure_reason == FailureReason.NCCL
    assert analysis.root_cause == "distributed_transport"


def test_analyze_model_snapshot_refinement(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("HfHubHTTPError: snapshot_download failed for model\n", encoding="utf-8")

    analysis = analyze_trial(
        trial_id="b200/sweep/t5",
        attempt=1,
        artifact_dir=tmp_path,
        log_path=log,
        return_code=1,
        expected_failure_reason=None,
    )

    assert analysis.failure_reason == FailureReason.HUB_OR_CACHE_ERROR
    assert analysis.root_cause == "model_snapshot_unavailable"


def test_analyze_timeout_refinement(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("Trial exceeded timeout after no progress\n", encoding="utf-8")

    analysis = analyze_trial(
        trial_id="b200/sweep/t6",
        attempt=1,
        artifact_dir=tmp_path,
        log_path=log,
        return_code=124,
        expected_failure_reason=None,
    )

    assert analysis.failure_reason == FailureReason.TIMEOUT
    assert analysis.root_cause == "stalled_or_hung"


def test_analyze_dataset_unavailable_category(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("FileNotFoundError: dataset manifest was not found\n", encoding="utf-8")

    analysis = analyze_trial(
        trial_id="b200/sweep/t7",
        attempt=1,
        artifact_dir=tmp_path,
        log_path=log,
        return_code=1,
        expected_failure_reason=None,
    )

    assert analysis.failure_reason == FailureReason.DATASET_ERROR
    assert analysis.root_cause == "dataset_unavailable"
