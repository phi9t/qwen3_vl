from research.models import FailureReason, TrialMetrics
from research.observability.redaction import redact_mapping
from research.observability.schema import (
    SCHEMA_VERSION,
    ResolvedTrialConfig,
    TrialAnalysis,
    TrialIntent,
)


def test_trial_intent_payload_has_schema_version_and_attempt() -> None:
    intent = TrialIntent.from_spec_payload(
        profile="b200",
        phase="probe",
        trial="probe_bs2",
        env={"BATCH_SIZE": "2"},
        attempt=1,
        campaign_id="campaign-1",
        workflow_id="wf-1",
        research_intent="capacity probe",
        expected_failure_reason="oom",
    )

    payload = intent.to_payload()

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["trial_id"] == "b200/probe/probe_bs2"
    assert payload["attempt"] == 1
    assert payload["planned_env"] == {"BATCH_SIZE": "2"}
    assert payload["expected_failure_reason"] == "oom"


def test_resolved_config_redacts_secrets() -> None:
    config = ResolvedTrialConfig(
        trial_id="b200/probe/t1",
        attempt=2,
        git_commit="abc1234",
        command=["bash", "scripts/sft_qwen3_8b.sh"],
        env={
            "BATCH_SIZE": "2",
            "HF_TOKEN": "secret",
            "AWS_SECRET_ACCESS_KEY": "secret2",
            "NORMAL": "visible",
        },
        model_path="/models/qwen",
        datasets=["train"],
        eval_datasets=["eval"],
        output_dir="outputs/b200/probe/t1/attempt_002",
        run_dir="experiments/runs/b200/probe/t1/attempt_002",
        hardware_profile="b200",
        distributed={"nproc_per_node": 8},
        artifact_root_ref="~/.automata/research/experiments",
        artifact_refs={
            "full_log": "logs/b200/probe/t1/attempt_002/run.log",
            "train_events": "events/b200/probe/t1/attempt_002/train_events.jsonl",
            "output_dir": "outputs/b200/probe/t1/attempt_002",
        },
    )

    payload = config.to_payload()

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["env"]["HF_TOKEN"] == "***REDACTED***"
    assert payload["env"]["AWS_SECRET_ACCESS_KEY"] == "***REDACTED***"
    assert payload["env"]["NORMAL"] == "visible"


def test_redact_mapping_matches_secret_key_patterns_case_insensitive() -> None:
    redacted = redact_mapping(
        {
            "api_key": "a",
            "PASSWORD": "b",
            "credential_path": "c",
            "TOKENIZERS_PARALLELISM": "false",
            "BATCH_SIZE": "2",
        }
    )

    assert redacted["api_key"] == "***REDACTED***"
    assert redacted["PASSWORD"] == "***REDACTED***"
    assert redacted["credential_path"] == "***REDACTED***"
    assert redacted["TOKENIZERS_PARALLELISM"] == "***REDACTED***"
    assert redacted["BATCH_SIZE"] == "2"


def test_trial_analysis_wraps_trial_metrics() -> None:
    metrics = TrialMetrics(
        status="ok",
        val_loss=0.5,
        peak_vram_mb=100.0,
        throughput_steps_per_sec=1.25,
        failure_reason=FailureReason.NONE,
    )
    analysis = TrialAnalysis(
        trial_id="b200/probe/t1",
        attempt=1,
        status="ok",
        failure_reason=FailureReason.NONE,
        root_cause="none",
        expected_failure=False,
        metrics=metrics,
        symptoms=[],
        evidence_refs=[],
        actions=[],
        recommendations=[],
        artifact_root_ref="~/.automata/research/experiments",
        artifact_refs={
            "full_log": "logs/b200/probe/t1/attempt_001/run.log",
            "train_events": "events/b200/probe/t1/attempt_001/train_events.jsonl",
            "output_dir": "outputs/b200/probe/t1/attempt_001",
        },
        artifact_dir="experiments/runs/b200/probe/t1/attempt_001",
    )

    payload = analysis.to_payload()

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["metrics"]["val_loss"] == 0.5
    assert payload["failure_reason"] == "none"
