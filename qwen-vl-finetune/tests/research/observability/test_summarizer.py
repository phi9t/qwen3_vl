import json
from pathlib import Path

from research.observability.summarizer import summarize_campaign


def test_summarizer_uses_analysis_json_as_authoritative(tmp_path: Path) -> None:
    attempt = (
        tmp_path / "experiments" / "runs" / "b200" / "probe" / "trial" / "attempt_001"
    )
    attempt.mkdir(parents=True)
    (attempt / "analysis.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trial_id": "b200/probe/trial",
                "attempt": 1,
                "status": "ok",
                "failure_reason": "none",
                "root_cause": "none",
                "expected_failure": False,
                "metrics": {
                    "status": "ok",
                    "val_loss": 0.5,
                    "peak_vram_mb": 100.0,
                    "throughput_steps_per_sec": 2.0,
                    "failure_reason": "none",
                },
                "symptoms": [],
                "evidence_refs": [],
                "actions": [],
                "recommendations": [],
                "artifact_dir": str(attempt),
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_campaign(tmp_path, profile="b200")

    assert "Fastest Safe Configuration" in summary
    assert "b200/probe/trial" in summary
    assert "0.5" in summary
