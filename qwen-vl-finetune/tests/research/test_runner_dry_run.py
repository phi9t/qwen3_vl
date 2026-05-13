import os
import subprocess
import sys
from pathlib import Path

from research.models import FailureReason, TrialMetrics, TrialSpec
from research.observability.trial_runner import SupervisorTrialRunner
from research.runner import (
    append_result_row,
    append_result_row_from_analysis,
    build_trial_env,
    results_header,
)


def test_build_trial_env_sets_output_dir(tmp_path: Path) -> None:
    root = tmp_path / "qwen-vl-finetune"
    spec = TrialSpec("b200", "probe", "probe1", {"BATCH_SIZE": "2"})

    env = build_trial_env(root, spec)

    assert env["BATCH_SIZE"] == "2"
    assert env["OUTPUT_DIR"].endswith("experiments/runs/b200/probe/probe1/output")
    assert env["RUN_NAME"] == "probe1"


def test_append_result_row_writes_profile_and_failure_reason(tmp_path: Path) -> None:
    path = tmp_path / "results.tsv"
    spec = TrialSpec("b200", "probe", "probe1", {"BATCH_SIZE": "2"})
    metrics = TrialMetrics(
        status="crash",
        failure_reason=FailureReason.MISSING_FOOTER,
    )

    append_result_row(path, spec, metrics, git_commit="abc123", log_path=Path("run.log"), output_dir=Path("out"))

    text = path.read_text(encoding="utf-8")
    assert results_header().split("\t")[2] == "profile"
    assert "b200\tprobe\tprobe1" in text
    assert "missing_footer" in text


def test_append_result_row_from_analysis_preserves_profiled_tsv(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path
    artifact_root = tmp_path / "external-artifacts"
    monkeypatch.setenv("RESEARCH_ARTIFACT_ROOT", str(artifact_root))
    (root / "scripts").mkdir()
    (root / "scripts" / "sft_qwen3_8b.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n",
        encoding="utf-8",
    )
    spec = TrialSpec("b200", "probe", "trial", {"BATCH_SIZE": "2"})
    analysis = SupervisorTrialRunner(root).run_from_spec(spec, dry_run=True)
    result_path = root / "experiments" / "results.profiled.tsv"

    append_result_row_from_analysis(result_path, spec, analysis, git_commit="abc123")

    text = result_path.read_text(encoding="utf-8")
    assert "trial" in text
    assert "\tok\t" in text
    assert "outputs/b200/probe/trial/attempt_001" in text


def test_runner_exposes_subagent_command() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    proc = subprocess.run(
        [sys.executable, "-m", "research.runner", "subagents", "--provider", "all"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0
