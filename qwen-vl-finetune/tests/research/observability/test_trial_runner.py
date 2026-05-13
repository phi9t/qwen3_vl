import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from research.models import TrialSpec
from research.observability.trial_runner import SupervisorTrialRunner


def test_dry_run_writes_attempt_artifacts(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path
    artifact_root = tmp_path / "external-artifacts"
    monkeypatch.setenv("RESEARCH_ARTIFACT_ROOT", str(artifact_root))
    launcher = root / "scripts" / "sft_qwen3_8b.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    spec = TrialSpec("b200", "probe", "trial", {"BATCH_SIZE": "2"})

    analysis = SupervisorTrialRunner(root).run_from_spec(spec, dry_run=True)

    attempt_dir = (
        root / "experiments" / "runs" / "b200" / "probe" / "trial" / "attempt_001"
    )
    assert analysis.status == "ok"
    assert (attempt_dir / "intent.json").exists()
    assert (attempt_dir / "resolved_config.json").exists()
    assert (attempt_dir / "system.pre.json").exists()
    assert (attempt_dir / "analysis.json").exists()
    assert not (attempt_dir / "run.log").exists()
    assert not (attempt_dir / "output").exists()
    assert (
        artifact_root / "logs" / "b200" / "probe" / "trial" / "attempt_001" / "run.log"
    ).exists()
    assert (
        artifact_root / "outputs" / "b200" / "probe" / "trial" / "attempt_001"
    ).is_dir()


def test_process_group_teardown_kills_child_processes(tmp_path: Path) -> None:
    script = tmp_path / "spawn_child.py"
    marker = tmp_path / "child.pid"
    script.write_text(
        "import pathlib, subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(p.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    proc = subprocess.Popen([sys.executable, str(script)], start_new_session=True)
    try:
        for _ in range(100):
            if marker.exists():
                break
            time.sleep(0.05)
        child_pid = int(marker.read_text(encoding="utf-8"))
        SupervisorTrialRunner.terminate_process_group(proc, grace_seconds=0.1)

        assert proc.poll() is not None
        for _ in range(20):
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            child_alive = False
        else:
            child_alive = True
        assert child_alive is False
    finally:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def test_gpu_memory_release_polling_uses_nvidia_smi(monkeypatch) -> None:
    calls = iter(["100\n25\n", "0\n0\n"])

    def fake_check_output(cmd, text=True, stderr=None):
        assert cmd[:2] == ["nvidia-smi", "--query-gpu=memory.used"]
        return next(calls)

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    assert SupervisorTrialRunner.poll_gpu_memory_released(timeout_sec=2.0) is True
