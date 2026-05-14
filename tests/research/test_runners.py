"""Tests for generic trial command runners."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap

import research.adapters
import research.models
import research.runners


def test_run_trial_command_writes_log_and_progress(
    tmp_path: pathlib.Path,
) -> None:
    adapter = research.adapters.load_adapter("tests.research.fake_adapter:FakeAdapter")
    context = research.models.TrialContext(
        experiment_id=1,
        trial_run_id=1,
        attempt=1,
        worktree=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        db_path=tmp_path / "research.sqlite",
    )
    context.artifact_dir.mkdir()
    intent = research.models.Intent(
        "fake",
        "unit-model",
        "cpu",
        "probe",
        "small",
        {"batch_size": 1},
    )
    command = adapter.build_trial(intent, context)

    result = research.runners.run_trial_command(adapter, command, context)

    assert result.returncode == 0
    assert result.progress[-1].metrics == {"metric": 1.0}
    assert (
        (context.artifact_dir / "run.log").read_text(encoding="utf-8").strip()
        == "metric=1.0"
    )


def test_run_trial_command_does_not_hang_on_inherited_stdout(
    tmp_path: pathlib.Path,
) -> None:
    script = tmp_path / "exercise_runner.py"
    artifact_dir = tmp_path / "artifacts"
    script.write_text(
        textwrap.dedent(
            f"""
            from __future__ import annotations

            import pathlib
            import sys

            import research.models
            import research.runners


            class Adapter:
                name = "inheritance"

                def parse_progress(self, line: str):
                    if "metric=" in line:
                        return research.models.ProgressUpdate(
                            metrics={{"metric": 1.0}},
                            message=line.strip(),
                        )
                    return None


            context = research.models.TrialContext(
                experiment_id=1,
                trial_run_id=1,
                attempt=1,
                worktree=pathlib.Path({str(tmp_path)!r}),
                artifact_dir=pathlib.Path({str(artifact_dir)!r}),
                db_path=pathlib.Path({str(tmp_path / "research.sqlite")!r}),
            )
            command = research.models.TrialCommand(
                argv=[
                    sys.executable,
                    "-c",
                    (
                        "import subprocess, sys; "
                        "subprocess.Popen([sys.executable, '-c', "
                        "'import time; time.sleep(30)']); "
                        "print('metric=1.0', flush=True)"
                    ),
                ],
            )
            result = research.runners.run_trial_command(Adapter(), command, context)
            assert result.returncode == 0
            assert result.progress[-1].metrics == {{"metric": 1.0}}
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=pathlib.Path.cwd(),
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
