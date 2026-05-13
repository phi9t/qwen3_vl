from __future__ import annotations

import asyncio
import concurrent.futures
import importlib.util
import subprocess
from pathlib import Path

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from autonomy import sandbox
from autonomy.activities import (
    launch_run,
    org_append_blocker,
    org_pick_next_ready,
    org_transition,
    run_done_gate,
    worktree_create,
    worktree_destroy_if_clean,
)
from autonomy.workflows import SymphonyTrackerWorkflow


TEMPORAL_TEST_SERVER_AVAILABLE = importlib.util.find_spec("temporalio.testing") is not None


@pytest.mark.skipif(not TEMPORAL_TEST_SERVER_AVAILABLE, reason="temporalio test server unavailable")
def test_workflow_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "-C", str(repo), "init"], check=True)
        (repo / "README.md").write_text("# repo\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.email=test@example.com",
                "-c",
                "user.name=Test User",
                "commit",
                "-m",
                "init",
            ],
            check=True,
        )

        monkeypatch.chdir(repo)
        monkeypatch.setenv("AUTONOMY_REPO_ROOT", str(repo))

        tracker_path = repo / ".autonomy" / "runs" / "echo-smoke" / "tracker.org"
        tracker_path.parent.mkdir(parents=True, exist_ok=True)
        tracker_path.write_text(
            "\n".join(
                [
                    "#+TITLE: Echo Smoke",
                    "#+AUTHOR: test",
                    "#+STARTUP: overview logdone",
                    "#+TODO: TODO(t) READY(r) IN-PROGRESS(i!) BLOCKED(b@/!) AWAITING-GATE(g!) | DONE(d!) WONTFIX(w@/!) FAILED(f@/!)",
                    "#+PROPERTY: header-args :eval no",
                    "#+FILETAGS: :autonomy:echo-smoke:",
                    "#+AUTONOMY_VERSION: 1",
                    "#+AUTONOMY_RUN_SLUG: echo-smoke",
                    "#+AUTONOMY_DEFAULT_EXECUTOR: shell",
                    "#+AUTONOMY_DEFAULT_GATE: none",
                    "",
                    "* Tasks",
                    "** TODO T01 Echo hello",
                    "   :PROPERTIES:",
                    "   :ID:        T01",
                    "   :EXECUTOR:  shell",
                    "   :GATE:      none",
                    "   :TIMEOUT:   1m",
                    "   :GPUS:      0",
                    "   :END:",
                    "",
                    "   *Acceptance.*",
                    "   cmd: echo hello",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        def fake_draccus_exec(
            cmd: list[str],
            *,
            workspace: Path,
            gpus: int,
            log_path: Path,
            env_overrides: dict[str, str] | None = None,
            timeout: float | None = None,
            _draccus_run_path: Path | None = None,
        ) -> sandbox.ExecResult:
            result = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True, check=False)
            output = result.stdout
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(output, encoding="utf-8")
            return sandbox.ExecResult(exit_code=result.returncode, log_path=log_path, stdout_tail=output)

        monkeypatch.setattr(sandbox, "draccus_exec", fake_draccus_exec)

        async with await WorkflowEnvironment.start_local() as env:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as activity_executor:
                async with Worker(
                    env.client,
                    task_queue="autonomy",
                    workflows=[SymphonyTrackerWorkflow],
                    activities=[
                        org_pick_next_ready,
                        org_transition,
                        org_append_blocker,
                        worktree_create,
                        worktree_destroy_if_clean,
                        launch_run,
                        run_done_gate,
                    ],
                    activity_executor=activity_executor,
                ):
                    handle = await env.client.start_workflow(
                        SymphonyTrackerWorkflow.run,
                        args=[str(tracker_path)],
                        id="autonomy-tracker-echo-smoke",
                        task_queue="autonomy",
                    )
                    result = await handle.result()

        assert result["completed"] == 1
        assert result["phase"] == "drained"

        tracker_text = tracker_path.read_text(encoding="utf-8")
        assert "** DONE T01 Echo hello" in tracker_text
        assert ":STARTED:" in tracker_text
        assert ":FINISHED:" in tracker_text
        assert ":EXIT_CODE:" in tracker_text
        assert 'State "IN-PROGRESS" from "TODO"' in tracker_text
        assert 'State "DONE" from "IN-PROGRESS"' in tracker_text

        run_log = repo / ".autonomy" / "runs" / "echo-smoke" / "artifacts" / "T01" / "run.log"
        assert run_log.read_text(encoding="utf-8").strip() == "hello"

    asyncio.run(run())
