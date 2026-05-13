from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from research.models import FailureReason, TrialSpec
from research.observability.analyzer import analyze_trial
from research.observability.artifact_store import ArtifactRoot
from research.observability.artifacts import TrialArtifactStore
from research.observability.schema import ResolvedTrialConfig, TrialAnalysis, TrialIntent
from research.observability.system_snapshot import collect_system_snapshot


class SupervisorTrialRunner:
    def __init__(
        self,
        root: Path,
        *,
        stall_timeout_sec: float = 1800.0,
        max_trial_wall_time_sec: float = 21600.0,
    ) -> None:
        self.root = Path(root)
        self.artifact_root = ArtifactRoot.from_env(self.root)
        self.stall_timeout_sec = stall_timeout_sec
        self.max_trial_wall_time_sec = max_trial_wall_time_sec

    @staticmethod
    def terminate_process_group(
        proc: subprocess.Popen,
        *,
        grace_seconds: float = 30.0,
    ) -> None:
        if proc.poll() is not None:
            return
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.05)
        if proc.poll() is None:
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=10)

    @staticmethod
    def poll_gpu_memory_released(timeout_sec: float = 60.0) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                return True
            used = [int(line.strip()) for line in output.splitlines() if line.strip()]
            if not used or all(value == 0 for value in used):
                return True
            time.sleep(1.0)
        return False

    def _artifact_refs(self, spec: TrialSpec, attempt: int) -> dict[str, str]:
        attempt_name = f"attempt_{attempt:03d}"
        return {
            "full_log": self.artifact_root.ref_for(
                "logs",
                spec.profile,
                spec.phase,
                spec.trial,
                attempt_name,
                "run.log",
            ),
            "train_events": self.artifact_root.ref_for(
                "events",
                spec.profile,
                spec.phase,
                spec.trial,
                attempt_name,
                "train_events.jsonl",
            ),
            "output_dir": self.artifact_root.ref_for(
                "outputs",
                spec.profile,
                spec.phase,
                spec.trial,
                attempt_name,
            ),
        }

    def _build_env(
        self,
        spec: TrialSpec,
        output_dir: Path,
        events_path: Path,
    ) -> dict[str, str]:
        env = dict(os.environ)
        env.update(spec.env)
        env["RUN_NAME"] = spec.trial
        env["OUTPUT_DIR"] = str(output_dir)
        env["RESEARCH_EVENTS_PATH"] = str(events_path)
        env["RESEARCH_TRIAL_ID"] = f"{spec.profile}/{spec.phase}/{spec.trial}"
        return env

    def _resolved_config(
        self,
        spec: TrialSpec,
        attempt: int,
        run_dir: Path,
        env: dict[str, str],
        artifact_refs: dict[str, str],
    ) -> ResolvedTrialConfig:
        trial_env = {key: env[key] for key in sorted(spec.env) if key in env}
        trial_env.update(
            {
                "RUN_NAME": spec.trial,
                "OUTPUT_DIR": artifact_refs["output_dir"],
                "RESEARCH_TRIAL_ID": f"{spec.profile}/{spec.phase}/{spec.trial}",
            }
        )
        return ResolvedTrialConfig(
            trial_id=f"{spec.profile}/{spec.phase}/{spec.trial}",
            attempt=attempt,
            git_commit="unknown",
            command=["bash", "scripts/sft_qwen3_8b.sh"],
            env=trial_env,
            model_path=env.get("MODEL_NAME_OR_PATH", "Qwen/Qwen3-VL-8B-Instruct"),
            datasets=[env.get("DATASETS", "visualwebinstruct_train")],
            eval_datasets=[env.get("EVAL_DATASETS", "visualwebinstruct_val")],
            output_dir=artifact_refs["output_dir"],
            run_dir=str(run_dir.relative_to(self.root)),
            hardware_profile=spec.profile,
            distributed={"nproc_per_node": env.get("NPROC_PER_NODE", "")},
            artifact_root_ref=self.artifact_root.root_ref,
            artifact_refs=artifact_refs,
        )

    def run_from_spec(self, spec: TrialSpec, *, dry_run: bool = False) -> TrialAnalysis:
        store = TrialArtifactStore.create_next(spec.run_dir(self.root))
        artifact_refs = self._artifact_refs(spec, store.attempt)
        output_dir = self.artifact_root.resolve(artifact_refs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        events_path = self.artifact_root.resolve(artifact_refs["train_events"])
        events_path.parent.mkdir(parents=True, exist_ok=True)
        env = self._build_env(spec, output_dir, events_path)
        intent = TrialIntent.from_spec_payload(
            profile=spec.profile,
            phase=spec.phase,
            trial=spec.trial,
            env=spec.env,
            attempt=store.attempt,
            research_intent=f"{spec.phase} trial",
            expected_failure_reason=FailureReason.OOM.value
            if spec.phase == "probe"
            else None,
        )
        config = self._resolved_config(
            spec,
            store.attempt,
            store.attempt_dir,
            env,
            artifact_refs,
        )
        return self.run(intent, config, env=env, store=store, dry_run=dry_run)

    def run(
        self,
        intent: TrialIntent,
        config: ResolvedTrialConfig,
        *,
        env: dict[str, str],
        store: TrialArtifactStore,
        dry_run: bool = False,
    ) -> TrialAnalysis:
        store.write_json("intent.json", intent.to_payload())
        store.write_json("resolved_config.json", config.to_payload())
        store.write_json(
            "system.pre.json",
            collect_system_snapshot(env=config.env, include_process_topology=False),
        )
        store.lifecycle("trial_started")

        log_path = self.artifact_root.resolve(config.artifact_refs["full_log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if dry_run:
            log_path.write_text(
                "DRY_RUN\nval_loss: 0.0\npeak_vram_mb: 0.0\n",
                encoding="utf-8",
            )
            return_code = 0
        else:
            return_code = self._run_process(
                config.command,
                env=env,
                log_path=log_path,
                store=store,
            )

        store.lifecycle("process_exited", return_code=return_code)
        store.write_json(
            "system.post.json",
            collect_system_snapshot(env=config.env, include_process_topology=True),
        )
        store.lifecycle("analysis_started")
        analysis = analyze_trial(
            trial_id=intent.trial_id,
            attempt=intent.attempt,
            artifact_dir=store.attempt_dir,
            artifact_dir_ref=str(store.attempt_dir.relative_to(self.root)),
            log_path=log_path,
            return_code=return_code,
            expected_failure_reason=intent.expected_failure_reason,
            artifact_root_ref=config.artifact_root_ref,
            artifact_refs=config.artifact_refs,
        )
        store.write_json("analysis.json", analysis.to_payload())
        store.path("analysis.md").write_text(
            self._analysis_markdown(analysis),
            encoding="utf-8",
        )
        store.lifecycle(
            "analysis_finished",
            status=analysis.status,
            failure_reason=analysis.failure_reason.value,
        )
        for action in analysis.actions:
            store.action(
                action.get("action", "record_analysis_action"),
                reason=action.get("reason", "analysis action"),
                outcome=action.get("outcome", "recorded"),
            )
        return analysis

    def _run_process(
        self,
        command: list[str],
        *,
        env: dict[str, str],
        log_path: Path,
        store: TrialArtifactStore,
    ) -> int:
        start = time.monotonic()
        last_growth = start
        last_size = 0
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                command,
                cwd=self.root,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            store.lifecycle("launcher_started", pid=proc.pid)
            while proc.poll() is None:
                now = time.monotonic()
                size = log_path.stat().st_size if log_path.exists() else 0
                if size > last_size:
                    last_growth = now
                    last_size = size
                if (
                    now - start > self.max_trial_wall_time_sec
                    or now - last_growth > self.stall_timeout_sec
                ):
                    store.lifecycle("timeout", pid=proc.pid)
                    self.terminate_process_group(proc, grace_seconds=30.0)
                    self.poll_gpu_memory_released()
                    return 124
                time.sleep(1.0)
            return proc.returncode or 0

    def _analysis_markdown(self, analysis: TrialAnalysis) -> str:
        return (
            "# Trial Analysis\n\n"
            f"- trial: `{analysis.trial_id}`\n"
            f"- attempt: `{analysis.attempt}`\n"
            f"- status: `{analysis.status}`\n"
            f"- failure_reason: `{analysis.failure_reason.value}`\n"
            f"- root_cause: `{analysis.root_cause}`\n"
        )
