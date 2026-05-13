from __future__ import annotations

import argparse
import concurrent.futures
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from temporalio.client import Client
from temporalio.worker import Worker

from research.activities import (
    load_profile_activity,
    plan_probe_trials_activity,
    plan_sweep_trials_activity,
    run_trial_activity,
    select_capacity_activity,
    select_capacity_from_results_activity,
    summarize_campaign_activity,
)
from research.experiment_helpers import (
    append_result_row,
    append_result_row_from_analysis,
    git_short,
    plan_sweep_trials,
    repo_root,
    results_header,
    results_path,
    workspace_root,
)
from research.local_temporal import (
    DEFAULT_ADDRESS,
    DEFAULT_TASK_QUEUE,
    ensure_local_temporal_server,
    terminate_local_temporal_server,
)
from research.metrics import parse_trial_metrics
from research.models import FailureReason, TrialMetrics, TrialSpec
from research.observability.summarizer import summarize_campaign
from research.profile_config import load_profile, plan_probe_trials
from research.selection import choose_capacity, write_selected_env
from research.subagents import (
    AgentProvider,
    SubagentTask,
    build_command,
    discover_providers,
    implementation_prompt,
)
from research.workflows import ProfiledExperimentWorkflow, SingleTrialWorkflow


__all__ = [
    "append_result_row",
    "append_result_row_from_analysis",
    "build_trial_env",
    "results_header",
]


def build_trial_env(root: Path, spec: TrialSpec) -> dict[str, str]:
    env = dict(os.environ)
    env.update(spec.env)
    env["RUN_NAME"] = spec.trial
    env["OUTPUT_DIR"] = str(spec.output_dir(root))
    return env


def run_trial(root: Path, spec: TrialSpec, *, dry_run: bool) -> TrialMetrics:
    run_dir = spec.run_dir(root)
    log_path = spec.log_path(root)
    run_dir.mkdir(parents=True, exist_ok=True)

    launcher = root / "scripts" / "sft_qwen3_8b.sh"
    if dry_run:
        log_path.write_text(f"DRY_RUN launcher={launcher}\n", encoding="utf-8")
        return TrialMetrics(
            status="ok",
            val_loss=0.0,
            peak_vram_mb=0.0,
            throughput_steps_per_sec=0.0,
            failure_reason=FailureReason.NONE,
        )

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            ["bash", str(launcher)],
            cwd=root,
            env=build_trial_env(root, spec),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return parse_trial_metrics(
        log_path.read_text(encoding="utf-8", errors="replace"), proc.returncode
    )


def command_probe(args: argparse.Namespace) -> int:
    root = repo_root()
    profile = load_profile(root / "research" / "profiles" / f"{args.profile}.env")
    trials = plan_probe_trials(profile)
    out_results = results_path(root)
    print(f"Profile: {profile.name}  VRAM ceiling: {profile.vram_ceiling_mb} MB")
    print(f"Trials planned: {len(trials)}")
    print(f"Results file: {out_results}")
    print("---")
    for i, spec in enumerate(trials, 1):
        print(f"[{i}/{len(trials)}] {spec.trial} ... ", end="", flush=True)
        metrics = run_trial(root, spec, dry_run=args.dry_run)
        print(
            f"{metrics.status}  val_loss={metrics.val_loss}  vram={metrics.peak_vram_mb}MB  throughput={metrics.throughput_steps_per_sec}steps/s"
        )
        append_result_row(
            out_results,
            spec,
            metrics,
            git_commit=git_short(root),
            log_path=spec.log_path(root),
            output_dir=spec.output_dir(root),
        )
    print("---")
    print(f"Done. {len(trials)} trials → {out_results}")
    return 0


def _parse_selected_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def command_sweep(args: argparse.Namespace) -> int:
    root = repo_root()
    selected = _parse_selected_env(
        root / "research" / "selected" / f"{args.profile}.env"
    )
    trials = plan_sweep_trials(args.profile, selected)
    out_results = results_path(root)
    print(f"Profile: {args.profile}  Sweep trials: {len(trials)}")
    print(f"Results file: {out_results}")
    print("---")
    for i, spec in enumerate(trials, 1):
        print(f"[{i}/{len(trials)}] {spec.phase}/{spec.trial} ... ", end="", flush=True)
        metrics = run_trial(root, spec, dry_run=args.dry_run)
        print(
            f"{metrics.status}  val_loss={metrics.val_loss}  vram={metrics.peak_vram_mb}MB  throughput={metrics.throughput_steps_per_sec}steps/s"
        )
        append_result_row(
            out_results,
            spec,
            metrics,
            git_commit=git_short(root),
            log_path=spec.log_path(root),
            output_dir=spec.output_dir(root),
        )
    print("---")
    print(f"Done. {len(trials)} trials → {out_results}")
    return 0


def command_select(args: argparse.Namespace) -> int:
    root = repo_root()
    profile = load_profile(root / "research" / "profiles" / f"{args.profile}.env")
    specs = {spec.trial: spec for spec in plan_probe_trials(profile)}
    rows: list[tuple[TrialSpec, TrialMetrics]] = []
    in_results = results_path(root)
    if not in_results.exists():
        raise SystemExit(f"Missing results file: {in_results}")

    for line in in_results.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 22 or parts[2] != args.profile or parts[3] != "probe":
            continue
        spec = specs.get(parts[4])
        if spec is None:
            continue
        metrics = TrialMetrics(
            status=parts[18],
            val_loss=float(parts[15]) if parts[15] else None,
            peak_vram_mb=float(parts[16]) if parts[16] else None,
            throughput_steps_per_sec=float(parts[17]) if parts[17] else None,
            failure_reason=FailureReason(parts[19])
            if parts[19]
            else FailureReason.NONE,
        )
        rows.append((spec, metrics))

    selected = choose_capacity(rows, vram_ceiling_mb=profile.vram_ceiling_mb)
    out = write_selected_env(root / "research" / "selected", selected)
    print(out)
    return 0


def command_subagents(args: argparse.Namespace) -> int:
    providers = discover_providers()
    if args.provider == "all":
        selected = sorted(providers)
    else:
        selected = [AgentProvider(args.provider)]

    if not selected:
        print("No requested subagent providers are available")
        return 0

    prompt = implementation_prompt(
        "B200 profiled experimentation implementation",
        [
            "qwen-vl-finetune/research/",
            "qwen-vl-finetune/tests/research/",
        ],
        "cd qwen-vl-finetune && PYTHONPATH=. pytest tests/research -v",
    )
    for provider in selected:
        if provider not in providers:
            print(f"{provider.value}: unavailable")
            continue
        task = SubagentTask(
            provider=provider,
            worktree=f"b200-research-{provider.value}",
            workspace=workspace_root(),
            prompt=prompt,
        )
        print(shlex.join(build_command(task)))
    return 0


def command_summarize(args: argparse.Namespace) -> int:
    root = repo_root()
    path = root / "experiments" / "summary.md"
    analysis_summary = summarize_campaign(root, profile=args.profile)
    if (
        "No successful analysis artifacts found." not in analysis_summary
        or (root / "experiments" / "runs" / args.profile).exists()
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(analysis_summary, encoding="utf-8")
        print(path)
        return 0

    result_file = results_path(root)
    selected_file = root / "research" / "selected" / f"{args.profile}.env"
    selected = (
        selected_file.read_text(encoding="utf-8")
        if selected_file.exists()
        else "No selected capacity yet.\n"
    )
    crashes: list[str] = []
    winners: dict[str, tuple[str, float]] = {}
    if result_file.exists():
        for line in result_file.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 22 or parts[2] != args.profile:
                continue
            if parts[18] != "ok":
                crashes.append(f"- {parts[3]}/{parts[4]}: {parts[19]}")
                continue
            if parts[15]:
                val = float(parts[15])
                phase = parts[3]
                if phase not in winners or val < winners[phase][1]:
                    winners[phase] = (parts[4], val)
    winner_lines = [
        f"- {phase}: {trial} val_loss={loss}"
        for phase, (trial, loss) in sorted(winners.items())
    ]
    path.write_text(
        "# Qwen3-VL Research Summary\n\n"
        f"Profile: `{args.profile}`\n\n"
        "## Selected Capacity\n\n"
        f"```bash\n{selected}```\n\n"
        "## Phase Winners\n\n"
        + ("\n".join(winner_lines) if winner_lines else "No successful phase rows yet.")
        + "\n\n## Crashes\n\n"
        + ("\n".join(crashes) if crashes else "No crashes recorded.")
        + f"\n\nResults: `{result_file}`\n",
        encoding="utf-8",
    )
    print(path)
    return 0


async def run_temporal_command(profile: str, command: str, dry_run: bool) -> dict:
    root = repo_root()
    server_proc = await ensure_local_temporal_server(root, DEFAULT_ADDRESS)
    client = await Client.connect(DEFAULT_ADDRESS)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as activity_executor:
        worker = Worker(
            client,
            task_queue=DEFAULT_TASK_QUEUE,
            workflows=[ProfiledExperimentWorkflow, SingleTrialWorkflow],
            activities=[
                load_profile_activity,
                plan_probe_trials_activity,
                plan_sweep_trials_activity,
                run_trial_activity,
                select_capacity_from_results_activity,
                select_capacity_activity,
                summarize_campaign_activity,
            ],
            activity_executor=activity_executor,
        )
        try:
            async with worker:
                handle = await client.start_workflow(
                    ProfiledExperimentWorkflow.run,
                    args=[profile, dry_run, command],
                    id=(
                        f"qwen3vl-{profile}-{command}-"
                        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
                    ),
                    task_queue=DEFAULT_TASK_QUEUE,
                )
                return await handle.result()
        finally:
            terminate_local_temporal_server(server_proc)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile-aware Qwen3-VL experiment runner"
    )
    parser.add_argument(
        "command", choices=["probe", "select", "sweep", "summarize", "subagents"]
    )
    parser.add_argument("--profile", default="b200")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--provider",
        choices=["all", "coco", "cursor_agent"],
        default="all",
    )
    args = parser.parse_args()

    if args.command == "subagents":
        return command_subagents(args)
    if args.command == "probe":
        return command_probe(args)
    if args.command == "select":
        return command_select(args)
    if args.command == "sweep":
        return command_sweep(args)
    if args.command == "summarize":
        return command_summarize(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
