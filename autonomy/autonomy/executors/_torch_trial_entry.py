from __future__ import annotations

import json
import os
import pathlib
import sys


def _find_repo_root() -> pathlib.Path:
    cwd = pathlib.Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not find .git directory walking up from CWD")


def _read_task_properties(
    tracker_path: pathlib.Path, task_id: str
) -> dict[str, str]:
    import orgparse

    root = orgparse.load(str(tracker_path))
    for node in root.children:
        if node.heading and node.heading.strip().lower() == "tasks":
            for child in node.children:
                props = child.properties or {}
                if props.get("ID", "").strip() == task_id:
                    return dict(props)
    raise ValueError(
        f"Task with :ID: {task_id!r} not found in {tracker_path}"
    )


def main() -> None:
    slug = os.environ.get("AUTONOMY_TASK_SLUG")
    task_id = os.environ.get("AUTONOMY_TASK_ID")
    if not slug or not task_id:
        print(
            "AUTONOMY_TASK_SLUG and AUTONOMY_TASK_ID must be set",
            file=sys.stderr,
        )
        sys.exit(1)

    repo_root = _find_repo_root()
    tracker_path = (
        repo_root / ".autonomy" / "runs" / slug / "tracker.org"
    )

    props = _read_task_properties(tracker_path, task_id)

    profile = props.get("PROFILE", "")
    phase = props.get("PHASE", "")
    trial = props.get("TRIAL", "")
    if not profile or not phase or not trial:
        print(
            "PROFILE, PHASE, and TRIAL must be set in task :PROPERTIES:",
            file=sys.stderr,
        )
        sys.exit(1)

    env: dict[str, str] = {}
    for key, value in props.items():
        if key.startswith("ENV_"):
            env[key[len("ENV_") :]] = value

    from research.models import TrialSpec
    from research.observability.trial_runner import SupervisorTrialRunner

    spec = TrialSpec(profile=profile, phase=phase, trial=trial, env=env)
    runner = SupervisorTrialRunner(root=repo_root)
    analysis = runner.run_from_spec(spec)
    print(json.dumps(analysis.to_payload()))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"torch-trial entry failed: {exc}", file=sys.stderr)
        sys.exit(1)