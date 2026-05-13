from __future__ import annotations

import pathlib
import subprocess


def create(
    repo_root: pathlib.Path,
    slug: str,
    task_id: str,
    branch_override: str | None = None,
) -> pathlib.Path:
    worktree_path = (repo_root / ".autonomy" / "worktrees" / slug / task_id).resolve()
    branch = branch_override or f"autonomy/{slug}/{task_id}"

    if worktree_path.exists():
        result = subprocess.run(
            ["git", "-C", str(repo_root), "worktree", "list", "--porcelain"],
            capture_output=True, text=True,
        )
        registered = False
        for line in result.stdout.splitlines():
            if line.startswith("worktree ") and pathlib.Path(line.split(" ", 1)[1]).resolve() == worktree_path:
                registered = True
                break
        if registered:
            return worktree_path

    try:
        subprocess.run(
            ["git", "-C", str(repo_root), "worktree", "add", str(worktree_path), "-b", branch],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.strip() if e.stderr else ""
        if "already exists" in stderr_text:
            subprocess.run(
                ["git", "-C", str(repo_root), "worktree", "add", str(worktree_path), branch],
                capture_output=True, text=True, check=True,
            )
        else:
            raise RuntimeError(
                f"git worktree add failed: {stderr_text}"
            ) from e

    return worktree_path


def destroy_if_clean(
    repo_root: pathlib.Path,
    worktree_path: pathlib.Path,
    ended_done: bool,
) -> bool:
    if not ended_done:
        return False

    result = subprocess.run(
        ["git", "-C", str(worktree_path), "status", "--porcelain"],
        capture_output=True, text=True,
    )
    if result.stdout.strip():
        return False

    try:
        subprocess.run(
            ["git", "-C", str(repo_root), "worktree", "remove", str(worktree_path)],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.strip() if e.stderr else ""
        raise RuntimeError(
            f"git worktree remove failed: {stderr_text}"
        ) from e

    return True