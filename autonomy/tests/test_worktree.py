from __future__ import annotations

import pathlib
import subprocess

import pytest

from autonomy.worktree import create, destroy_if_clean


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "README.md").write_text("# test")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], check=True)
    return repo


def test_create_makes_worktree_and_branch(git_repo):
    wt = create(git_repo, "my-slug", "T01")

    expected = git_repo / ".autonomy" / "worktrees" / "my-slug" / "T01"
    assert wt == expected.resolve()
    assert wt.exists()
    assert (wt / "README.md").exists()

    result = subprocess.run(
        ["git", "-C", str(git_repo), "branch", "--list", "autonomy/my-slug/T01"],
        capture_output=True, text=True,
    )
    assert "autonomy/my-slug/T01" in result.stdout


def test_create_idempotent(git_repo):
    wt1 = create(git_repo, "my-slug", "T01")
    wt2 = create(git_repo, "my-slug", "T01")

    assert wt1 == wt2


def test_destroy_if_clean_not_done_returns_false(git_repo):
    wt = create(git_repo, "my-slug", "T02")

    result = destroy_if_clean(git_repo, wt, ended_done=False)
    assert result is False
    assert wt.exists()


def test_destroy_if_clean_dirty_returns_false(git_repo):
    wt = create(git_repo, "my-slug", "T03")
    (wt / "dirty.txt").write_text("dirty")

    result = destroy_if_clean(git_repo, wt, ended_done=True)
    assert result is False
    assert wt.exists()


def test_destroy_if_clean_clean_and_done_returns_true(git_repo):
    wt = create(git_repo, "my-slug", "T04")

    result = destroy_if_clean(git_repo, wt, ended_done=True)
    assert result is True
    assert not wt.exists()