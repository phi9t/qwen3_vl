from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import pytest

from autonomy.sandbox import draccus_exec, ExecResult


class _FakePopen:
    def __init__(self, args, env, stdout, stderr, cwd):
        self.args = args
        self.env = env
        self._stdout_data = [b"hello\n", b"world\n"]
        self._idx = 0
        self.returncode = 0
        self._killed = False
        self._timed_out = False

    @property
    def stdout(self):
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self._stdout_data):
            raise StopIteration
        data = self._stdout_data[self._idx]
        self._idx += 1
        return data

    def wait(self, timeout=None):
        if self._timed_out:
            raise subprocess.TimeoutExpired(cmd=self.args, timeout=timeout or 0)
        return self.returncode

    def kill(self):
        self._killed = True


def test_draccus_exec_writes_log_and_returns_exit_code(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "run.log"
    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured_env = {}

    def fake_popen(args, env, stdout, stderr, cwd):
        captured_env.update(env)
        return _FakePopen(args, env, stdout, stderr, cwd)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    result = draccus_exec(
        ["echo", "hi"],
        workspace=workspace,
        gpus=0,
        log_path=log_path,
        _draccus_run_path=pathlib.Path("/fake/draccus-run"),
    )

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.log_path == log_path
    assert log_path.exists()
    content = log_path.read_text()
    assert "hello" in content
    assert "world" in content
    assert "hello\nworld" in result.stdout_tail


def test_gpus_zero_sets_empty_cuda_visible_devices(tmp_path, monkeypatch):
    log_path = tmp_path / "run.log"
    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured_env = {}

    def fake_popen(args, env, stdout, stderr, cwd):
        captured_env.update(env)
        return _FakePopen(args, env, stdout, stderr, cwd)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    draccus_exec(
        ["true"],
        workspace=workspace,
        gpus=0,
        log_path=log_path,
        _draccus_run_path=pathlib.Path("/fake/draccus-run"),
    )

    assert captured_env.get("CUDA_VISIBLE_DEVICES") == ""


def test_gpus_gt_zero_no_nvidia_smi_raises(tmp_path, monkeypatch):
    log_path = tmp_path / "run.log"
    workspace = tmp_path / "ws"
    workspace.mkdir()

    monkeypatch.setattr(shutil, "which", lambda _x: None)

    with pytest.raises(RuntimeError, match="CUDA not available"):
        draccus_exec(
            ["true"],
            workspace=workspace,
            gpus=1,
            log_path=log_path,
            _draccus_run_path=pathlib.Path("/fake/draccus-run"),
        )


def test_file_not_found_when_draccus_run_missing(tmp_path, monkeypatch):
    log_path = tmp_path / "run.log"
    workspace = tmp_path / "ws"
    workspace.mkdir()

    nonexistent = tmp_path / "nonexistent" / "draccus-run"

    with pytest.raises(FileNotFoundError):
        draccus_exec(
            ["true"],
            workspace=workspace,
            gpus=0,
            log_path=log_path,
            _draccus_run_path=nonexistent,
        )


def test_timeout_kills_process(tmp_path, monkeypatch):
    log_path = tmp_path / "run.log"
    workspace = tmp_path / "ws"
    workspace.mkdir()

    class _TimeoutPopen(_FakePopen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._timed_out = True

        def kill(self):
            self._killed = True
            self._timed_out = False

    def fake_popen(args, env, stdout, stderr, cwd):
        return _TimeoutPopen(args, env, stdout, stderr, cwd)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    result = draccus_exec(
        ["sleep", "999"],
        workspace=workspace,
        gpus=0,
        log_path=log_path,
        timeout=0.1,
        _draccus_run_path=pathlib.Path("/fake/draccus-run"),
    )

    assert "killed after timeout" in result.stdout_tail
    content = log_path.read_text()
    assert "killed after timeout" in content