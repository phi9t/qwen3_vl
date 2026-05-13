from __future__ import annotations

import dataclasses
from datetime import timedelta
from pathlib import Path

import pytest

from autonomy.done_gate import GateResult, run_gate
from autonomy.org.schema import OrgTask
from autonomy.sandbox import ExecResult


def _make_task(
    *,
    gate: str = "none",
    acceptance_cmds: list[str] | None = None,
) -> OrgTask:
    return OrgTask(
        id="T01",
        executor="shell",
        gate=gate,
        depends=frozenset(),
        timeout=timedelta(hours=1),
        gpus=1,
        branch=None,
        goal="test",
        constraints=[],
        acceptance_cmds=acceptance_cmds or [],
        slug="test-run",
        state="TODO",
        position=1,
    )


def _make_exec_result(exit_code: int) -> ExecResult:
    return ExecResult(
        exit_code=exit_code,
        log_path=Path("/dev/null"),
        stdout_tail="",
    )


def test_gate_none_returns_ok(tmp_path, monkeypatch):
    task = _make_task(gate="none")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    result = run_gate(task, tmp_path, "none", tmp_path, artifact_dir)
    assert result.ok is True
    assert result.checks == []
    assert result.artifacts == []
    assert result.summary == "gate=none"


def test_gate_tests_pass(tmp_path, monkeypatch):
    task = _make_task(gate="tests")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    calls = []

    def fake_draccus_exec(cmd, *, workspace, gpus, log_path, **kwargs):
        calls.append((cmd, workspace, gpus, log_path))
        log_path.write_text("ok\n")
        return _make_exec_result(0)

    monkeypatch.setattr("autonomy.done_gate.sandbox.draccus_exec", fake_draccus_exec)

    result = run_gate(task, tmp_path, "tests", tmp_path, artifact_dir)
    assert result.ok is True
    assert len(result.checks) == 1
    assert result.checks[0].name == "tests"
    assert result.checks[0].ok is True
    assert result.summary == "pass"
    assert len(calls) == 1


def test_gate_tests_fail_stops_before_typecheck(tmp_path, monkeypatch):
    task = _make_task(gate="tests+typecheck")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    calls = []

    def fake_draccus_exec(cmd, *, workspace, gpus, log_path, **kwargs):
        calls.append((cmd, workspace, gpus, log_path))
        log_path.write_text("fail\n")
        if len(calls) == 1:
            return _make_exec_result(1)
        return _make_exec_result(0)

    monkeypatch.setattr("autonomy.done_gate.sandbox.draccus_exec", fake_draccus_exec)

    result = run_gate(task, tmp_path, "tests+typecheck", tmp_path, artifact_dir)
    assert result.ok is False
    assert result.summary == "fail:tests"
    assert len(result.checks) == 1
    assert result.checks[0].name == "tests"
    assert result.checks[0].ok is False
    assert len(calls) == 1


def test_gate_all_three_pass(tmp_path, monkeypatch):
    task = _make_task(gate="tests+typecheck+gpt2-smoke")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    calls = []

    def fake_draccus_exec(cmd, *, workspace, gpus, log_path, **kwargs):
        calls.append((cmd, workspace, gpus, log_path))
        log_path.write_text("ok\n")
        return _make_exec_result(0)

    monkeypatch.setattr("autonomy.done_gate.sandbox.draccus_exec", fake_draccus_exec)

    result = run_gate(task, tmp_path, "tests+typecheck+gpt2-smoke", tmp_path, artifact_dir)
    assert result.ok is True
    assert len(result.checks) == 3
    assert [c.name for c in result.checks] == ["tests", "typecheck", "gpt2-smoke"]
    assert result.summary == "pass"
    assert len(calls) == 3


def test_gate_appends_acceptance_cmds(tmp_path, monkeypatch):
    task = _make_task(gate="tests", acceptance_cmds=["echo hi", "exit 2"])
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    calls = []

    def fake_draccus_exec(cmd, *, workspace, gpus, log_path, **kwargs):
        calls.append((cmd, workspace, gpus, log_path))
        log_path.write_text("ok\n")
        if len(calls) == 2:
            return _make_exec_result(2)
        return _make_exec_result(0)

    monkeypatch.setattr("autonomy.done_gate.sandbox.draccus_exec", fake_draccus_exec)

    result = run_gate(task, tmp_path, "tests", tmp_path, artifact_dir)
    assert result.ok is False
    assert result.summary == "fail:cmd:0"
    assert len(result.checks) == 2
    assert result.checks[0].name == "tests"
    assert result.checks[0].ok is True
    assert result.checks[1].name == "cmd:0"
    assert result.checks[1].ok is False


def test_gate_unknown_name_raises(tmp_path, monkeypatch):
    task = _make_task(gate="tests+badcheck")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    with pytest.raises(ValueError, match="unknown gate check: badcheck"):
        run_gate(task, tmp_path, "tests+badcheck", tmp_path, artifact_dir)
