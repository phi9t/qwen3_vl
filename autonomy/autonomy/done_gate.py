from __future__ import annotations

"""Done-gate composition for Symphony tracker workflows.

Check order: tests -> typecheck -> gpt2-smoke -> cmd:* (acceptance cmds).
Stop on first failure.

Typecheck: ruff is the hard gate; pyright is soft (|| true) because pyright
is not yet pinned in pyproject.toml.
"""

from dataclasses import dataclass
from pathlib import Path

from autonomy import sandbox
from autonomy.org.schema import OrgTask


@dataclass(frozen=True)
class GateCheck:
    name: str
    ok: bool
    summary: str
    log_excerpt: str


@dataclass(frozen=True)
class GateResult:
    ok: bool
    checks: list[GateCheck]
    artifacts: list[str]
    summary: str


_KNOWN_CHECKS = {"tests", "typecheck", "gpt2-smoke"}


def _last_nonempty_line(lines: list[str]) -> str:
    for line in reversed(lines):
        stripped = line.rstrip("\n").strip()
        if stripped:
            return stripped
    return ""


def _read_log_excerpt(log_path: Path) -> tuple[str, str]:
    if not log_path.exists():
        return "", ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    summary = _last_nonempty_line(lines)
    excerpt_lines = lines[-30:] if len(lines) >= 30 else lines
    return summary, "\n".join(excerpt_lines)


def _build_command(check_name: str, task: OrgTask) -> list[str]:
    if check_name == "tests":
        return ["uv", "run", "pytest", "-x", "--quiet"]
    if check_name == "typecheck":
        return [
            "bash",
            "-lc",
            "uv run ruff check . && uv run pyright autonomy/ qwen-vl-finetune/ || true",
        ]
    if check_name == "gpt2-smoke":
        return ["python", "-m", "autonomy.smoke.gpt2_synthetic"]
    if check_name.startswith("cmd:"):
        idx = int(check_name.split(":", 1)[1])
        return ["/bin/bash", "-lc", task.acceptance_cmds[idx]]
    raise ValueError(f"unknown check: {check_name}")


def _gpus_for_check(check_name: str, task: OrgTask) -> int:
    if check_name == "gpt2-smoke":
        return task.gpus
    return 0


def run_gate(
    task: OrgTask,
    worktree: Path,
    gate_spec: str,
    repo_root: Path,
    artifact_dir: Path,
) -> GateResult:
    if gate_spec == "none":
        return GateResult(ok=True, checks=[], artifacts=[], summary="gate=none")

    names = [name.strip() for name in gate_spec.split("+") if name.strip()]
    for name in names:
        if name not in _KNOWN_CHECKS:
            raise ValueError(f"unknown gate check: {name}")

    ordered = list(names)
    for i, cmd in enumerate(task.acceptance_cmds):
        ordered.append(f"cmd:{i}")

    checks: list[GateCheck] = []
    for check_name in ordered:
        log_path = artifact_dir / f"{check_name}.log"
        cmd = _build_command(check_name, task)
        gpus = _gpus_for_check(check_name, task)
        result = sandbox.draccus_exec(
            cmd,
            workspace=worktree,
            gpus=gpus,
            log_path=log_path,
        )
        summary, excerpt = _read_log_excerpt(log_path)
        check = GateCheck(
            name=check_name,
            ok=result.exit_code == 0,
            summary=summary,
            log_excerpt=excerpt,
        )
        checks.append(check)
        if not check.ok:
            artifacts = [str(artifact_dir / f"{c.name}.log") for c in checks]
            return GateResult(
                ok=False,
                checks=checks,
                artifacts=artifacts,
                summary=f"fail:{check_name}",
            )

    artifacts = [str(artifact_dir / f"{c.name}.log") for c in checks]
    return GateResult(ok=True, checks=checks, artifacts=artifacts, summary="pass")
