from __future__ import annotations

import collections
import os
import pathlib
import shutil
import subprocess
import threading

from dataclasses import dataclass


@dataclass
class ExecResult:
    exit_code: int
    log_path: pathlib.Path
    stdout_tail: str


_GPU_LOCK = threading.Lock()
_GPU_COUNTER = 0


def _find_draccus_run() -> pathlib.Path:
    current = pathlib.Path(__file__).resolve().parent
    for _ in range(20):
        candidate = current / "draccus" / "bin" / "draccus-run"
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    raise FileNotFoundError(
        "Cannot locate draccus/bin/draccus-run. "
        "Expected a 'draccus/bin/draccus-run' in a parent directory of "
        f"{pathlib.Path(__file__).resolve().parent}"
    )


def _pick_gpu_indices(gpus: int) -> str:
    if gpus == 0:
        return ""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("CUDA not available: nvidia-smi not found on PATH")
    result = subprocess.run(
        [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    all_gpus = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    if not all_gpus:
        raise RuntimeError("CUDA not available: nvidia-smi reported zero GPUs")
    if gpus > len(all_gpus):
        raise RuntimeError(
            f"Requested {gpus} GPU(s) but only {len(all_gpus)} available"
        )
    global _GPU_COUNTER
    with _GPU_LOCK:
        start = _GPU_COUNTER % len(all_gpus)
        indices = []
        for i in range(gpus):
            indices.append(all_gpus[(start + i) % len(all_gpus)])
        _GPU_COUNTER = (_GPU_COUNTER + gpus) % len(all_gpus)
    return ",".join(indices)


def draccus_exec(
    cmd: list[str],
    *,
    workspace: pathlib.Path,
    gpus: int,
    log_path: pathlib.Path,
    env_overrides: dict[str, str] | None = None,
    timeout: float | None = None,
    _draccus_run_path: pathlib.Path | None = None,
) -> ExecResult:
    if _draccus_run_path is not None:
        draccus_run = _draccus_run_path
    else:
        draccus_run = _find_draccus_run()

    cuda_visible_devices = _pick_gpu_indices(gpus)

    env = os.environ.copy()
    env["DRACCUS_WORKSPACE"] = str(workspace)
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if env_overrides:
        env.update(env_overrides)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    tail_lines: collections.deque[str] = collections.deque(maxlen=30)

    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(
            [str(draccus_run), "--"] + cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=workspace,
        )
        assert proc.stdout is not None
        for line_bytes in proc.stdout:
            line = line_bytes.decode("utf-8", errors="replace")
            log_fh.write(line)
            log_fh.flush()
            tail_lines.append(line.rstrip("\n"))

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            marker = f"\n[draccus_exec] killed after timeout ({timeout}s)\n"
            log_fh.write(marker)
            log_fh.flush()
            tail_lines.append(marker.strip())

    return ExecResult(
        exit_code=proc.returncode,
        log_path=log_path,
        stdout_tail="\n".join(tail_lines),
    )