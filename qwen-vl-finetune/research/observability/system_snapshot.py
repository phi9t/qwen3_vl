from __future__ import annotations

import subprocess
import sys
from importlib import metadata
from typing import Any

from research.observability.redaction import redact_mapping
from research.observability.schema import SCHEMA_VERSION


VERSION_PACKAGES = ("torch", "transformers", "accelerate", "deepspeed", "peft", "temporalio")


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in VERSION_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _gpu_inventory() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []

    gpus: list[dict[str, Any]] = []
    for raw in output.splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 5:
            continue
        index = _parse_int(parts[0])
        memory_total = _parse_int(parts[2])
        memory_used = _parse_int(parts[3])
        utilization = _parse_int(parts[4])
        if None in (index, memory_total, memory_used, utilization):
            continue
        gpus.append(
            {
                "index": index,
                "name": parts[1],
                "memory_total_mb": memory_total,
                "memory_used_mb": memory_used,
                "utilization_gpu_percent": utilization,
            }
        )
    return gpus


def collect_system_snapshot(
    *,
    env: dict[str, str],
    include_process_topology: bool,
    process_topology: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": sys.platform,
        },
        "packages": _package_versions(),
        "gpus": _gpu_inventory(),
        "env": redact_mapping(env),
    }
    if include_process_topology:
        payload["process_topology"] = process_topology or []
    return payload
