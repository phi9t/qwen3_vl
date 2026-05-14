"""Adapter loading helpers for scientific experiments."""

from __future__ import annotations

import importlib
import importlib.util
import pathlib

from research import models


DEFAULT_ADAPTERS = {
    "qwen_vl": "qwen-vl-finetune/experiments/qwen_adapter.py:QwenVlAdapter",
}

_REQUIRED_METHODS = (
    "generate_probe_intents",
    "preflight",
    "build_trial",
    "parse_progress",
    "analyze_result",
)


def load_adapter(name_or_path: str) -> models.ExperimentAdapter:
    """Load and validate an experiment adapter by registry key or import path."""
    target = DEFAULT_ADAPTERS.get(name_or_path, name_or_path)
    module_name, class_name = _split_target(target, name_or_path)
    module = _load_module(module_name)
    adapter_class = getattr(module, class_name)
    adapter = adapter_class()
    _validate_adapter(adapter, target)
    return adapter


def _split_target(target: str, source: str) -> tuple[str, str]:
    if ":" not in target:
        raise ValueError(
            f"Expected adapter path 'module:Class' or 'file.py:Class', got {source!r}."
        )
    return target.split(":", 1)


def _load_module(module_name: str) -> object:
    if module_name.endswith(".py") or "/" in module_name:
        return _load_file_module(module_name)
    return importlib.import_module(module_name)


def _load_file_module(module_name: str) -> object:
    module_path = pathlib.Path(module_name)
    if not module_path.is_absolute():
        cwd_path = pathlib.Path.cwd() / module_path
        if cwd_path.exists():
            module_path = cwd_path
        else:
            module_path = pathlib.Path(__file__).resolve().parents[1] / module_path
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load adapter module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_adapter(adapter: object, target: str) -> None:
    missing = []
    for method_name in _REQUIRED_METHODS:
        method = getattr(adapter, method_name, None)
        if not callable(method):
            missing.append(method_name)
    if missing:
        missing_methods = ", ".join(missing)
        raise TypeError(f"Adapter {target} is missing methods: {missing_methods}.")
