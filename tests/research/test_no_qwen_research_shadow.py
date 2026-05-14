"""Tests that the old Qwen research package no longer shadows top-level research."""

from __future__ import annotations

import pathlib


def test_qwen_finetune_no_longer_shadows_top_level_research_package() -> None:
    """The finetune tree should not contain an older `research` package."""
    assert not pathlib.Path("qwen-vl-finetune/research/__init__.py").exists()
    assert not pathlib.Path("qwen-vl-finetune/research/runner.py").exists()
