"""Tests that the old autonomy methodology package has been removed."""

from __future__ import annotations

import pathlib


def test_autonomy_methodology_removed() -> None:
    """The autonomy package should not coexist with the generic research core."""
    assert not pathlib.Path("autonomy/autonomy/workflows.py").exists()
    assert not pathlib.Path("autonomy/autonomy/org").exists()
