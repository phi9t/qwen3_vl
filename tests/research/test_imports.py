from __future__ import annotations

from pathlib import Path

import tomli


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_research_package_imports() -> None:
    import research

    assert research.__version__ == "0.1.0"


def test_pyproject_exposes_research_script() -> None:
    pyproject = tomli.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert pyproject["project"]["name"] == "qwen3-vl-research"
    assert pyproject["project"]["scripts"]["research"] == "research.cli:main"
