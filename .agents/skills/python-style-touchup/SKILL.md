---
name: python-style-touchup
description: Pre-commit Python style touch-up following Google style guide. Formats, lints, and reports docstring/type-annotation gaps on staged Python files. Use when the user asks to clean up Python code before committing, do a style pass, fix lint errors, or run a pre-commit check on Python files.
---

# Python Style Touch-Up

Automated pre-commit style touch-up for Python files, following the project's [Google-style guide](docs/python-style-guide.md).

## Workflow

Run the touch-up script, then review and fix any remaining manual issues.

### Step 1: Run automated fixes

Execute the touch-up script against staged files:

```bash
bash .agents/skills/python-style-touchup/scripts/touchup.sh
```

The script:
1. Collects staged `.py` files (`git diff --cached`)
2. Runs `ruff format` (auto-formatting)
3. Runs `ruff check --fix` (auto-fixable lint)
4. Prints a summary of what was fixed and what remains

If no files are staged, it falls back to all modified `.py` files in the working tree.

### Step 2: Re-stage auto-fixed files

After the script runs, re-add any files it modified:

```bash
git add $(git diff --name-only --diff-filter=M | grep '\.py$')
```

### Step 3: Address remaining issues manually

Read the script output. For each remaining violation:

**Missing module docstring (`D100`/`D104`)**
Add a one-line summary + optional longer description at the top of the file, before imports:

```python
"""One-line summary of what this module does.

Optional longer description here.
"""
from __future__ import annotations
```

**Missing function/class docstring (`D101`/`D102`/`D103`)**
Add a Google-style docstring immediately after the `def`/`class` line. See the format in [docs/python-style-guide.md](docs/python-style-guide.md#3-docstrings):

```python
def my_function(arg: str) -> int:
    """Brief description of what this function does.

    Args:
        arg: Description of the argument.

    Returns:
        Description of the return value.
    """
```

**Missing type annotations (`ANN001`/`ANN201`)**
Add parameter and return type annotations. Prefer `X | None` over `Optional[X]`, built-in generics (`list[str]`) over `typing.List[str]`, and `collections.abc` types for signatures.

```python
# Before
def process(items, config=None):
    ...

# After
from __future__ import annotations
from collections.abc import Sequence

def process(items: Sequence[str], config: dict[str, str] | None = None) -> list[str]:
    ...
```

**Import order (`I001`)**
Ruff fixes these automatically. If any remain: order is `__future__` → stdlib → third-party → local, each group separated by a blank line.

## Quick reference

| Ruff rule | Meaning | Fix |
|-----------|---------|-----|
| `D100–D107` | Missing docstring | Add Google-style docstring |
| `ANN001` | Missing param annotation | Add `: Type` |
| `ANN201` | Missing return annotation | Add `-> Type` |
| `E501` | Line too long | Break at highest syntactic level |
| `I001` | Import order | Auto-fixed by `ruff check --fix` |
| `UP007` | `Optional[X]` → `X \| None` | Auto-fixed |
| `B006` | Mutable default argument | Use `None` default + guard |

Full style rules: [docs/python-style-guide.md](docs/python-style-guide.md)
