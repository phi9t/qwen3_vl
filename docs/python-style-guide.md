# Python Style Guide

This guide distills the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) into the conventions used in this codebase. Follow this document for all Python files under `qwen-vl-finetune/` and `qwen-vl-utils/`.

---

## 1. Language Rules

### 1.1 Imports

- Import **modules and packages**, not individual names (exceptions: `typing` and `collections.abc`).
- Use the **full package path**; no relative imports.
- One import per line (exception: `from typing import X, Y` and `from collections.abc import X, Y`).
- Group imports in this order, separated by a blank line each:
  1. `from __future__ import annotations` (always first)
  2. Standard library
  3. Third-party packages
  4. Local packages (`research`, `qwenvl`)
- Sort each group lexicographically.

```python
# Good
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import torch
from transformers import AutoProcessor

from research.models import TrialSpec
```

```python
# Bad
import os, subprocess          # multiple on one line
from .models import TrialSpec  # relative import
import research.models         # import module then use research.models.TrialSpec
```

### 1.2 `from __future__ import annotations`

Include in **every** Python file. It enables postponed evaluation of annotations (forward references work without quotes; type hints are not evaluated at runtime).

### 1.3 Exceptions

- Raise specific exception classes; prefer built-ins (`ValueError`, `RuntimeError`, `TypeError`).
- Never catch bare `except:` or `except Exception:` unless re-raising or at a top-level isolation boundary.
- Minimize code inside `try` blocks.
- Use `finally` for cleanup; prefer `with` statements for resources.
- Custom exceptions must inherit from an existing class and end in `Error`.

```python
# Good
if minimum < 1024:
    raise ValueError(f"Port must be >= 1024, got {minimum}.")

try:
    result = run_trial(spec)
except subprocess.CalledProcessError as exc:
    raise RuntimeError(f"Trial {spec.trial} failed with code {exc.returncode}.") from exc
```

### 1.4 Mutable Global State

Avoid mutable module-level variables. Module-level **constants** are fine and encouraged; name them `ALL_CAPS_WITH_UNDERSCORES`.

```python
# Good — constant
MAX_RETRY_ATTEMPTS = 3
_DEFAULT_TIMEOUT_S = 30  # internal constant

# Bad — mutable global
_results_cache = {}  # mutated at runtime without a clear owner
```

### 1.5 Comprehensions

One `for` clause and at most one `if` filter. Break onto multiple lines when the expression is long. Never nest multiple `for` clauses in a single comprehension.

```python
# Good
failed = [r for r in results if r.status != "ok"]

active_trials = [
    spec.trial
    for spec in probe_specs
    if spec.trial not in self.skip_trials
]

# Bad
pairs = [(x, y) for x in range(10) for y in range(5) if x * y > 10]
```

### 1.6 Default Argument Values

Never use mutable objects as default argument values.

```python
# Good
def run_trials(specs: list[dict], extra_env: dict[str, str] | None = None) -> list[dict]:
    if extra_env is None:
        extra_env = {}
    ...

# Bad
def run_trials(specs: list[dict], extra_env: dict[str, str] = {}) -> list[dict]:
    ...
```

### 1.7 True/False Evaluations

Use implicit truthiness for emptiness checks. Use `is None` / `is not None` explicitly for `None` checks.

```python
# Good
if not results:
    return
if metrics is None:
    metrics = {}

# Bad
if len(results) == 0:
    return
if metrics == None:
    metrics = {}
```

### 1.8 Properties and `staticmethod`/`classmethod`

- Use `@property` only for simple, cheap attribute-like access.
- Avoid `@staticmethod`; write a module-level function instead.
- Use `@classmethod` only for named constructors or class-level cache management.

---

## 2. Style Rules

### 2.1 Line Length

**88 characters** maximum (Ruff/Black default). Exceptions:

- Long URLs or paths in comments that cannot be split.
- `# pylint: disable=...` / `# type: ignore` comments.
- Ruff `noqa` directives.

Use implicit line continuation inside parentheses/brackets; never use `\` for line continuation.

```python
# Good
result = some_long_function_name(
    argument_one, argument_two, argument_three
)

# Bad
result = some_long_function_name(argument_one, argument_two, \
    argument_three)
```

### 2.2 Indentation

4 spaces. Never tabs. Continuation lines align with the opening delimiter or use a 4-space hanging indent.

### 2.3 Blank Lines

- **2 blank lines** between top-level definitions (functions, classes).
- **1 blank line** between method definitions within a class.
- **1 blank line** between the class docstring and the first method.
- No blank line immediately after a `def` or `class` line (before the docstring).

### 2.4 Whitespace

- No spaces inside brackets: `spam(ham[1], {'eggs': 2})`.
- No space before `:`, `,`, `;`.
- Space after `:`, `,`, `;` (except at end of line).
- No trailing whitespace.
- Surround binary operators with one space; no spaces around `=` in keyword arguments *unless* the argument has a type annotation.

```python
# Good
def foo(a: int, b: float = 0.0) -> str:
    return f"{a},{b}"

foo(a=1, b=2.0)

# Bad
def foo(a:int, b:float=0.0) -> str: ...
foo(a = 1, b = 2.0)
```

### 2.5 Strings

- Prefer **double quotes** (`"`) for string literals (matches Ruff formatter default).
- Use f-strings for interpolation in application code.
- Use `%`-style format strings for **logging calls** (not f-strings — preserves the unexpanded pattern for log aggregators).
- Multi-line strings: use `textwrap.dedent()` or implicit string concatenation rather than unindented heredocs.

```python
# Good — logging
import logging
logger = logging.getLogger(__name__)
logger.info("Trial %s completed in %.1fs.", trial_name, elapsed)

# Bad — logging
logger.info(f"Trial {trial_name} completed in {elapsed:.1f}s.")

# Good — error message
raise ValueError(f"Unknown profile: {profile_name!r}")
```

### 2.6 Semicolons and Statements

Never use semicolons. One statement per line.

### 2.7 Main Guard

Every executable script must use:

```python
def main() -> None:
    ...

if __name__ == "__main__":
    main()
```

---

## 3. Docstrings

### 3.1 Format

Use **Google-style docstrings** with `"""triple double quotes"""`. The summary line must fit within 80 characters and end with a period, question mark, or exclamation point.

```python
def parse_trial_metrics(log_text: str, return_code: int) -> TrialMetrics:
    """Parses structured metrics from a training run log.

    Extracts validation loss, peak VRAM, and throughput from the log footer
    emitted by the training script. Classifies failures when metrics are absent.

    Args:
        log_text: Full stdout/stderr content of the training subprocess.
        return_code: Exit code of the training subprocess.

    Returns:
        A TrialMetrics instance. status is "ok" when all metrics are present
        and return_code is 0; otherwise status reflects the failure reason.

    Raises:
        ValueError: If log_text is empty and return_code is 0.
    """
```

### 3.2 When Docstrings Are Required

A docstring is **mandatory** for:

- Every **module** (first statement in the file).
- Every **public class**.
- Every **public function or method** that is part of the public API, has nontrivial logic, or is not fully self-describing from its name and type signature alone.

**Not required** for:

- `@override` methods that add no new contract.
- Simple `__repr__` / `__eq__` methods.
- Test functions (module-level test docstrings are optional).

### 3.3 Module Docstrings

```python
"""Trial metrics extraction and failure classification for training runs.

Parses structured output from the training script footer and maps
subprocess exit codes to typed FailureReason values.
"""
from __future__ import annotations
...
```

### 3.4 Class Docstrings

The class docstring describes what an **instance** represents. Public attributes go in an `Attributes:` section.

```python
class HardwareProfile:
    """Hardware configuration profile for a training capacity sweep.

    Attributes:
        name: Profile identifier, e.g. "b200" or "h100".
        enabled: Whether this profile is active for scheduling.
        vram_ceiling_mb: VRAM budget ceiling in megabytes.
    """
```

### 3.5 Sections

| Section | Used when |
|---------|-----------|
| `Args:` | Function has parameters |
| `Returns:` | Non-None return value |
| `Yields:` | Generator function |
| `Raises:` | Function raises documented exceptions |
| `Attributes:` | Class with public attributes |
| `Example:` | Usage example is non-obvious |

Each parameter in `Args:` uses a 4-space hanging indent for wrapped descriptions:

```
Args:
    log_text: Full stdout/stderr content of the training subprocess.
        May include ANSI escape codes, which are stripped before parsing.
    return_code: Exit code from the subprocess.
```

---

## 4. Naming

| Entity | Convention | Example |
|--------|------------|---------|
| Module / package | `lower_with_under` | `profile_config`, `research` |
| Class | `CapWords` | `HardwareProfile`, `TrialSpec` |
| Exception | `CapWords` ending in `Error` | `CapacitySelectionError` |
| Function / method | `lower_with_under()` | `parse_trial_metrics()` |
| Global constant | `CAPS_WITH_UNDER` | `MAX_RETRY_ATTEMPTS` |
| Module-internal constant | `_CAPS_WITH_UNDER` | `_VAL_RE`, `_VRAM_RE` |
| Instance / class variable | `lower_with_under` | `peak_vram_mb` |
| Protected member | `_lower_with_under` | `_home_local_bin()` |

**Avoid:**

- Single-character names outside of counters (`i`, `j`, `k`) or exception identifiers (`e`).
- Abbreviations that are ambiguous outside the project.
- Names that include the type (`id_to_name_dict` → `names_by_id`).
- Dashes in module or package names.

---

## 5. Type Annotations

### 5.1 General Rules

- **Always** annotate public API functions (arguments and return type).
- Include `from __future__ import annotations` so annotations are evaluated lazily.
- Use **built-in generics** (`list[int]`, `dict[str, str]`, `tuple[int, ...]`) not `typing.List`, `typing.Dict`, etc.
- Use `X | None` (union shorthand) not `Optional[X]`.
- Use `collections.abc` abstract types in signatures (`Sequence`, `Mapping`, `Callable`, `Iterator`) not concrete types.
- Explicitly annotate `-> None` on functions that return nothing.

```python
# Good
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path


def discover_providers(
    which: Callable[[str], str | None] = shutil_which,
) -> dict[AgentProvider, str]:
    ...


def write_selected_env(path: Path, env: dict[str, str]) -> None:
    ...
```

```python
# Bad
from typing import Callable, Dict, List, Optional

def discover_providers(which: Optional[Callable] = None) -> Dict:
    ...
```

### 5.2 `self` and `cls`

Do not annotate `self` or `cls`. Annotate `__init__` parameters but not its return type.

### 5.3 TypeAlias

Name type aliases with `CapWords`; prefix private aliases with `_`.

```python
from typing import TypeAlias

_EnvMap: TypeAlias = dict[str, str]
TrialRow: TypeAlias = dict[str, str | float | None]
```

### 5.4 Suppressing Type Errors

Use `# type: ignore[<code>]` with a specific code. Add a comment explaining the suppression.

```python
result = some_untyped_library_call()  # type: ignore[no-any-return]  # third-party stub missing
```

---

## 6. ML/PyTorch-Specific Notes

- **`assert` statements**: Use only in `pytest` tests. In production code, raise explicit exceptions instead.
- **Logging**: Always pass the format string and arguments separately (`logger.info("val_loss=%s", val)`, not f-strings).
- **`torch.no_grad()`**: Use as a context manager (`with torch.no_grad():`), not as a decorator on non-trivial methods.
- **Global model/optimizer state**: Avoid module-level mutable state. Pass model and optimizer as function arguments or encapsulate in a class.
- **`subprocess`**: Capture output explicitly; pass `check=True` or handle `CalledProcessError` explicitly.
- **Environment variables**: Read at call time, not at import time, to avoid side effects during module import.

---

## 7. Tooling

| Tool | Purpose | Config |
|------|---------|--------|
| `ruff format` | Auto-formatting (Black-compatible) | `pyproject.toml` (root) |
| `ruff check --fix` | Linting + auto-fixable issues | `pyproject.toml` (root) |
| `mypy` or `pyright` | Static type checking | Add as needed |
| `pytest` | Unit tests | `qwen-vl-finetune/tests/` |

Run before every commit:

```bash
# Format and fix all staged Python files
ruff format $(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')
ruff check --fix $(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')
```

See the `.cursor/skills/python-style-touchup/` skill for an automated touch-up workflow.
