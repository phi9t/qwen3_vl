#!/usr/bin/env bash
# Pre-commit Python style touch-up.
#
# Usage:
#   bash .cursor/skills/python-style-touchup/scripts/touchup.sh [file ...]
#
# With no arguments: operates on staged Python files (git diff --cached).
# If no staged files are found, falls back to all modified Python files in the
# working tree.
# With explicit file arguments: operates on those files directly.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

# ── Collect target files ────────────────────────────────────────────────────

if [[ $# -gt 0 ]]; then
    mapfile -t PY_FILES < <(printf '%s\n' "$@" | grep '\.py$' || true)
else
    mapfile -t PY_FILES < <(
        git diff --cached --name-only --diff-filter=ACM 2>/dev/null \
        | grep '\.py$' || true
    )

    if [[ ${#PY_FILES[@]} -eq 0 ]]; then
        echo "No staged Python files found. Falling back to modified files in working tree."
        mapfile -t PY_FILES < <(
            git diff --name-only --diff-filter=ACM 2>/dev/null \
            | grep '\.py$' || true
        )
    fi
fi

if [[ ${#PY_FILES[@]} -eq 0 ]]; then
    echo "No Python files to process."
    exit 0
fi

echo "Files to process (${#PY_FILES[@]}):"
printf '  %s\n' "${PY_FILES[@]}"
echo

# ── Check ruff is available ──────────────────────────────────────────────────

if ! command -v ruff &>/dev/null; then
    echo "ERROR: ruff not found. Install it:"
    echo "  pip install ruff"
    echo "  # or: uv pip install ruff"
    exit 1
fi

RUFF_VERSION="$(ruff --version 2>&1 | head -1)"
echo "Using: $RUFF_VERSION"
echo

# ── Step 1: Auto-format ──────────────────────────────────────────────────────

echo "=== ruff format (auto-format) ==="
FORMAT_OUTPUT="$(ruff format "${PY_FILES[@]}" 2>&1)" || true
echo "$FORMAT_OUTPUT"
FORMATTED_COUNT="$(echo "$FORMAT_OUTPUT" | grep -c 'reformatted' || echo 0)"
echo

# ── Step 2: Auto-fix lint ────────────────────────────────────────────────────

echo "=== ruff check --fix (auto-fix lint) ==="
FIX_OUTPUT="$(ruff check --fix "${PY_FILES[@]}" 2>&1)" || true
echo "$FIX_OUTPUT"
FIXED_COUNT="$(echo "$FIX_OUTPUT" | grep -c 'Fixed' || echo 0)"
echo

# ── Step 3: Report remaining issues ─────────────────────────────────────────

echo "=== ruff check (remaining issues) ==="
REMAINING_OUTPUT="$(ruff check "${PY_FILES[@]}" 2>&1)" || true

if echo "$REMAINING_OUTPUT" | grep -q "^.*\.py:"; then
    echo "$REMAINING_OUTPUT"
    REMAINING_COUNT="$(echo "$REMAINING_OUTPUT" | grep -c "^.*\.py:" || echo 0)"
else
    echo "No remaining issues."
    REMAINING_COUNT=0
fi
echo

# ── Summary ──────────────────────────────────────────────────────────────────

echo "=== Summary ==="
echo "  Files processed : ${#PY_FILES[@]}"
echo "  Auto-formatted  : $FORMATTED_COUNT"
echo "  Auto-fixed lint : $FIXED_COUNT"
echo "  Remaining issues: $REMAINING_COUNT"
echo

if [[ "$REMAINING_COUNT" -gt 0 ]]; then
    echo "Remaining issues need manual attention."
    echo "Common fixes:"
    echo "  D100-D107 (missing docstrings)  → add Google-style docstrings"
    echo "  ANN001/ANN201 (missing types)   → annotate parameters and return types"
    echo "  B006 (mutable default arg)      → use None default with guard"
    echo ""
    echo "Style guide: docs/python-style-guide.md"
    exit 1
fi

echo "All issues resolved. Ready to commit."
