from __future__ import annotations

import autonomy.activities
import autonomy.cli
import autonomy.done_gate
import autonomy.executors.base
import autonomy.executors.claude_code
import autonomy.executors.shell
import autonomy.executors.torch_trial
import autonomy.org.mutator
import autonomy.org.parser
import autonomy.org.schema
import autonomy.sandbox
import autonomy.smoke.gpt2_synthetic
import autonomy.worker
import autonomy.workflows
import autonomy.worktree


def test_modules_import() -> None:
    assert autonomy.activities is not None
    assert autonomy.cli is not None
    assert autonomy.done_gate is not None
    assert autonomy.executors.base is not None
    assert autonomy.executors.claude_code is not None
    assert autonomy.executors.shell is not None
    assert autonomy.executors.torch_trial is not None
    assert autonomy.org.mutator is not None
    assert autonomy.org.parser is not None
    assert autonomy.org.schema is not None
    assert autonomy.sandbox is not None
    assert autonomy.smoke.gpt2_synthetic is not None
    assert autonomy.worker is not None
    assert autonomy.workflows is not None
    assert autonomy.worktree is not None
