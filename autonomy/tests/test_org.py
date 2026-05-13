from __future__ import annotations

import fcntl
import threading
import time
from pathlib import Path

import pytest
from autonomy.org.mutator import append_blocker, append_logbook, transition
from autonomy.org.parser import read_tracker
from autonomy.org.schema import KEYWORDS, WORKFLOW_OWNED_PROPS


MINIMAL_TRACKER = """\
#+TITLE: Test Run
#+TODO: TODO(t) READY(r) IN-PROGRESS(i!) BLOCKED(b@/!) AWAITING-GATE(g!) | DONE(d!) WONTFIX(w@/!) FAILED(f@/!)
#+AUTONOMY_RUN_SLUG: test-run
#+AUTONOMY_DEFAULT_EXECUTOR: claude-code
#+AUTONOMY_DEFAULT_GATE: tests+typecheck+gpt2-smoke

* Tasks
** TODO T01 First task
   :PROPERTIES:
   :ID:        T01
   :EXECUTOR:  shell
   :GATE:      none
   :TIMEOUT:   1h
   :GPUS:      0
   :END:

   *Goal.* Do the first thing.

   *Constraints.*
   - do not break tests

   *Acceptance.*
   cmd: echo first

** TODO T02 Second task
   :PROPERTIES:
   :ID:        T02
   :DEPENDS:   T01
   :END:

   *Goal.* Do the second thing.

** TODO T03 Replace bilinear with nearest-neighbor
   :PROPERTIES:
   :ID:        T03
   :DEPENDS:   T01 T02
   :TIMEOUT:   2h
   :END:

   *Goal.* Replace bilinear with nearest-neighbor in vision tower preproc.

   *Acceptance.*
   cmd: pytest -x
"""


def _write_tracker(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "tracker.org"
    p.write_text(content, encoding="utf-8")
    return p


class TestReadTracker:
    def test_roundtrip(self, tmp_path: Path) -> None:
        path = _write_tracker(tmp_path, MINIMAL_TRACKER)
        doc = read_tracker(path)

        assert doc.header["AUTONOMY_RUN_SLUG"] == "test-run"
        assert len(doc.tasks) == 3

        ids = [t.id for t in doc.tasks]
        assert ids == ["T01", "T02", "T03"]

        t03 = doc.tasks[2]
        assert t03.id == "T03"
        assert t03.depends == {"T01", "T02"}
        assert t03.timeout.total_seconds() == 7200
        assert t03.state == "TODO"
        assert t03.position == 2
        assert t03.slug == "test-run"
        assert t03.gpus == 1
        assert t03.executor == "claude-code"
        assert t03.gate == "tests+typecheck+gpt2-smoke"
        assert t03.goal.strip() == "Replace bilinear with nearest-neighbor in vision tower preproc."

    def test_defaults(self, tmp_path: Path) -> None:
        content = """\
#+AUTONOMY_RUN_SLUG: defaults-run

* Tasks
** TODO T01 Simple task
   :PROPERTIES:
   :ID: T01
   :END:
"""
        path = _write_tracker(tmp_path, content)
        doc = read_tracker(path)
        t = doc.tasks[0]
        assert t.executor == "claude-code"
        assert t.gate == "tests+typecheck+gpt2-smoke"
        assert t.timeout.total_seconds() == 6 * 3600
        assert t.gpus == 1
        assert t.depends == frozenset()
        assert t.branch is None
        assert t.goal == ""
        assert t.constraints == []
        assert t.acceptance_cmds == []

    def test_missing_slug_raises(self, tmp_path: Path) -> None:
        content = """\
* Tasks
** TODO T01 Simple task
   :PROPERTIES:
   :ID: T01
   :END:
"""
        path = _write_tracker(tmp_path, content)
        with pytest.raises(ValueError, match="AUTONOMY_RUN_SLUG"):
            read_tracker(path)

    def test_duplicate_id_raises(self, tmp_path: Path) -> None:
        content = """\
#+AUTONOMY_RUN_SLUG: dup

* Tasks
** TODO T01 First
   :PROPERTIES:
   :ID: T01
   :END:
** TODO T01 Second
   :PROPERTIES:
   :ID: T01
   :END:
"""
        path = _write_tracker(tmp_path, content)
        with pytest.raises(ValueError, match="Duplicate"):
            read_tracker(path)

    def test_unknown_state_raises(self, tmp_path: Path) -> None:
        content = """\
#+AUTONOMY_RUN_SLUG: bad-state

* Tasks
** UNKNOWN T01 First
   :PROPERTIES:
   :ID: T01
   :END:
"""
        path = _write_tracker(tmp_path, content)
        with pytest.raises(ValueError, match="Unknown state"):
            read_tracker(path)


class TestTransition:
    def test_state_and_props_updated(self, tmp_path: Path) -> None:
        path = _write_tracker(tmp_path, MINIMAL_TRACKER)
        transition(
            path,
            "T01",
            "IN-PROGRESS",
            {"OWNER": "wf:1", "STARTED": "2026-05-13T14:00:00Z"},
        )
        doc = read_tracker(path)
        t = doc.tasks[0]
        assert t.state == "IN-PROGRESS"
        assert t.id == "T01"

        raw = path.read_text(encoding="utf-8")
        assert "IN-PROGRESS T01 First task" in raw
        assert ":OWNER:" in raw and "wf:1" in raw
        assert ":STARTED:" in raw and "2026-05-13T14:00:00Z" in raw

    def test_logbook_created(self, tmp_path: Path) -> None:
        path = _write_tracker(tmp_path, MINIMAL_TRACKER)
        transition(path, "T01", "IN-PROGRESS", {"_note": "claimed by wf:1"})
        raw = path.read_text(encoding="utf-8")
        assert "*** Logbook" in raw
        assert "State \"IN-PROGRESS\" from \"TODO\"" in raw
        assert "claimed by wf:1" in raw


class TestAppendLogbook:
    def test_idempotent_order(self, tmp_path: Path) -> None:
        path = _write_tracker(tmp_path, MINIMAL_TRACKER)
        append_logbook(path, "T01", "- first line")
        append_logbook(path, "T01", "- second line")
        raw = path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        first_idx = next(i for i, l in enumerate(lines) if "first line" in l)
        second_idx = next(i for i, l in enumerate(lines) if "second line" in l)
        assert first_idx < second_idx


class TestAppendBlocker:
    def test_blocker_created(self, tmp_path: Path) -> None:
        path = _write_tracker(tmp_path, MINIMAL_TRACKER)
        append_blocker(path, "T02", "gate fail:gpt2-smoke", "loss did not decrease\nAssertionError")
        raw = path.read_text(encoding="utf-8")
        assert "*** Blocker" in raw
        assert "Reason: gate fail:gpt2-smoke" in raw
        assert "#+begin_src text" in raw
        assert "loss did not decrease" in raw
        assert "AssertionError" in raw
        assert "#+end_src" in raw


class TestConcurrency:
    def test_flock_blocks(self, tmp_path: Path) -> None:
        path = _write_tracker(tmp_path, MINIMAL_TRACKER)
        barrier = threading.Barrier(2)

        def hold_lock() -> None:
            with open(path, "r+", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                barrier.wait()
                time.sleep(0.2)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        t = threading.Thread(target=hold_lock)
        t.start()
        barrier.wait()
        start = time.monotonic()
        transition(path, "T01", "READY", {})
        elapsed = time.monotonic() - start
        t.join()
        assert elapsed >= 0.15, f"expected blocking, but elapsed={elapsed:.3f}s"


class TestSchemaConstants:
    def test_keywords_tuple(self) -> None:
        assert KEYWORDS == (
            "TODO",
            "READY",
            "IN-PROGRESS",
            "BLOCKED",
            "AWAITING-GATE",
            "DONE",
            "WONTFIX",
            "FAILED",
        )

    def test_workflow_owned_props(self) -> None:
        assert WORKFLOW_OWNED_PROPS == (
            "OWNER",
            "STARTED",
            "FINISHED",
            "EXIT_CODE",
            "GATE_RESULT",
            "ARTIFACTS",
            "WORKTREE",
            "PR",
        )
