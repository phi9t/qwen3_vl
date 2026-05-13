# autonomy

Stub scaffolding for the repo-local Symphony implementation described in `docs/superpowers/specs/2026-05-13-autonomy-symphony-design.md`.

Module imports are guarded so import smoke works even when `temporalio` is not installed system-wide: `autonomy.workflows` and `autonomy.activities` fall back to no-op decorator shims at import time, while all stub behavior continues to raise `NotImplementedError` only when called.
