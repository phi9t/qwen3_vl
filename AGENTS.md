# Repository Instructions

- Use `uv` to install and run Python dependencies for this repository.
- Do not add conditional imports or dependency-skipping test paths for missing Python packages; install the dependency with `uv` instead.
- In Python code, keep imports at module top level. Do not add imports inside functions, do not wrap imports in `try`/`except`, and treat import failures as missing dependencies to fix with `uv`.
