from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_analysis(root: Path, profile: str) -> list[dict[str, Any]]:
    analyses: list[dict[str, Any]] = []
    base = root / "experiments" / "runs" / profile
    for path in sorted(base.glob("*/*/attempt_*/analysis.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") == 1:
            analyses.append(payload)
    return analyses


def summarize_campaign(root: Path, *, profile: str) -> str:
    analyses = _load_analysis(root, profile)
    ok = [analysis for analysis in analyses if analysis.get("status") == "ok"]
    crashes = [analysis for analysis in analyses if analysis.get("status") != "ok"]
    fastest = sorted(
        ok,
        key=lambda analysis: analysis.get("metrics", {}).get(
            "throughput_steps_per_sec",
        )
        or 0.0,
        reverse=True,
    )
    best_val = sorted(
        ok,
        key=lambda analysis: analysis.get("metrics", {}).get("val_loss")
        if analysis.get("metrics", {}).get("val_loss") is not None
        else float("inf"),
    )

    lines = ["# Qwen3-VL Research Summary", "", f"Profile: `{profile}`", ""]
    lines.append("## Fastest Safe Configuration")
    if fastest:
        first = fastest[0]
        lines.append(
            f"- {first['trial_id']} attempt={first['attempt']} "
            f"throughput={first['metrics'].get('throughput_steps_per_sec')}"
        )
    else:
        lines.append("No successful analysis artifacts found.")
    lines.append("")
    lines.append("## Best Validation Configuration")
    if best_val:
        first = best_val[0]
        lines.append(
            f"- {first['trial_id']} attempt={first['attempt']} "
            f"val_loss={first['metrics'].get('val_loss')}"
        )
    else:
        lines.append("No successful analysis artifacts found.")
    lines.append("")
    lines.append("## Crashes")
    if crashes:
        for analysis in crashes:
            lines.append(
                f"- {analysis['trial_id']} attempt={analysis['attempt']} "
                f"root_cause={analysis.get('root_cause')}"
            )
    else:
        lines.append("No crashes recorded.")
    return "\n".join(lines) + "\n"
