#!/usr/bin/env python3
"""VisualWebInstruct JSONL → Unsloth chat `messages` samples (plain Python list).

Each row is expected to have:
  - `conversations`: list of {from: "human"|"gpt", value: str}
  - `image`: relative path (str) or list of relative paths to images under `--data-root`

`<image>` placeholders in human `value` strings are replaced in order with
PIL.Image objects. Rows where the number of `<image>` tokens does not match
the number of image paths are dropped (logged to stderr).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image


def _normalize_image_paths(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(p) for p in raw]
    raise TypeError(f"'image' must be str or list, got {type(raw)}")


def _load_pil(root: Path, rel: str) -> Image.Image:
    path = (root / rel).resolve()
    return Image.open(path).convert("RGB")


def _build_user_content(text: str, images: list[Image.Image]) -> list[dict[str, Any]]:
    n = text.count("<image>")
    if n != len(images):
        return []  # caller handles mismatch
    parts = text.split("<image>")
    content: list[dict[str, Any]] = []
    for i, img in enumerate(images):
        seg = parts[i]
        if seg:
            content.append({"type": "text", "text": seg})
        content.append({"type": "image", "image": img})
    tail = parts[-1]
    if tail:
        content.append({"type": "text", "text": tail})
    return content


def vwi_row_to_messages(
    row: dict[str, Any],
    data_root: Path,
    *,
    drop_reason: list[str] | None = None,
) -> dict[str, Any] | None:
    """Return ``{'messages': [...]}`` or None if the row must be dropped."""
    convs = row.get("conversations")
    if not isinstance(convs, list) or not convs:
        if drop_reason is not None:
            drop_reason.append("missing_or_empty_conversations")
        return None

    rels = _normalize_image_paths(row.get("image"))
    try:
        pil_images = [_load_pil(data_root, r) for r in rels]
    except (FileNotFoundError, OSError) as exc:
        if drop_reason is not None:
            drop_reason.append(f"image_load_error:{exc}")
        return None

    messages: list[dict[str, Any]] = []
    img_cursor = 0

    for turn in convs:
        if not isinstance(turn, dict):
            if drop_reason is not None:
                drop_reason.append("bad_turn_not_dict")
            return None
        who = turn.get("from")
        value = turn.get("value")
        if who not in ("human", "gpt") or not isinstance(value, str):
            if drop_reason is not None:
                drop_reason.append(f"bad_turn_from_or_value:{who!r}")
            return None

        if who == "human":
            k = value.count("<image>")
            slice_imgs = pil_images[img_cursor : img_cursor + k]
            if len(slice_imgs) != k:
                if drop_reason is not None:
                    drop_reason.append("image_cursor_overflow")
                return None
            user_content = _build_user_content(value, slice_imgs)
            if not user_content:
                if drop_reason is not None:
                    drop_reason.append("empty_user_content")
                return None
            img_cursor += k
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": value}],
                }
            )

    if img_cursor != len(pil_images):
        if drop_reason is not None:
            drop_reason.append(
                f"unused_images:{img_cursor}!={len(pil_images)}"
            )
        return None

    return {"messages": messages}


def load_vwi_unsloth_list(
    jsonl_path: str | Path,
    data_root: str | Path,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of ``{'messages': [...]}`` dicts (not a HF Dataset)."""
    path = Path(jsonl_path)
    root = Path(data_root)
    out: list[dict[str, Any]] = []
    skipped = 0

    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and len(out) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            reason: list[str] = []
            sample = vwi_row_to_messages(row, root, drop_reason=reason)
            if sample is None:
                skipped += 1
                print(
                    f"[vwi_to_unsloth] skip line {i+1}: {','.join(reason)}",
                    file=sys.stderr,
                )
                continue

            out.append(sample)

    if skipped:
        print(
            f"[vwi_to_unsloth] kept={len(out)} skipped={skipped} path={path}",
            file=sys.stderr,
        )
    return out


def _summarize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """JSON-friendly view of one converted sample (no raw pixel data)."""
    msgs = sample.get("messages", [])
    sm: list[dict[str, Any]] = []
    for m in msgs:
        role = m.get("role")
        content = m.get("content")
        blocks: list[dict[str, Any]] = []
        if isinstance(content, list):
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "text":
                    t = c.get("text", "")
                    prev = t[:120] + ("…" if len(t) > 120 else "")
                    blocks.append({"type": "text", "text_preview": prev, "len": len(t)})
                elif c.get("type") == "image":
                    im = c.get("image")
                    if isinstance(im, Image.Image):
                        blocks.append(
                            {
                                "type": "image",
                                "pil": f"PIL.Image(mode={im.mode}, size={im.size})",
                            }
                        )
                    else:
                        blocks.append({"type": "image", "pil": repr(im)})
        sm.append({"role": role, "content": blocks})
    return {"messages": sm}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--limit", type=int, default=1, help="max rows to convert (default 1)")
    args = p.parse_args()

    rows = load_vwi_unsloth_list(args.jsonl, args.data_root, limit=args.limit)
    if not rows:
        print("No rows converted; see stderr for skip reasons.", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(_summarize_sample(rows[0]), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
