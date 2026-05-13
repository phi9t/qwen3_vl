#!/usr/bin/env bash

research_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

finetune_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

require_8b_launcher() {
  local root
  root="$(finetune_root)"
  if [[ ! -x "$root/scripts/sft_qwen3_8b.sh" && ! -f "$root/scripts/sft_qwen3_8b.sh" ]]; then
    echo "Missing launcher: $root/scripts/sft_qwen3_8b.sh" >&2
    return 1
  fi
}
