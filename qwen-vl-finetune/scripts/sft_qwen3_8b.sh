#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

if [[ "${QWEN_RESEARCH_DRY_RUN:-0}" == "1" || "${QWEN_RESEARCH_DRY_RUN:-}" == "true" ]]; then
  echo "DRY_RUN: qwen3-vl launcher smoke"
  if [[ -n "${OUTPUT_DIR:-}" ]]; then
    mkdir -p "${OUTPUT_DIR}"
  fi
  echo "val_loss: 0.0"
  echo "peak_vram_mb: 0.0"
  exit 0
fi

# Activate the spack env that holds torch/transformers/peft/hf_hub (only if not
# already inside a python env). The prior 4B sweep ran under this env; without
# it `python` resolves to /usr/local/bin/python which lacks huggingface_hub.
if [[ -z "${SPACK_ENV:-}" && -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f /data02/home/philip.yang/spack/share/spack/setup-env.sh ]]; then
    set +u
    . /data02/home/philip.yang/spack/share/spack/setup-env.sh
    spack env activate "${SPACK_TRAIN_ENV:-torch-jax-ffmpeg}" || true
    set -u
  fi
fi

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}

# DeepSpeed configuration (optional)
deepspeed=${DEEPSPEED_CONFIG:-""}
if [[ -n "${deepspeed}" && "${deepspeed}" != /* ]]; then
  deepspeed="${ROOT_DIR}/${deepspeed}"
fi

# Model configuration
llm=${MODEL_NAME_OR_PATH:-"Qwen/Qwen3-VL-8B-Instruct"}

# Resolve HF model IDs to a local snapshot path to avoid repeated Hub API calls
# (notably, tokenizers may call `model_info()` on non-local ids, which can hit 429 rate limits).
resolve_model_path() {
  local model_id_or_path="$1"
  if [[ -d "$model_id_or_path" ]]; then
    echo "$model_id_or_path"
    return 0
  fi
  MODEL_ID="$model_id_or_path" python - <<'PY'
import os, re, sys, time
from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
cache_dir = os.environ.get("HF_HUB_CACHE") or None

def try_snapshot(local_only: bool):
    return snapshot_download(
        repo_id=model_id,
        repo_type="model",
        cache_dir=cache_dir,
        local_files_only=local_only,
        token=os.environ.get("HF_TOKEN") or None,
    )

try:
    p = try_snapshot(local_only=True)
    print(p)
    sys.exit(0)
except Exception:
    pass

max_tries = int(os.environ.get("HF_SNAPSHOT_MAX_TRIES", "5"))
for i in range(1, max_tries + 1):
    try:
        p = try_snapshot(local_only=False)
        print(p)
        sys.exit(0)
    except Exception as e:
        s = str(e)
        # crude 429 backoff: parse "Retry after <N> seconds" if present
        m = re.search(r"Retry after (\d+) seconds", s)
        if m:
            wait_s = int(m.group(1)) + 5
        else:
            wait_s = min(60 * i, 300)
        if i == max_tries:
            raise
        time.sleep(wait_s)
PY
}

# Training hyperparameters
lr=${LR:-1e-5}
batch_size=${BATCH_SIZE:-2}
grad_accum_steps=${GRAD_ACCUM_STEPS:-4}
max_steps=${MAX_STEPS:-5000}
eval_steps=${EVAL_STEPS:-500}
save_steps=${SAVE_STEPS:-1000}

weight_decay=${WEIGHT_DECAY:-0}
warmup_ratio=${WARMUP_RATIO:-0.03}
lr_scheduler_type=${LR_SCHEDULER_TYPE:-"cosine"}

max_pixels=${MAX_PIXELS:-50176}
min_pixels=${MIN_PIXELS:-784}
model_max_length=${MODEL_MAX_LENGTH:-8192}
gradient_checkpointing=${GRADIENT_CHECKPOINTING:-True}

tune_mm_vision=${TUNE_MM_VISION:-False}
tune_mm_mlp=${TUNE_MM_MLP:-True}
tune_mm_llm=${TUNE_MM_LLM:-True}

lora_enable=${LORA_ENABLE:-False}
lora_r=${LORA_R:-16}
lora_alpha=${LORA_ALPHA:-32}
lora_dropout=${LORA_DROPOUT:-0.0}

# Training entry point
entry_file="${ROOT_DIR}/qwenvl/train/train_qwen.py"

# Dataset configuration (must be registered in qwenvl/data/__init__.py)
datasets=${DATASETS:-"visualwebinstruct_train"}
eval_datasets=${EVAL_DATASETS:-"visualwebinstruct_val"}

# Output configuration
run_name=${RUN_NAME:-"qwen3vl_8b"}
output_dir=${OUTPUT_DIR:-"${ROOT_DIR}/output_8b"}

# CUDA toolkit runtime libraries (required for nvrtc on this machine)
CUDA_HOME=${CUDA_HOME:-"/data02/home/philip.yang/spack/opt/spack/linux-icelake/cuda-12.9.1-wctiaiinsx6ez7f6ynrhmbgqspgp3kzd"}
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export HF_HOME=${HF_HOME:-"/data02/home/philip.yang/hf_home"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$HF_HOME"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$HF_HOME/datasets"}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-"false"}

# Ensure a stable hub cache location for snapshot_download
export HF_HUB_CACHE=${HF_HUB_CACHE:-"$HF_HOME/hub"}

# Prefetch/resolve model once (single process) then pass local path to torchrun.
if [[ "$llm" != /* && "$llm" == */* ]]; then
  resolved_model_path="$(resolve_model_path "$llm")"
  if [[ -z "$resolved_model_path" || ! -d "$resolved_model_path" ]]; then
    echo "ERROR: failed to resolve model '$llm' to a local snapshot." >&2
    echo "- Provide a local directory via MODEL_NAME_OR_PATH, OR" >&2
    echo "- Set HF_TOKEN and rerun so the snapshot can be downloaded." >&2
    exit 2
  fi
  llm="$resolved_model_path"
fi

# If we are using a resolved local snapshot, force offline mode during the run to
# prevent any accidental Hub API calls (which can hit 429 rate limits).
if [[ -d "$llm" ]]; then
  export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
  export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
fi

# NCCL stability/debug (single-node DDP)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}
export NCCL_DEBUG=${NCCL_DEBUG:-"WARN"}

dataloader_workers=${DATALOADER_WORKERS:-4}

# Training arguments
args="
    --do_train \
    --do_eval \
    ${deepspeed:+--deepspeed ${deepspeed}} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --dataset_eval_use ${eval_datasets} \
    --data_flatten False \
    --tune_mm_vision ${tune_mm_vision} \
    --tune_mm_mlp ${tune_mm_mlp} \
    --tune_mm_llm ${tune_mm_llm} \
    --lora_enable ${lora_enable} \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_dropout ${lora_dropout} \
    --bf16 \
    --output_dir ${output_dir} \
    --max_steps ${max_steps} \
    --eval_strategy "steps" \
    --eval_steps ${eval_steps} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --min_pixels ${min_pixels} \
    --save_strategy "steps" \
    --save_steps ${save_steps} \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay ${weight_decay} \
    --warmup_ratio ${warmup_ratio} \
    --max_grad_norm 1 \
    --lr_scheduler_type ${lr_scheduler_type} \
    --logging_steps 10 \
    --model_max_length ${model_max_length} \
    --gradient_checkpointing ${gradient_checkpointing} \
    --dataloader_num_workers ${dataloader_workers} \
    --run_name ${run_name} \
    --report_to none \
    --seed 42"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
