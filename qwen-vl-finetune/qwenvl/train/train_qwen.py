# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import json
import os
import logging
import pathlib
import torch
import transformers
import sys
from datetime import datetime, timezone
from pathlib import Path
from transformers import AutoConfig

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def research_event(event: str, **payload):
    path = os.environ.get("RESEARCH_EVENTS_PATH")
    if not path:
        return

    record = {
        "schema_version": 1,
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "trial_id": os.environ.get("RESEARCH_TRIAL_ID", ""),
    }
    record.update(payload)

    event_path = Path(path)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    # The wrapper layout differs across Qwen2/2.5-VL vs Qwen3-VL.
    # - Qwen2/2.5: attributes are on the top-level `model`.
    # - Qwen3: the wrapper exposes `model.model.visual` and `model.model.language_model`.
    vision_root = None
    llm_root = None
    if hasattr(model, "visual"):
        vision_root = model.visual
    elif hasattr(model, "model") and hasattr(model.model, "visual"):
        vision_root = model.model.visual

    if hasattr(model, "language_model"):
        llm_root = model.language_model
    elif hasattr(model, "model") and hasattr(model.model, "language_model"):
        llm_root = model.model.language_model

    if vision_root is None or llm_root is None:
        raise AttributeError(
            f"Unexpected model layout for {model.__class__.__name__}: "
            f"vision_root={vision_root is not None}, llm_root={llm_root is not None}"
        )

    if model_args.tune_mm_vision:
        for _, p in vision_root.named_parameters():
            p.requires_grad = True
    else:
        for _, p in vision_root.named_parameters():
            p.requires_grad = False

    if hasattr(vision_root, "merger") and vision_root.merger is not None:
        if model_args.tune_mm_mlp:
            for _, p in vision_root.merger.named_parameters():
                p.requires_grad = True
        else:
            for _, p in vision_root.merger.named_parameters():
                p.requires_grad = False

    if model_args.tune_mm_llm:
        for _, p in llm_root.named_parameters():
            p.requires_grad = True
        if hasattr(model, "lm_head") and model.lm_head is not None:
            model.lm_head.requires_grad = True
    else:
        for _, p in llm_root.named_parameters():
            p.requires_grad = False
        if hasattr(model, "lm_head") and model.lm_head is not None:
            model.lm_head.requires_grad = False


def train(attn_implementation="eager"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)
    research_event("trainer_started", local_rank=local_rank)

    if "qwen3" in model_args.model_name_or_path.lower():
        # Do NOT infer MoE vs dense from path strings (local snapshot directories often contain 'a'
        # and will be misdetected). Use config.model_type instead.
        cfg = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
        model_type = getattr(cfg, "model_type", "")
        if "moe" in str(model_type).lower():
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    research_event(
        "model_loaded",
        model_class=model.__class__.__name__,
        model_name_or_path=model_args.model_name_or_path,
    )
    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if data_args.data_flatten or data_args.data_packing:
        from trainer import replace_qwen2_vl_attention_class

        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            # Print trainable parameters if the method exists (PEFT provides this).
            if hasattr(model, "model") and hasattr(model.model, "visual") and hasattr(model.model.visual, "print_trainable_parameters"):
                model.model.visual.print_trainable_parameters()
            if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "print_trainable_parameters"):
                model.model.language_model.print_trainable_parameters()
    
    data_module = make_supervised_data_module(
        processor,
        data_args=data_args,
        do_train=training_args.do_train,
        do_eval=training_args.do_eval,
    )
    research_event(
        "data_module_ready",
        do_train=training_args.do_train,
        do_eval=training_args.do_eval,
    )
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if training_args.do_train:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            logging.info("checkpoint found, resume training")
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        trainer.save_state()

    # Eval-only (or explicit eval without training): must run on all ranks.
    eval_metrics = None
    if training_args.do_eval and not training_args.do_train:
        eval_metrics = trainer.evaluate()

    # Greppable summary footer (autoresearch-style)
    # IMPORTANT: Do NOT call `trainer.evaluate()` here in distributed mode. `evaluate()` involves
    # collective operations and must be executed by all ranks; calling it only on rank0 can
    # deadlock and trigger NCCL watchdog timeouts. Instead, reuse the last `eval_loss` already
    # computed during training (when `--eval_strategy steps` is enabled).
    if trainer.is_world_process_zero():
        peak_vram_mb = 0.0
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        val_loss = None
        if eval_metrics is not None and isinstance(eval_metrics, dict) and "eval_loss" in eval_metrics:
            val_loss = eval_metrics["eval_loss"]
        else:
            # Reuse the last eval loss from training-time evaluation.
            try:
                for rec in reversed(getattr(trainer.state, "log_history", []) or []):
                    if isinstance(rec, dict) and "eval_loss" in rec:
                        val_loss = rec["eval_loss"]
                        break
            except Exception:
                val_loss = None

        if val_loss is not None:
            print(f"val_loss: {val_loss}")
        print(f"peak_vram_mb: {peak_vram_mb}")
        research_event(
            "trainer_footer",
            val_loss=val_loss,
            peak_vram_mb=peak_vram_mb,
            world_process_zero=True,
        )

    model.config.use_cache = True

    if training_args.do_train:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="eager")
