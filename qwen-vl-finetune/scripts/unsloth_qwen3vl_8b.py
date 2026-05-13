#!/usr/bin/env python3
"""
Qwen3-VL-8B SFT via Unsloth — translated from the Colab recipe in §B.1 of the
approved program plan (env pins documented in design_unsloth.md).

Training hyperparameters that Part A may override are exposed via argparse;
all other notebook defaults stay fixed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from vwi_to_unsloth import load_vwi_unsloth_list  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train-jsonl",
        type=Path,
        default=Path(
            "/data02/home/philip.yang/datasets/visualwebinstruct/train.jsonl"
        ),
    )
    p.add_argument(
        "--val-jsonl",
        type=Path,
        default=Path("/data02/home/philip.yang/datasets/visualwebinstruct/val.jsonl"),
        help="Optional validation JSONL; converted rows are passed to SFTTrainer when non-empty.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data02/home/philip.yang/datasets/visualwebinstruct"),
    )
    p.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Max training rows after adapter filtering (None = all lines).",
    )
    p.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Max eval rows (None = all lines); only used when eval is enabled.",
    )
    p.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="When --eval-strategy=steps and validation data is loaded, sets eval_steps.",
    )
    p.add_argument(
        "--eval-strategy",
        type=str,
        default="no",
        choices=["no", "steps", "epoch"],
        help="Trainer eval strategy; 'no' keeps prior behavior (no eval_* keys in SFTConfig).",
    )
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument(
        "--finetune-vision-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Notebook default True; Part A may turn this off.",
    )
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Tutorial default; ignored if --num-train-epochs is set.",
    )
    p.add_argument(
        "--num-train-epochs",
        type=float,
        default=None,
        help="If set, runs epoch-based training (SFTConfig max_steps=-1).",
    )
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="linear",
        help="e.g. linear (notebook) or cosine to mirror Part A.",
    )
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--save-lora-dir", type=str, default="qwen_lora")
    p.add_argument("--save-merged-dir", type=str, default="unsloth_finetune")
    p.add_argument(
        "--model-id",
        type=str,
        default="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
    )
    args = p.parse_args()

    from unsloth import FastVisionModel  # noqa: PLC0415
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415
    from unsloth.trainer import UnslothVisionDataCollator  # noqa: PLC0415

    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_id,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    converted = load_vwi_unsloth_list(
        args.train_jsonl, args.data_root, limit=args.train_limit
    )
    if not converted:
        raise SystemExit(
            "No training samples after VWI adapter; check stderr for skip reasons."
        )

    eval_dataset = None
    if args.val_jsonl.is_file():
        eval_maybe = load_vwi_unsloth_list(
            args.val_jsonl, args.data_root, limit=args.eval_limit
        )
        if eval_maybe:
            eval_dataset = eval_maybe

    sft_kw: dict = dict(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        learning_rate=args.learning_rate,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=3407,
        output_dir=args.output_dir,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=args.max_length,
    )
    if args.num_train_epochs is not None:
        sft_kw["max_steps"] = -1
        sft_kw["num_train_epochs"] = float(args.num_train_epochs)
    else:
        sft_kw["max_steps"] = int(args.max_steps)

    if args.eval_strategy != "no" and eval_dataset is not None:
        sft_kw["eval_strategy"] = args.eval_strategy
        if args.eval_strategy == "steps" and args.eval_steps is not None:
            sft_kw["eval_steps"] = args.eval_steps

    FastVisionModel.for_training(model)
    tr_kw: dict = dict(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=converted,
        args=SFTConfig(**sft_kw),
    )
    if eval_dataset is not None:
        tr_kw["eval_dataset"] = eval_dataset

    trainer = SFTTrainer(**tr_kw)

    trainer_stats = trainer.train()

    model.save_pretrained(args.save_lora_dir)
    tokenizer.save_pretrained(args.save_lora_dir)
    model.save_pretrained_merged(args.save_merged_dir, tokenizer)


if __name__ == "__main__":
    main()
