"""QLoRA fine-tuning script for Qwen3.5-2B on 6 GB VRAM.

Usage:
  python scripts/train_qlora.py
  python scripts/train_qlora.py --data datasets/motogp/train.jsonl \
      --output-dir models/My-lm-motogp-lora --epochs 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-id", default="../models/qwen3.5-2b",
                   help="HF model id or local path")
    p.add_argument("--data", type=Path, default=Path("../datasets/train.jsonl"),
                   help="Training JSONL ({\"messages\":[...]} per line)")
    p.add_argument("--output-dir", type=Path, default=Path("../models/My-lm-lora"))
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Per-device batch (keep 1 on 6 GB VRAM)")
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Gradient accumulation steps (effective batch = batch * accum)")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.data.exists():
        raise SystemExit(f"data file not found: {args.data}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(args.data), split="train")

    def tokenize(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        out = tokenizer(text, truncation=True, max_length=args.max_length,
                        padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    targs = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(model=model, args=targs, train_dataset=dataset)
    trainer.train()
    model.save_pretrained(str(args.output_dir))
    print(f"saved adapter to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
