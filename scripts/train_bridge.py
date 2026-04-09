"""
train_bridge.py - QLoRA training bridge for Lavely UI.

Spawns a training run using Hugging Face Trainer + PEFT QLoRA, streaming
newline-delimited JSON events to stdout so the Electron UI can render progress.

Event types:
  {"type": "status",   "message": "..."}
  {"type": "config",   "model": "...", "dataset": "...", "output": "...", "total_steps": N}
  {"type": "step",     "step": N, "total": N, "loss": 0.xx, "lr": 0.xx, "epoch": 0.xx}
  {"type": "epoch",    "epoch": N, "total": N}
  {"type": "log",      "text": "..."}
  {"type": "done",     "output": "..."}
  {"type": "error",    "message": "..."}

Usage (positional args, JSON config via --config):
  python train_bridge.py --config '{"model_path": "...", "dataset_path": "...", ...}'

Config fields (all optional except model_path and dataset_path):
  model_path        - base model directory
  dataset_path      - JSONL file (chat messages format) or HF dataset id
  output_dir        - where to save adapter (default: models/lavely-lm-lora)
  epochs            - int (default 3)
  batch_size        - per-device batch (default 1)
  grad_accum        - gradient accumulation steps (default 8)
  lr                - learning rate (default 2e-4)
  lora_r            - LoRA rank (default 16)
  lora_alpha        - LoRA alpha (default 32)
  lora_dropout      - (default 0.05)
  max_seq_len       - (default 2048)
  save_steps        - save every N steps (default: save per epoch)
  logging_steps     - log every N steps (default 5)
  use_4bit          - bool (default True)
"""

import argparse
import io
import json
import os
import sys
import traceback
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON config string")
    args = parser.parse_args()

    try:
        cfg = json.loads(args.config)
    except json.JSONDecodeError as e:
        emit({"type": "error", "message": f"Invalid JSON config: {e}"})
        sys.exit(1)

    model_path = cfg.get("model_path")
    dataset_path = cfg.get("dataset_path")
    if not model_path or not dataset_path:
        emit({"type": "error", "message": "model_path and dataset_path are required"})
        sys.exit(1)

    output_dir = cfg.get("output_dir", "models/lavely-lm-lora")
    epochs = int(cfg.get("epochs", 3))
    batch_size = int(cfg.get("batch_size", 1))
    grad_accum = int(cfg.get("grad_accum", 8))
    lr = float(cfg.get("lr", 2e-4))
    lora_r = int(cfg.get("lora_r", 16))
    lora_alpha = int(cfg.get("lora_alpha", 32))
    lora_dropout = float(cfg.get("lora_dropout", 0.05))
    max_seq_len = int(cfg.get("max_seq_len", 2048))
    save_steps = cfg.get("save_steps")  # optional
    logging_steps = int(cfg.get("logging_steps", 5))
    use_4bit = bool(cfg.get("use_4bit", True))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    emit({"type": "status", "message": "Importing libraries..."})
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import load_dataset
    except Exception as e:
        emit({"type": "error", "message": f"Import failed: {e}"})
        sys.exit(1)

    emit({"type": "status", "message": f"Loading tokenizer from {model_path}..."})
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        emit({"type": "error", "message": f"Failed to load tokenizer: {e}"})
        sys.exit(1)

    emit({"type": "status", "message": "Loading base model (this may take a moment)..."})
    try:
        load_kwargs = {"device_map": "auto", "trust_remote_code": True}
        if use_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if use_4bit:
            model = prepare_model_for_kbit_training(model)
    except Exception as e:
        emit({"type": "error", "message": f"Failed to load model: {e}\n{traceback.format_exc()}"})
        sys.exit(1)

    emit({"type": "status", "message": "Attaching LoRA adapters..."})
    try:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        trainable, total = 0, 0
        for _, p in model.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        emit({
            "type": "log",
            "text": f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)",
        })
    except Exception as e:
        emit({"type": "error", "message": f"LoRA setup failed: {e}"})
        sys.exit(1)

    emit({"type": "status", "message": f"Loading dataset from {dataset_path}..."})
    try:
        if os.path.isfile(dataset_path):
            dataset = load_dataset("json", data_files=dataset_path, split="train")
        else:
            dataset = load_dataset(dataset_path, split="train")
    except Exception as e:
        emit({"type": "error", "message": f"Failed to load dataset: {e}"})
        sys.exit(1)

    def tokenize(example: dict) -> dict:
        if "messages" in example:
            text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        elif "text" in example:
            text = example["text"]
        else:
            # best-effort: concatenate all string fields
            text = " ".join(str(v) for v in example.values() if isinstance(v, str))
        out = tokenizer(
            text, truncation=True, max_length=max_seq_len, padding="max_length"
        )
        out["labels"] = out["input_ids"].copy()
        return out

    try:
        dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
    except Exception as e:
        emit({"type": "error", "message": f"Dataset tokenization failed: {e}"})
        sys.exit(1)

    # Estimate total steps
    steps_per_epoch = max(1, len(dataset) // (batch_size * grad_accum))
    total_steps = steps_per_epoch * epochs

    emit({
        "type": "config",
        "model": model_path,
        "dataset": dataset_path,
        "output": output_dir,
        "total_steps": total_steps,
        "epochs": epochs,
        "samples": len(dataset),
    })

    # ── Progress callback ─────────────────────────────────────────────────
    class StreamCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            payload = {
                "type": "step",
                "step": state.global_step,
                "total": state.max_steps or total_steps,
                "epoch": state.epoch or 0.0,
            }
            if "loss" in logs:
                payload["loss"] = logs["loss"]
            if "learning_rate" in logs:
                payload["lr"] = logs["learning_rate"]
            emit(payload)

        def on_epoch_begin(self, args, state, control, **kwargs):
            emit({
                "type": "epoch",
                "epoch": int(state.epoch or 0) + 1,
                "total": int(args.num_train_epochs),
            })

        def on_train_end(self, args, state, control, **kwargs):
            emit({"type": "log", "text": "Training finished, saving adapter..."})

    # ── Training arguments ────────────────────────────────────────────────
    train_args_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        fp16=not use_4bit,
        bf16=False,
        logging_steps=logging_steps,
        save_strategy="epoch" if not save_steps else "steps",
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        report_to=[],
        disable_tqdm=True,
        gradient_checkpointing=True,
    )
    if save_steps:
        train_args_kwargs["save_steps"] = int(save_steps)

    try:
        training_args = TrainingArguments(**train_args_kwargs)
    except Exception as e:
        emit({"type": "error", "message": f"TrainingArguments failed: {e}"})
        sys.exit(1)

    emit({"type": "status", "message": "Starting training..."})
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            callbacks=[StreamCallback()],
        )
        trainer.train()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        emit({"type": "error", "message": f"Training failed: {e}\n{traceback.format_exc()}"})
        sys.exit(1)

    emit({"type": "done", "output": output_dir})


if __name__ == "__main__":
    main()
