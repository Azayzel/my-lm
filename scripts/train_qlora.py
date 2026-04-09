"""QLoRA fine-tuning script for Qwen3.5-2B on 6 GB VRAM."""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

model_id = "../models/qwen3.5-2b"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

# LoRA config — targets the attention layers
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Should be ~0.5-2% of total

# Load dataset
dataset = load_dataset("json", data_files="../datasets/train.jsonl", split="train")


def tokenize(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    out = tokenizer(text, truncation=True, max_length=2048, padding="max_length")
    out["labels"] = out["input_ids"].copy()
    return out


dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# Training
args = TrainingArguments(
    output_dir="../models/lavely-lm-lora",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # keep at 1 for 6 GB
    gradient_accumulation_steps=8,  # effective batch = 8
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
)

trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()
model.save_pretrained("../models/lavely-lm-lora")
