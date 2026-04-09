"""Qwen3.5-2B inference script for Lavely-LM."""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "../models/qwen3.5-2b"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    # Uncomment below for 4-bit quantization if VRAM is tight:
    # load_in_4bit=True,
)

messages = [{"role": "user", "content": "Hello, introduce yourself as Lavely-LM."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
