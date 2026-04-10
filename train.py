from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from requests.exceptions import ChunkedEncodingError
from torch.utils.data import DataLoader
import torch
import re
import time

# Load dataset
dataset = load_dataset("openai/gsm8k", "main")

# CPU-friendly defaults so the script finishes in reasonable time.
train_data = dataset["train"].select(range(300))
test_data = dataset["test"].select(range(100))

model_name = "distilgpt2"  # lightweight substitute


def load_with_retry(fn, *args, retries=3, delay=5, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except ChunkedEncodingError as e:
            if attempt == retries:
                raise
            print(f"Download error ({e}); retrying {attempt}/{retries} in {delay}s...")
            time.sleep(delay)


tokenizer = load_with_retry(AutoTokenizer.from_pretrained, model_name)
model = load_with_retry(AutoModelForCausalLM.from_pretrained, model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_config)

def preprocess(example):
    text = f"Q: {example['question']}\nA: {example['answer']}"
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=256)
    enc["labels"] = enc["input_ids"].copy()
    return enc

train_data = train_data.map(preprocess)
test_data = test_data.map(preprocess)

train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()

for step, batch in enumerate(train_loader, start=1):
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 10 == 0:
        print(f"step={step} loss={loss.item():.4f}")

    # Keep runtime practical for a laptop baseline
    if step >= 50:
        break

def evaluate():
    correct = 0
    sample_count = 0
    model.eval()

    for sample in test_data:
        prompt = f"Q: {sample['question']}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=48)
        prediction = tokenizer.decode(output[0], skip_special_tokens=True)

        pred_num = re.findall(r"-?\d+(?:\.\d+)?", prediction.split("A:")[-1])
        gold_num = re.findall(r"-?\d+(?:\.\d+)?", sample["answer"])
        if pred_num and gold_num and pred_num[-1] == gold_num[-1]:
            correct += 1
        sample_count += 1

    print("Approx numeric-match accuracy:", correct / max(sample_count, 1))

evaluate()