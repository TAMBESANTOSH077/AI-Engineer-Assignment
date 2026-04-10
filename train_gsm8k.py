from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


LOGGER = logging.getLogger("train_gsm8k")


_ANSWER_AFTER_MARKER_RE = re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)")
_FALLBACK_NUMBER_RE = re.compile(r"([-+]?\d[\d,]*\.?\d*)")


def extract_gsm8k_final_answer(text: str) -> Optional[str]:
    """
    Extract the final numeric answer from a GSM8K 'answer' string.
    GSM8K solutions typically end with a marker: '#### <number>'.
    """
    m = _ANSWER_AFTER_MARKER_RE.search(text)
    if m:
        return m.group(1).replace(",", "").strip()

    # Fallback: last number-like token in the string
    nums = _FALLBACK_NUMBER_RE.findall(text)
    if not nums:
        return None
    return nums[-1].replace(",", "").strip()


def format_train_text(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


def format_eval_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def build_splits(train_n: int = 3000, eval_n: int = 1000) -> Tuple[Dataset, Dataset]:
    ds = load_dataset("openai/gsm8k", "main")
    train_ds = ds["train"].select(range(train_n))
    eval_ds = ds["test"].select(range(eval_n))
    return train_ds, eval_ds


def tokenize_for_causal_lm(
    dataset: Dataset,
    tokenizer: Any,
    max_length: int,
) -> Dataset:
    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        text = format_train_text(ex["question"], ex["answer"])
        out = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    return dataset.map(_map, remove_columns=dataset.column_names)


@dataclass
class AccuracyConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0


@torch.no_grad()
def evaluate_exact_match_accuracy(
    model: torch.nn.Module,
    tokenizer: Any,
    eval_raw: Dataset,
    device: torch.device,
    cfg: AccuracyConfig,
    max_length: int,
) -> float:
    model.eval()

    correct = 0
    total = 0

    for ex in eval_raw:
        q = ex["question"]
        gold = extract_gsm8k_final_answer(ex["answer"])
        if gold is None:
            continue

        prompt = format_eval_prompt(q)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.temperature > 0.0,
            temperature=cfg.temperature if cfg.temperature > 0.0 else None,
            top_p=cfg.top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
        pred = extract_gsm8k_final_answer(decoded)

        if pred is not None and pred == gold:
            correct += 1
        total += 1

    return (correct / total) if total else 0.0


class GSM8KTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        eval_raw: Dataset,
        tokenizer_for_eval: Any,
        acc_cfg: AccuracyConfig,
        prompt_max_length: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._eval_raw = eval_raw
        self._tokenizer_for_eval = tokenizer_for_eval
        self._acc_cfg = acc_cfg
        self._prompt_max_length = prompt_max_length

    def evaluate(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        metrics = super().evaluate(*args, **kwargs)

        device = self.model.device if hasattr(self.model, "device") else torch.device("cpu")
        acc = evaluate_exact_match_accuracy(
            model=self.model,
            tokenizer=self._tokenizer_for_eval,
            eval_raw=self._eval_raw,
            device=device,
            cfg=self._acc_cfg,
            max_length=self._prompt_max_length,
        )
        metrics["eval_exact_match"] = acc

        self.log(metrics)
        LOGGER.info("eval_loss=%.4f eval_exact_match=%.4f", metrics.get("eval_loss", float("nan")), acc)
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--output_dir", type=str, default="./results_llama32_1b_gsm8k_lora")
    parser.add_argument("--train_n", type=int, default=3000)
    parser.add_argument("--eval_n", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--prompt_max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    set_seed(args.seed)

    if os.name == "nt":
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    LOGGER.info("Loading GSM8K dataset…")
    train_raw, eval_raw = build_splits(train_n=args.train_n, eval_n=args.eval_n)

    LOGGER.info("Loading tokenizer/model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    LOGGER.info("Applying LoRA (r=16, alpha=32)…")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    LOGGER.info("Tokenizing…")
    train_tok = tokenize_for_causal_lm(train_raw, tokenizer=tokenizer, max_length=args.max_length)
    eval_tok = tokenize_for_causal_lm(eval_raw, tokenizer=tokenizer, max_length=args.max_length)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
    )

    trainer = GSM8KTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data_collator,
        eval_raw=eval_raw,
        tokenizer_for_eval=tokenizer,
        acc_cfg=AccuracyConfig(),
        prompt_max_length=args.prompt_max_length,
    )

    LOGGER.info("Starting training…")
    trainer.train()

    LOGGER.info("Final evaluation…")
    metrics = trainer.evaluate()
    LOGGER.info("Done. eval_exact_match=%.4f", metrics.get("eval_exact_match", 0.0))


if __name__ == "__main__":
    main()

