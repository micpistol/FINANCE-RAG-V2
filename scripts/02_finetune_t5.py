# scripts/02_finetune_t5.py
# Vanilla T5 instruction fine-tuning (Seq2SeqTrainer) — no PEFT, no extras.

import os
import json
import yaml
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
)

import evaluate


def read_cfg(path: str = "config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def get_cfg_with_defaults():
    cfg = read_cfg()

    # Default hyperparams (override in config.yaml if you want)
    cfg.setdefault("model_name", "t5-small")
    cfg.setdefault("finetuned_dir", "artifacts/t5-small-finetuned")
    cfg.setdefault("max_source_length", 512)
    cfg.setdefault("max_target_length", 128)
    cfg.setdefault("gen_max_len", 128)
    cfg.setdefault("train_batch_size", 8)
    cfg.setdefault("eval_batch_size", 8)
    cfg.setdefault("grad_accum_steps", 1)
    cfg.setdefault("num_train_epochs", 1)
    cfg.setdefault("learning_rate", 5e-5)
    cfg.setdefault("warmup_ratio", 0.05)
    cfg.setdefault("weight_decay", 0.0)
    cfg.setdefault("lr_scheduler_type", "linear")
    cfg.setdefault("seed", 42)
    cfg.setdefault("report_to", "none")  # avoid wandb noise
    cfg.setdefault("fp16", True)         # good on Colab T4
    cfg.setdefault("bf16", False)

    # Data paths (produced by scripts/00_prepare_data.py)
    cfg.setdefault("train_path", "data/finance_instruct_sample_train.jsonl")
    cfg.setdefault("val_path",   "data/finance_instruct_sample_val.jsonl")

    # Make sure output dir exists
    os.makedirs(cfg["finetuned_dir"], exist_ok=True)
    return cfg


def load_jsonl_dataset(train_path: str, val_path: str):
    data_files = {
        "train": train_path,
        "validation": val_path,
    }
    ds = load_dataset("json", data_files=data_files)
    # Expect each line like: {"input": "instruction: ...\ncontext: ...", "target": "..."}
    return ds


def tokenize_function_builder(tok, max_source_length: int, max_target_length: int):
    def _tok_fn(batch):
        # Encode source (input)
        model_inputs = tok(
            batch["input"],
            max_length=max_source_length,
            truncation=True,
            padding=False,
        )
        # Encode target (labels)
        with tok.as_target_tokenizer():
            labels = tok(
                batch["target"],
                max_length=max_target_length,
                truncation=True,
                padding=False,
            )["input_ids"]

        model_inputs["labels"] = labels
        return model_inputs
    return _tok_fn


def build_metrics(tok):
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # Replace -100 with pad_token_id for decoding
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        labels = np.where(labels != -100, labels, tok.pad_token_id)

        pred_str = tok.batch_decode(preds, skip_special_tokens=True)
        label_str = tok.batch_decode(labels, skip_special_tokens=True)

        r = rouge.compute(predictions=pred_str, references=label_str)
        b = bleu.compute(predictions=pred_str, references=[[l] for l in label_str])

        return {
            "rouge1": r["rouge1"],
            "rouge2": r["rouge2"],
            "rougeL": r["rougeL"],
            "bleu": b["score"],
        }

    return compute_metrics


def main():
    cfg = get_cfg_with_defaults()
    set_seed(cfg["seed"])

    print("Loading training data...")
    if not os.path.exists(cfg["train_path"]) or not os.path.exists(cfg["val_path"]):
        raise FileNotFoundError(
            f"Expected train/val at {cfg['train_path']} and {cfg['val_path']}. "
            f"Run scripts/00_prepare_data.py first."
        )

    raw_ds = load_jsonl_dataset(cfg["train_path"], cfg["val_path"])
    print("Train samples:", len(raw_ds["train"]))
    print("Val samples:", len(raw_ds["validation"]))

    print("Loading", cfg["model_name"], "...")
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"])

    print("Preprocessing data...")
    tok_fn = tokenize_function_builder(
        tok,
        max_source_length=cfg["max_source_length"],
        max_target_length=cfg["max_target_length"],
    )
    tokenized = raw_ds.map(tok_fn, batched=True, remove_columns=raw_ds["train"].column_names)

    train_ds = tokenized["train"]
    val_ds = tokenized["validation"]

    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model)
    compute_metrics = build_metrics(tok)

    args = Seq2SeqTrainingArguments(
        output_dir=cfg["finetuned_dir"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=cfg["gen_max_len"],
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        fp16=cfg["fp16"],
        bf16=cfg["bf16"],
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        save_total_limit=2,
        report_to=cfg.get("report_to", "none"),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save final model + tokenizer
    trainer.save_model(cfg["finetuned_dir"])
    tok.save_pretrained(cfg["finetuned_dir"])
    print("Saved fine-tuned model to:", cfg["finetuned_dir"])


if __name__ == "__main__":
    main()
