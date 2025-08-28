import os, json, yaml
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
import torch

CFG = yaml.safe_load(open("config.yaml"))

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    print("Loading training data...")
    train_data = load_jsonl("data/finance_instruct_sample_train.jsonl")
    val_data = load_jsonl("data/finance_instruct_sample_val.jsonl")
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Load tokenizer and model
    print(f"Loading {CFG['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(CFG["model_name"])
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def preprocess_function(examples):
        # Tokenize inputs and targets
        model_inputs = tokenizer(
            examples["input"],
            max_length=CFG["max_source_length"],
            padding="max_length",
            truncation=True,
        )
        
        labels = tokenizer(
            examples["target"],
            max_length=CFG["max_target_length"],
            padding="max_length",
            truncation=True,
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Apply preprocessing
    print("Preprocessing data...")
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(CFG["artifacts_dir"], "t5-small-finetuned"),
        evaluation_strategy="epoch",
        learning_rate=CFG["lr"],
        per_device_train_batch_size=CFG["train_batch_size"],
        per_device_eval_batch_size=CFG["eval_batch_size"],
        weight_decay=CFG["weight_decay"],
        save_total_limit=2,
        num_train_epochs=CFG["epochs"],
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        warmup_ratio=CFG["warmup_ratio"],
        logging_steps=100,
        save_steps=500,
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(os.path.join(CFG["artifacts_dir"], "t5-small-finetuned"))
    
    print("Training complete! Model saved to artifacts/t5-small-finetuned/")

if __name__ == "__main__":
    main()
