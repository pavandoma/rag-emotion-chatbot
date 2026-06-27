"""LoRA fine-tune of DistilBERT for 6-class emotion classification on
dair-ai/emotion.

This is the project's "real fine-tuning" deliverable: a genuine LoRA/PEFT
adapter trained on emotion-specific data, runnable on a laptop CPU in a few
minutes. It is NOT a fine-tune of Llama itself -- Ollama serves quantized
GGUF weights and exposes no training API, so literal Llama fine-tuning
isn't practical here. See README "Honest Scope" for the full explanation.
"""

import argparse

import numpy as np
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.config import EMOTION_ADAPTER_DIR, EMOTION_BASE_MODEL, EMOTION_LABELS


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main(epochs: int = 3) -> None:
    dataset = load_dataset("dair-ai/emotion")
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_BASE_MODEL)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=64)

    tokenized = dataset.map(tokenize, batched=True)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        EMOTION_BASE_MODEL, num_labels=len(EMOTION_LABELS)
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(EMOTION_ADAPTER_DIR / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-4,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        report_to=[],
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_metrics = trainer.evaluate(tokenized["test"])
    print(f"Held-out test metrics (LoRA fine-tuned): {test_metrics}")

    EMOTION_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(EMOTION_ADAPTER_DIR))
    tokenizer.save_pretrained(str(EMOTION_ADAPTER_DIR))
    print(f"Saved LoRA adapter to {EMOTION_ADAPTER_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    main(epochs=args.epochs)
