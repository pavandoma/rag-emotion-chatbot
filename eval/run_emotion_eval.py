"""Measures the real emotion-classification accuracy improvement of the
LoRA fine-tuned classifier vs the un-adapted base model, on the held-out
dair-ai/emotion test split.
"""

import json
import time

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

from src.config import EMOTION_LABELS, EVAL_RESULTS_DIR
from src.emotion.emotion_classifier import predict_emotion, predict_emotion_baseline


def main(n_samples: int | None = None) -> None:
    test = load_dataset("dair-ai/emotion")["test"]
    if n_samples:
        test = test.shuffle(seed=42).select(range(n_samples))

    baseline_preds, finetuned_preds, labels = [], [], []

    for i, row in enumerate(test):
        text, label_idx = row["text"], row["label"]
        labels.append(label_idx)

        base_label, _ = predict_emotion_baseline(text)
        ft_label, _ = predict_emotion(text)

        baseline_preds.append(EMOTION_LABELS.index(base_label))
        finetuned_preds.append(EMOTION_LABELS.index(ft_label))

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(test)}")

    baseline_acc = accuracy_score(labels, baseline_preds)
    finetuned_acc = accuracy_score(labels, finetuned_preds)
    baseline_f1 = f1_score(labels, baseline_preds, average="macro")
    finetuned_f1 = f1_score(labels, finetuned_preds, average="macro")

    improvement_pct = (
        (finetuned_acc - baseline_acc) / baseline_acc * 100 if baseline_acc > 0 else float("inf")
    )

    summary = {
        "n_samples": len(test),
        "baseline_accuracy": baseline_acc,
        "finetuned_accuracy": finetuned_acc,
        "accuracy_improvement_pct": improvement_pct,
        "baseline_f1_macro": baseline_f1,
        "finetuned_f1_macro": finetuned_f1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "baseline = un-adapted base model with randomly-initialized classification head",
    }

    print("\n=== Emotion Eval Summary ===")
    print(json.dumps(summary, indent=2))

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (EVAL_RESULTS_DIR / "emotion_eval.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
