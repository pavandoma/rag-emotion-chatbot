"""Inference wrapper for the LoRA-fine-tuned emotion classifier, plus a
zero-shot baseline (un-adapted base model) used as the "before" side of the
emotion-accuracy eval.
"""

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import EMOTION_ADAPTER_DIR, EMOTION_BASE_MODEL, EMOTION_LABELS

_finetuned_model = None
_finetuned_tokenizer = None
_baseline_model = None
_baseline_tokenizer = None


def _load_finetuned():
    global _finetuned_model, _finetuned_tokenizer
    if _finetuned_model is None:
        base = AutoModelForSequenceClassification.from_pretrained(
            EMOTION_BASE_MODEL, num_labels=len(EMOTION_LABELS)
        )
        _finetuned_model = PeftModel.from_pretrained(base, str(EMOTION_ADAPTER_DIR))
        _finetuned_model.eval()
        _finetuned_tokenizer = AutoTokenizer.from_pretrained(str(EMOTION_ADAPTER_DIR))
    return _finetuned_model, _finetuned_tokenizer


def _load_baseline():
    global _baseline_model, _baseline_tokenizer
    if _baseline_model is None:
        _baseline_model = AutoModelForSequenceClassification.from_pretrained(
            EMOTION_BASE_MODEL, num_labels=len(EMOTION_LABELS)
        )
        _baseline_model.eval()
        _baseline_tokenizer = AutoTokenizer.from_pretrained(EMOTION_BASE_MODEL)
    return _baseline_model, _baseline_tokenizer


def _predict(model, tokenizer, text: str) -> tuple[str, float]:
    inputs = tokenizer(text, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    idx = int(torch.argmax(probs))
    return EMOTION_LABELS[idx], float(probs[idx])


def predict_emotion(text: str) -> tuple[str, float]:
    """Predict emotion using the LoRA fine-tuned classifier (the real,
    measurably-improved model)."""
    model, tokenizer = _load_finetuned()
    return _predict(model, tokenizer, text)


def predict_emotion_baseline(text: str) -> tuple[str, float]:
    """Predict emotion using the un-adapted base model -- the 'before'
    comparison point that the LoRA fine-tune is measured against."""
    model, tokenizer = _load_baseline()
    return _predict(model, tokenizer, text)
