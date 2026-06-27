"""Two-layer safety/bias mitigation gate, applied to every generated
response before it reaches the user.

Layer 1: fast keyword/regex denylist (cheap first pass).
Layer 2: unitary/toxic-bert classifier score (catches subtler toxicity the
keyword list misses). A response failing either layer is replaced with a
safe fallback.
"""

import re

from transformers import pipeline

from src.config import SAFETY_TOXICITY_THRESHOLD, TOXIC_BERT_MODEL

DENYLIST_PATTERNS = [
    r"\bkill yourself\b",
    r"\bhow to make a bomb\b",
    r"\bhow to make (a )?weapon\b",
    r"\bracial slur\b",
]

SAFE_FALLBACK = "I'm not able to provide a response to that. Could you rephrase your question?"

_toxicity_classifier = None


def _get_classifier():
    global _toxicity_classifier
    if _toxicity_classifier is None:
        _toxicity_classifier = pipeline("text-classification", model=TOXIC_BERT_MODEL)
    return _toxicity_classifier


def keyword_check(text: str) -> bool:
    """Returns True if text is safe (no denylisted pattern found)."""
    lowered = text.lower()
    return not any(re.search(pattern, lowered) for pattern in DENYLIST_PATTERNS)


def toxicity_score(text: str) -> float:
    classifier = _get_classifier()
    result = classifier(text[:512])[0]
    if result["label"].lower() in ("toxic", "label_1"):
        return float(result["score"])
    return 1.0 - float(result["score"])


def check_safety(text: str) -> tuple[bool, float, str]:
    """Returns (is_safe, toxicity_score, possibly-replaced text)."""
    if not keyword_check(text):
        return False, 1.0, SAFE_FALLBACK

    score = toxicity_score(text)
    if score >= SAFETY_TOXICITY_THRESHOLD:
        return False, score, SAFE_FALLBACK

    return True, score, text
