"""Conversational tone adjustment via prompt-conditioning.

This is explicitly NOT weight-level fine-tuning of the Llama model -- it
detects the user's emotion with the LoRA classifier (emotion_classifier.py)
and injects an instruction into Ollama's system prompt so the response tone
adapts. The classifier is the genuinely fine-tuned component; this layer is
prompt engineering on top of it.
"""

from src.emotion.emotion_classifier import predict_emotion

TONE_INSTRUCTIONS = {
    "sadness": "The user seems to be feeling sad. Respond with warmth and empathy.",
    "joy": "The user seems to be feeling happy. Match their positive energy.",
    "love": "The user's message conveys affection or warmth. Respond warmly in kind.",
    "anger": "The user seems frustrated or angry. Respond calmly and validate their feelings before addressing the question.",
    "fear": "The user seems anxious or worried. Respond reassuringly and clearly.",
    "surprise": "The user seems surprised. Acknowledge that before answering.",
}


def build_emotion_aware_system_prefix(text: str) -> tuple[str, str, float]:
    emotion, confidence = predict_emotion(text)
    instruction = TONE_INSTRUCTIONS.get(emotion, "")
    prefix = f"{instruction} Keep responses natural and human-like." if instruction else ""
    return prefix, emotion, confidence
