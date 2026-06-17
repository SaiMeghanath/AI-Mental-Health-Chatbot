# src/emotion_classifier.py
# Emotion classification using j-hartmann/emotion-english-distilroberta-base

from __future__ import annotations
from typing import Dict

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from .config import EMOTION_MODEL_NAME, EMOTION_LABELS

# --- Singleton model loading ---
_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_NAME)
        _model.eval()


def predict_emotion(text: str) -> Dict[str, object]:
    """
    Predict the dominant emotion in the given text.

    Args:
        text: Input user message.

    Returns:
        dict with keys:
            label (str): Dominant emotion label.
            score (float): Confidence score (0–1).
            raw_output (list): Full softmax probability vector.
    """
    if not text or not text.strip():
        return {"label": "neutral", "score": 1.0, "raw_output": []}

    _load_model()

    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = _model(**inputs)

    scores = torch.softmax(outputs.logits, dim=1).squeeze().tolist()
    predicted_index = int(torch.argmax(torch.tensor(scores)).item())

    return {
        "label": EMOTION_LABELS[predicted_index],
        "score": round(scores[predicted_index], 4),
        "raw_output": [round(s, 4) for s in scores],
    }
