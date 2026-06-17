# src/sentiment_classifier.py
# Sentiment analysis using cardiffnlp/twitter-roberta-base-sentiment

from __future__ import annotations
from typing import Dict

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from .config import SENTIMENT_MODEL_NAME, SENTIMENT_LABELS, SENTIMENT_NEUTRAL_THRESHOLD

# --- Singleton model loading ---
_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
        _model.eval()


def predict_sentiment(text: str) -> Dict[str, object]:
    """
    Predict sentiment polarity of the given text.

    Args:
        text: Input user message.

    Returns:
        dict with keys:
            label (str): 'positive', 'neutral', or 'negative'.
            score (float): Confidence score for the predicted label.
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
    predicted_score = scores[predicted_index]

    # Threshold-based neutral remapping for low-confidence predictions
    if predicted_score < SENTIMENT_NEUTRAL_THRESHOLD and predicted_index != 1:
        label = "neutral"
        score = scores[1]
    else:
        label = SENTIMENT_LABELS[predicted_index]
        score = predicted_score

    return {
        "label": label,
        "score": round(score, 4),
        "raw_output": [round(s, 4) for s in scores],
    }
