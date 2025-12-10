# src/sentiment_analysis.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

tokenizer_s = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
model_s = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)

SENTIMENT_LABELS = ['negative', 'neutral', 'positive']


def predict_sentiment(text: str) -> str:
    """
    Predict sentiment (negative / neutral / positive) for the given text.
    """
    if not text or not text.strip():
        return "neutral"

    inputs = tokenizer_s(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model_s(**inputs)

    scores = torch.softmax(outputs.logits, dim=1)
    predicted_index = torch.argmax(scores).item()
    return SENTIMENT_LABELS[predicted_index]
