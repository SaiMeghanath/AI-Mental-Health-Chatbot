# src/emotion_detection.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Pretrained emotion model from Hugging Face
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Label order for this model
LABELS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']


def predict_emotion(text: str) -> str:
    """
    Predict the dominant emotion in the given text.
    """
    if not text or not text.strip():
        return "neutral"

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    scores = torch.softmax(outputs.logits, dim=1)
    predicted_index = torch.argmax(scores).item()
    return LABELS[predicted_index]
