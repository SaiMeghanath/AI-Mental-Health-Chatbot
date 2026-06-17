# src/config.py
# Central configuration for the Mental Health Chatbot

# --- Model Configuration ---
EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

# Emotion labels (j-hartmann model output order)
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Sentiment labels (cardiffnlp model output order)
SENTIMENT_LABELS = ["negative", "neutral", "positive"]

# --- Crisis Detection ---
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "self harm", "self-harm",
    "hurt myself", "die", "dont want to live", "don't want to live",
    "cut myself", "overdose", "no reason to live", "can't go on",
    "want to disappear", "end it all", "not worth living", "ending my life",
    "take my own life", "harming myself", "worthless", "hopeless",
    "no way out", "give up on life", "want to die", "better off dead",
    "can't take it anymore", "nothing to live for", "too much pain",
    "want it to stop", "make it stop", "can't do this anymore",
]

# Confidence threshold for neutral sentiment assignment (SST-2 remapping)
SENTIMENT_NEUTRAL_THRESHOLD = 0.55

# --- Logging ---
LOG_FILE = "data/conversation_logs.txt"

# --- Evaluation ---
GOEMOTIONS_SUPPORTED_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
EVAL_RESULTS_PATH = "data/evaluation_results.json"
