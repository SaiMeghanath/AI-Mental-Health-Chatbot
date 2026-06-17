# src/chatbot.py
# Core chatbot logic — coordinates emotion, sentiment, crisis detection, and response generation

import os
from typing import Dict, List
from datetime import datetime

from .emotion_classifier import predict_emotion
from .sentiment_classifier import predict_sentiment
from .crisis_detector import is_crisis, CRISIS_RESPONSE
from .response_generator import generate_response
from .config import LOG_FILE


class MentalHealthChatbot:
    """
    AI-based Mental Health Support Chatbot.

    Coordinates the full NLP pipeline:
    1. Crisis keyword detection (highest priority)
    2. Emotion classification
    3. Sentiment analysis
    4. Empathetic response generation
    """

    def __init__(self):
        self.conversation_history: List[Dict] = []
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    def get_response(self, user_input: str) -> Dict:
        """
        Process user input and return a response with emotion and sentiment metadata.

        Args:
            user_input: Raw text message from the user.

        Returns:
            dict with keys: emotion, sentiment, reply, emotion_score, sentiment_score.
        """
        user_input = user_input.strip()
        if not user_input:
            return {
                "emotion": "neutral",
                "emotion_score": 1.0,
                "sentiment": "neutral",
                "sentiment_score": 1.0,
                "reply": "I'm here whenever you're ready to share.",
            }

        # --- Step 1: Crisis check (overrides all other logic) ---
        if is_crisis(user_input):
            result = {
                "emotion": "crisis",
                "emotion_score": None,
                "sentiment": "negative",
                "sentiment_score": None,
                "reply": CRISIS_RESPONSE,
            }
        else:
            # --- Step 2: NLP pipeline ---
            emotion_result = predict_emotion(user_input)
            sentiment_result = predict_sentiment(user_input)

            reply = generate_response(emotion_result["label"], sentiment_result["label"])

            result = {
                "emotion": emotion_result["label"],
                "emotion_score": emotion_result["score"],
                "sentiment": sentiment_result["label"],
                "sentiment_score": sentiment_result["score"],
                "reply": reply,
            }

        # --- Step 3: Log and store ---
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "user": user_input,
            **result,
        }
        self.conversation_history.append(record)
        self._log(record)

        return result

    def _log(self, record: Dict) -> None:
        """Append interaction record to the log file."""
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(str(record) + "\n")
        except Exception:
            pass  # Logging must never crash the app

    def reset_conversation(self) -> None:
        """Clear in-memory conversation history."""
        self.conversation_history = []
