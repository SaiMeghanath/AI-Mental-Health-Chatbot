"""
Main chatbot module for AI Mental Health Chatbot
"""

import os
from typing import List, Dict
from datetime import datetime

from .emotion_detection import predict_emotion
from .sentiment_analysis import predict_sentiment

# Keywords for crisis / self-harm detection
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "self harm", "self-harm",
    "hurt myself", "die", "dont want to live", "don't want to live",
    "cut myself"
]


class MentalHealthChatbot:
    """Main chatbot class for mental health support"""
    
    def __init__(
        self, 
        model_name: str = "local-rule-based",
        log_file: str = "data/conversation_logs.txt"
    ):
        """
        Initialize the chatbot.
        
        Args:
            model_name: Name of the LLM model to use (future expansion)
            log_file: Path to local log file for saving conversations
        """
        self.model_name = model_name
        self.conversation_history: List[Dict[str, str]] = []
        self.log_file = log_file

        # ensure log directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    # ---------- helper methods ----------

    def _is_crisis(self, text: str) -> bool:
        """Check if the message contains crisis / self-harm indicators."""
        lowered = text.lower()
        return any(word in lowered for word in CRISIS_KEYWORDS)

    def _log_interaction(self, record: Dict[str, str]) -> None:
        """Append interaction to a simple log file."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(str(record) + "\n")
        except Exception:
            # logging must never crash the app
            pass
    
    def generate_supportive_reply(self, emotion: str, sentiment: str) -> str:
        """
        Rule-based empathetic replies based on detected emotion.
        """
        
        replies = {
            "sadness": "I'm really sorry you're feeling this way. Want to share what’s hurting you?",
            "anger": "It’s okay to feel angry. I’m here to understand—what made you feel this way?",
            "fear": "That sounds frightening. You can talk to me safely.",
            "joy": "That’s wonderful to hear! I’m glad you’re feeling positive.",
            "disgust": "That must feel uncomfortable. I'm here to listen if you want to share more.",
            "surprise": "That sounds unexpected! How are you feeling about it?",
            "neutral": "I'm here with you. Feel free to share anything on your mind."
        }

        reply = replies.get(emotion, replies["neutral"])

        if sentiment == "negative" and emotion not in ["sadness", "fear"]:
            reply += " It's okay to take a moment if things feel overwhelming."

        return reply
    
    def get_response(self, user_input: str) -> Dict[str, str]:
        """
        Process user input and return an empathetic response.
        
        Args:
            user_input: User's message
            
        Returns:
            Dictionary containing emotion, sentiment, and chatbot reply.
        """
        user_input = user_input.strip()

        # 1. Crisis check first
        if self._is_crisis(user_input):
            emotion = "crisis"
            sentiment = "negative"
            reply = (
                "I'm really glad you reached out. I'm only an AI and can't provide emergency help, "
                "but your feelings are important.\n\n"
                "Please consider talking to someone you trust or a licensed mental health professional.\n\n"
                "If you feel you are in immediate danger, contact your local emergency number "
                "or visit the nearest hospital right away."
            )
        else:
            # 2. Detect emotion & sentiment
            emotion = predict_emotion(user_input)
            sentiment = predict_sentiment(user_input)
            # 3. Generate a supportive reply
            reply = self.generate_supportive_reply(emotion, sentiment)

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "user": user_input,
            "emotion": emotion,
            "sentiment": sentiment,
            "bot": reply,
        }

        # Save in memory & log to file
        self.conversation_history.append(record)
        self._log_interaction(record)
        
        return {
            "emotion": emotion,
            "sentiment": sentiment,
            "reply": reply
        }
    
    def reset_conversation(self) -> None:
        """Reset the conversation history"""
        self.conversation_history = []
