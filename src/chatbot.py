"""
Main chatbot module for AI Mental Health Chatbot
"""

import os
from typing import Optional, List, Dict

from .emotion_detection import predict_emotion
from .sentiment_analysis import predict_sentiment


class MentalHealthChatbot:
    """Main chatbot class for mental health support"""
    
    def __init__(self, model_name: str = "local-rule-based"):
        """
        Initialize the chatbot.
        
        Args:
            model_name: Name of the LLM model to use (future expansion)
        """
        self.model_name = model_name
        self.conversation_history: List[Dict[str, str]] = []
    
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

        # If emotion not recognized
        reply = replies.get(emotion, replies["neutral"])

        # Modify tone based on sentiment
        if sentiment == "negative" and emotion not in ["sadness", "fear"]:
            reply = reply + " It's okay to take a moment if things feel overwhelming."

        return reply
    
    def get_response(self, user_input: str) -> Dict[str, str]:
        """
        Process user input and return an empathetic response.
        
        Args:
            user_input: User's message
            
        Returns:
            Dictionary containing emotion, sentiment, and chatbot reply.
        """
        
        # 1. Detect emotion & sentiment
        emotion = predict_emotion(user_input)
        sentiment = predict_sentiment(user_input)
        
        # 2. Generate a supportive reply
        reply = self.generate_supportive_reply(emotion, sentiment)

        # 3. Save to conversation history
        self.conversation_history.append({
            "user": user_input,
            "emotion": emotion,
            "sentiment": sentiment,
            "bot": reply
        })
        
        return {
            "emotion": emotion,
            "sentiment": sentiment,
            "reply": reply
        }
    
    def reset_conversation(self) -> None:
        """Reset the conversation history"""
        self.conversation_history = []
