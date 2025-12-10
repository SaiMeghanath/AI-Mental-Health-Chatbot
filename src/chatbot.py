"""Main chatbot module for AI Mental Health Chatbot"""

import os
from typing import Optional, List, Dict


class MentalHealthChatbot:
    """Main chatbot class for mental health support"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize the chatbot with a model name
        
        Args:
            model_name: Name of the LLM model to use
        """
        self.model_name = model_name
        self.conversation_history: List[Dict[str, str]] = []
    
    def get_response(self, user_input: str) -> str:
        """Get a response from the chatbot
        
        Args:
            user_input: User's message
            
        Returns:
            Chatbot's response
        """
        # TODO: Implement actual LLM integration
        return f"Thank you for sharing. I'm here to help."
    
    def reset_conversation(self) -> None:
        """Reset the conversation history"""
        self.conversation_history = []
