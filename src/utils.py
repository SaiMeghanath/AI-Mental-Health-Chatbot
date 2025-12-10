"""Utility functions for AI Mental Health Chatbot"""

import logging
from datetime import datetime
from typing import Dict, Any
import json


def setup_logging(log_level: str = 'INFO', log_file: str = None) -> logging.Logger:
    """Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level))
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def format_response(response: str) -> str:
    """Format chatbot response for display
    
    Args:
        response: Raw response text
        
    Returns:
        Formatted response
    """
    return response.strip()


def log_conversation(user_input: str, bot_response: str) -> Dict[str, Any]:
    """Log a conversation exchange
    
    Args:
        user_input: User's message
        bot_response: Bot's response
        
    Returns:
        Conversation log entry
    """
    return {
        'timestamp': datetime.now().isoformat(),
        'user': user_input,
        'assistant': bot_response
    }
