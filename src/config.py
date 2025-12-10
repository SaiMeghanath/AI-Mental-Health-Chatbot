"""Configuration settings for AI Mental Health Chatbot"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')

# Application Configuration
APP_NAME = 'AI Mental Health Chatbot'
APP_VERSION = '0.1.0'

# Chat Configuration
MAX_CONVERSATION_LENGTH = 50
RESPONSE_TIMEOUT = 30  # seconds

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'logs/chatbot.log')

# Database Configuration (optional)
DATABASE_URL = os.getenv('DATABASE_URL')

# Feature Flags
ENABLE_CONVERSATION_HISTORY = True
ENABLE_USER_FEEDBACK = True
