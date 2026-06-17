# src/response_generator.py
# Rule-based empathetic response generation keyed on (emotion, sentiment) pairs

from typing import Tuple

# Response lookup table: key = (emotion, sentiment), value = response string
_RESPONSE_TABLE = {
    ("sadness", "negative"): (
        "I'm really sorry you're feeling this way. It takes courage to share something so heavy. "
        "Would you like to tell me more about what's been weighing on you?"
    ),
    ("sadness", "neutral"): (
        "It sounds like things have been difficult. I'm here to listen — take your time."
    ),
    ("sadness", "positive"): (
        "Even when things feel hard, there's something in you that keeps going. "
        "I'm here if you'd like to talk through it."
    ),
    ("anger", "negative"): (
        "It's completely okay to feel angry. Something must have really gotten to you. "
        "Want to share what happened?"
    ),
    ("anger", "neutral"): (
        "I can sense some frustration in what you've shared. I'm here to listen without judgment."
    ),
    ("anger", "positive"): (
        "Anger can sometimes come from caring deeply about something. "
        "I'm here if you'd like to talk it through."
    ),
    ("fear", "negative"): (
        "That sounds really frightening. Fear can feel overwhelming, but you're not alone. "
        "Can you tell me more about what's worrying you?"
    ),
    ("fear", "neutral"): (
        "It sounds like something has you unsettled. Whatever it is, you can share it here safely."
    ),
    ("fear", "positive"): (
        "Sometimes facing fears takes a lot of strength. I'm here to listen."
    ),
    ("joy", "positive"): (
        "That's wonderful to hear! It sounds like something really good is happening for you. "
        "I'd love to hear more!"
    ),
    ("joy", "neutral"): (
        "It sounds like there's something positive going on — tell me more!"
    ),
    ("joy", "negative"): (
        "Sometimes good and difficult feelings can coexist. I'm here for whichever you'd like to talk about."
    ),
    ("disgust", "negative"): (
        "That sounds really uncomfortable or upsetting. It's okay to feel that way. "
        "I'm here if you want to talk about it."
    ),
    ("disgust", "neutral"): (
        "Something seems to have bothered you. You can share as much or as little as you'd like."
    ),
    ("disgust", "positive"): (
        "Even when something feels off, there's often something worth holding onto. "
        "I'm here to listen."
    ),
    ("surprise", "positive"): (
        "That sounds unexpected in a good way! How are you feeling about it?"
    ),
    ("surprise", "neutral"): (
        "Sounds like something caught you off guard. How are you processing it?"
    ),
    ("surprise", "negative"): (
        "Unexpected things can be really jarring. I'm here if you'd like to talk through how you're feeling."
    ),
    ("neutral", "neutral"): (
        "I'm here with you. Feel free to share whatever is on your mind."
    ),
    ("neutral", "negative"): (
        "It sounds like something might be bothering you. Take your time — I'm listening."
    ),
    ("neutral", "positive"): (
        "It's good to hear from you. What would you like to talk about today?"
    ),
}

_DEFAULT_RESPONSE = (
    "I'm here for you. Whatever you're feeling right now, you can share it with me."
)


def generate_response(emotion: str, sentiment: str) -> str:
    """
    Generate an empathetic response based on detected emotion and sentiment.

    Args:
        emotion: Emotion label from emotion_classifier.
        sentiment: Sentiment label from sentiment_classifier.

    Returns:
        Empathetic response string.
    """
    key: Tuple[str, str] = (emotion, sentiment)
    return _RESPONSE_TABLE.get(key, _DEFAULT_RESPONSE)
