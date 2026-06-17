# src/crisis_detector.py
# Keyword-based crisis and self-harm signal detection

from .config import CRISIS_KEYWORDS


def is_crisis(text: str) -> bool:
    """
    Check whether user input contains crisis or self-harm signals.

    Args:
        text: Raw user input string.

    Returns:
        True if any crisis keyword is detected, False otherwise.
    """
    if not text:
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in CRISIS_KEYWORDS)


CRISIS_RESPONSE = (
    "I'm really glad you reached out, and I want you to know that what you're feeling matters.\n\n"
    "I'm an AI and I'm not equipped to provide the help you need right now, but please know "
    "that support is available. You don't have to face this alone.\n\n"
    "**Please consider reaching out to:**\n"
    "- iCall (India): 9152987821\n"
    "- Vandrevala Foundation: 1860-2662-345 (24/7)\n"
    "- Your nearest hospital emergency department\n\n"
    "If you are in immediate danger, please contact emergency services or go to your nearest hospital right away."
)
