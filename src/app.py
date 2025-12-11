# app.py
import streamlit as st
from src.chatbot import MentalHealthChatbot

st.set_page_config(page_title="AI Mental Health Chatbot", page_icon="🧠", layout="centered")

@st.cache_resource
def load_bot():
    # instantiate chatbot (this loads tokenizer/models lazily when first used)
    return MentalHealthChatbot()

bot = load_bot()

st.title("🧠 AI Mental Health Support Chatbot")
st.write(
    "This chatbot offers supportive, non-clinical conversation using NLP.\n\n"
    "**Important:** This is not a replacement for professional mental health care."
)

if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = st.text_area("How are you feeling today?", height=140, placeholder="Type anything on your mind...")

cols = st.columns([1, 1, 2])
with cols[0]:
    send = st.button("Send")
with cols[1]:
    clear = st.button("Reset")

if send and user_input.strip():
    # get response dict: {emotion, sentiment, reply}
    try:
        result = bot.get_response(user_input)
    except Exception as e:
        # Fallback safe message if models aren't loaded or an error occurs
        result = {
            "emotion": "error",
            "sentiment": "neutral",
            "reply": "Sorry, I'm having trouble analyzing that right now. Please try again."
        }
    # push to session history
    st.session_state["history"].append(
        {"user": user_input, "bot": result["reply"], "emotion": result["emotion"], "sentiment": result["sentiment"]}
    )

if clear:
    st.session_state["history"] = []
    bot.reset_conversation()

# show conversation (most recent last)
for turn in st.session_state["history"]:
    st.markdown(f"**You:** {turn['user']}")
    st.markdown(f"**Bot:** {turn['bot']}")
    st.markdown(f"_Emotion: {turn['emotion']} • Sentiment: {turn['sentiment']}_")
    st.markdown("---")

st.caption(
    "If you feel you may harm yourself or are in immediate danger, contact local emergency services or a trusted person immediately."
)
