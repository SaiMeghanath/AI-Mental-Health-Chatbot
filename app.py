"""
app.py — Streamlit web interface for the Mental Health Support Chatbot
Run: streamlit run app.py
"""
import streamlit as st
from src.chatbot import MentalHealthChatbot

st.set_page_config(
    page_title="Mental Health Support Chatbot",
    page_icon="🧠",
    layout="centered",
)

# --- Load chatbot (cached across reruns) ---
@st.cache_resource
def load_bot():
    return MentalHealthChatbot()

bot = load_bot()

# --- Header ---
st.title("🧠 Mental Health Support Chatbot")
st.markdown(
    "> **Disclaimer:** This tool provides empathetic conversational support only. "
    "It is **not** a substitute for professional mental health care."
)
st.divider()

# --- Session state ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Input area ---
user_input = st.text_area(
    "How are you feeling today?",
    height=120,
    placeholder="Share what's on your mind — this is a safe, private space.",
)

col1, col2 = st.columns([1, 1])
with col1:
    send = st.button("💬 Send", use_container_width=True)
with col2:
    clear = st.button("🔄 Reset Conversation", use_container_width=True)

# --- Handle send ---
if send and user_input.strip():
    result = bot.get_response(user_input)
    st.session_state.history.append({
        "user": user_input,
        "bot": result["reply"],
        "emotion": result["emotion"],
        "sentiment": result["sentiment"],
        "emotion_score": result.get("emotion_score"),
        "sentiment_score": result.get("sentiment_score"),
    })

# --- Handle reset ---
if clear:
    st.session_state.history = []
    bot.reset_conversation()

# --- Display conversation (newest first) ---
if st.session_state.history:
    st.subheader("Conversation")
    for msg in reversed(st.session_state.history):
        with st.chat_message("user"):
            st.write(msg["user"])
        with st.chat_message("assistant"):
            st.write(msg["bot"])
            if msg["emotion"] != "crisis":
                score_text = ""
                if msg.get("emotion_score") is not None:
                    score_text += f"Emotion: **{msg['emotion']}** ({msg['emotion_score']:.0%})"
                if msg.get("sentiment_score") is not None:
                    score_text += f"  |  Sentiment: **{msg['sentiment']}** ({msg['sentiment_score']:.0%})"
                if score_text:
                    st.caption(score_text)

# --- Footer ---
st.divider()
st.caption(
    "⚠️ If you are in crisis or feel unsafe, please contact emergency services immediately "
    "or call iCall: **9152987821** (India)"
)
