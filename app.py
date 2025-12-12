import streamlit as st
from src.chatbot import MentalHealthChatbot

st.set_page_config(page_title="AI Mental Health Chatbot", page_icon="🧠")

# Load the chatbot once (cached)
@st.cache_resource
def load_bot():
    return MentalHealthChatbot()

bot = load_bot()

st.title("🧠 AI Mental Health Support Chatbot")
st.write(
    "This chatbot provides supportive, non-clinical emotional assistance using NLP.\n\n"
    "**Disclaimer:** This tool does not replace professional mental health care."
)

if "history" not in st.session_state:
    st.session_state["history"] = []

# User input field
user_input = st.text_area("How are you feeling today?", height=150, placeholder="Share what's on your mind...")

col1, col2 = st.columns([1, 1])
with col1:
    send = st.button("Send")
with col2:
    clear = st.button("Reset")

# Handle send button
if send and user_input.strip():
    result = bot.get_response(user_input)
    
    st.session_state["history"].append({
        "user": user_input,
        "bot": result["reply"],
        "emotion": result["emotion"],
        "sentiment": result["sentiment"]
    })

# Handle reset
if clear:
    st.session_state["history"] = []
    bot.reset_conversation()

# Display conversation history
for msg in reversed(st.session_state["history"]):
    st.markdown(f"**You:** {msg['user']}")
    st.markdown(f"**Bot:** {msg['bot']}")
    st.markdown(f"_Emotion: {msg['emotion']} • Sentiment: {msg['sentiment']}_")
    st.markdown("---")

st.caption(
    "If you're feeling unsafe or in crisis, please contact emergency services or a trusted professional immediately."
)
