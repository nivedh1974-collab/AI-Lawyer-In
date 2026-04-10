import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(
    page_title="Indian AI Lawyer",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Indian AI Lawyer")
st.caption("Ask questions about Indian law. Powered by BNS, BNSS, BSA, Constitution & Kerala Motor Vehicles Act.")

# ── CHAT HISTORY ─────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── INPUT ─────────────────────────────────────────────
if question := st.chat_input("Ask a legal question..."):
    
    # Show user question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Get answer from API
    with st.chat_message("assistant"):
        with st.spinner("Searching law documents..."):
            response = requests.post(API_URL, json={"question": question})
            answer = response.json()["answer"]
            st.write(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})