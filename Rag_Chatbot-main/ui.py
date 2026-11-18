import streamlit as st
import requests

st.title("RAG Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Enter your query here")

if query:
    url = "http://127.0.0.1:8000/ask"

    with st.spinner("Thinking..."):
        response = requests.post(url, json={"query": query})

    if response.status_code == 200:
        data = response.json()
        answer = data.get("answer", "No response")

        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Bot", answer))
    else:
        st.write("Error connecting to API")

for role, msg in st.session_state.chat_history:
    st.write(f"{role}: {msg}")