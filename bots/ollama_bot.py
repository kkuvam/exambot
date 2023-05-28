import streamlit as st
from ollama import chat
from typing import Dict, Generator

PAGE_TITLE = "Streamlit Ollama Chatbot"
SYSTEM_PROMPT = f"""You are a helpful chatbot and you can can answer questions for users on any topic."""

# Function to interact with Ollama and stream the response
def ollama_generator(model_name: str, messages: Dict) -> Generator:
    stream = chat(model=model_name, messages=messages, stream=True)
    for chunk in stream:
        yield chunk['message']['content']

st.set_page_config(
    page_title=PAGE_TITLE,
    initial_sidebar_state="expanded"
)

st.title(PAGE_TITLE)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("What up?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.spinner("Generating response..."):
        # retrieves response from model
        response_generator = ollama_generator("llama2:7b", st.session_state.messages)

        # streams the response back to the screen
        response = st.write_stream(response_generator)

        # appends response to the message list
        st.session_state.messages.append({"role": "assistant", "content": response})