from openai import OpenAI
import streamlit as st

st.title("ChatGPT-like clone")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="your_openrouter_api_key_here"  # Replace with your OpenRouter API key,
)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "mistralai/devstral-small"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})