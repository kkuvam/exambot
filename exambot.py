from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import openai
import ollama

import streamlit as st
from typing import Dict, Generator

# --- CONFIG ---
CHROMA_DB_PATH = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
USE_OPENAI = True
OPENAI_MODEL = "gpt-3.5-turbo"
OLLAMA_MODEL = "llama3.1"
TOP_K = 5

# --- SETUP ---
@st.cache_resource
def load_chroma_client(path):
    return PersistentClient(path=path)

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource
def retrieve_chunks(model, query, top_k=TOP_K):
    embedding = model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    return results["documents"][0] if results["documents"] else []

@st.cache_resource
def ask_openai(context, query):
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful study tutor."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ]
    )
    return response['choices'][0]['message']['content']

@st.cache_resource
def ask_ollama(context, query):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    response = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        system="You are a teacher and evaluator of class10_english",
        stream=False
    )
    return response["response"]

@st.cache_resource
def stream_ollama_response(chat_history):
    # Stream response from Ollama
    stream = ollama.chat(
        model=OLLAMA_MODEL,
        messages=chat_history,
        stream=True
    )    
    for chunk in stream:
        chunk_content = chunk["message"]["content"]
        if chunk_content:
            yield chunk_content

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        {"role": "system", "content": "You are a helpful study tutor."}
    ]

client = load_chroma_client(CHROMA_DB_PATH)
embedder = load_embedder()


# --- SIDEBAR FOR COLLECTION SELECTION ---
st.sidebar.title("📚 ExamBot Settings")

# List available collections
available_collections = [col.name for col in client.list_collections()]

if not available_collections:
    st.sidebar.warning("⚠️ No Chroma collections found.")
    st.stop()

selected_collection = st.sidebar.selectbox("Select a Subject Collection", available_collections, index=0)
collection = client.get_collection(name=selected_collection)

# Store selection in session state
st.session_state["collection"] = selected_collection
st.info(f"✅ Using Subject: `{selected_collection}`")


# --- MAIN APP ---
PAGE_TITLE = "Exambot"
SYSTEM_PROMPT = f"""You are a helpful exambot and you can help student in exam preparation on selected topic."""


st.set_page_config(
    page_title=PAGE_TITLE,
    initial_sidebar_state="expanded",
    layout="wide",
)
st.title(PAGE_TITLE)
st.markdown("Ask questions based on your study notes stored in ChromaDB.")

# --- QUERY INPUT ---
query = st.text_input("🔍 What would you like to learn?", placeholder="e.g. Select a subject and ask quesion")

if query:
    # --- Embed query and retrieve context ---
    embedding = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[embedding], n_results=TOP_K)
    documents = results["documents"][0] if results["documents"] else []

    if not documents:
        st.error("❌ No relevant context found.")
        st.stop()

    context = "\n\n---\n\n".join(documents)
    with st.expander("🧾 View Retrieved Context", expanded=False):
        st.markdown(context)
    # --- Show user query ---
    st.subheader("🧠 Your Question")
    st.markdown(query)

    # --- Append user input to chat history ---
    st.session_state.chat_history.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion:\n{query}"
    })

    # --- Stream LLM response ---
    st.subheader("💡 Answer")
    response_container = st.empty()
    response_text = ""

    with st.spinner("💬 Thinking..."):
        for token in stream_ollama_response(st.session_state.chat_history):
            response_text += token
            response_container.markdown(response_text + "▌")

    # Finalize response
    response_container.markdown(response_text)

    # --- Append assistant reply to chat history ---
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response_text
    })