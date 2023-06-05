import os
from pathlib import Path
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_chroma_client(persist_path: str):
    if not os.path.exists(persist_path):
        print(f"Creating new Chroma DB at: {persist_path}")
    else:
        print(f"Loading existing Chroma DB from: {persist_path}")

    client = PersistentClient(path=persist_path)  
    return client


def get_embedding_function(model_name: str):
    print(f"Using embedding model: {model_name}")
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def get_chroma_collection(client, collection_name: str, embedding_function: SentenceTransformerEmbeddingFunction):
    if collection_name in client.list_collections():
        print(f"📚 Using existing collection: {collection_name}")
    else:
        print(f"🆕 Creating new collection: {collection_name}")
    
    return client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)


def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)


def embed_chunks(text: str, source: str):
    chunks = []
    metadatas = []
    for i, chunk in enumerate(split_text(text)):
        chunks.append(chunk)
        metadatas.append({"source": source, "chunk": i})
    return chunks, metadatas


def store_in_chroma(collection, chunks, metadatas):
    ids = [f"doc_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    print(f"{len(chunks)} chunks stored.")



if __name__ == "__main__":

    # ---- CONFIG ----
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    MODEL_NAME = "all-MiniLM-L6-v2"
    # ----------------
    markdown_dir = "../books/10/english"
    persist_dir = "../chroma_db"
    collection_name = "class10_english"

    # Initialize Chroma client
    client = get_chroma_client(persist_dir)
    embedding_function = get_embedding_function(MODEL_NAME)
    collection = get_chroma_collection(client, collection_name, embedding_function)

    # Read all markdown files from the directory
    markdown_files = list(Path(markdown_dir).glob("*.md"))
    if not markdown_files:
        print(f"No markdown files found in {markdown_dir}.")
        exit(1)
    else:
        for file in markdown_files:
            print(f"Processing file: {file.name}")
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Embed and prepare metadata
            embedded_chunks, metadatas = embed_chunks(text, str(file))
            
            # Store in Chroma
            store_in_chroma(collection, embedded_chunks, metadatas)
            print(f"Finished processing {file.name}\n")
    print("All files processed and stored in Chroma DB.")
