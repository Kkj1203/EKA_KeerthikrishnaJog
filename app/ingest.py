"""
RAG ingestion pipeline.
Steps:
1. Load documents from a knowledge source   -> load_documents()
2. Perform chunking                         -> chunk_documents()
3. Generate embeddings + store in vector db -> build_vectorstore()
Run this file directly to (re)build the vector store, or let main.py call
build_vectorstore() automatically the first time it runs.
"""

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import DATA_DIR, CHROMA_DIR, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL


def load_documents() -> list[Document]:
    """Loads every .txt file in data into a LangChain Document."""
    docs = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(DATA_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append(Document(page_content=text, metadata={"source": fname}))
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Simple fixed-size chunking with a small overlap so context isn't cut
    mid-sentence too often. 500/50 works well for short policy documents."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)


def build_vectorstore() -> Chroma:
    docs = load_documents()
    chunks = chunk_documents(docs)
    print(f"[Ingest] Loaded {len(docs)} document(s), split into {len(chunks)} chunk(s).")

    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"[Ingest] Vector store built and saved to '{CHROMA_DIR}'.")
    return vectordb


if __name__ == "__main__":
    build_vectorstore()