"""
Streamlit chat UI for the Enterprise Knowledge Assistant.
"""

import os
import streamlit as st

from app.ingest import build_vectorstore
from app.graph import build_graph
from app.config import CHROMA_DIR

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    layout="centered",
)

st.title("Enterprise Knowledge Assistant")
st.caption("LangGraph · RAG · RAGAS · MCP · Streamlit")


@st.cache_resource(show_spinner="Building vector store, please wait...")
def get_graph():
    if not os.path.exists(CHROMA_DIR):
        build_vectorstore()
    return build_graph()


graph = get_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Rendering chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            st.caption(f"Audit report → `{msg['report_path']}`")

# Chat input
if question := st.chat_input("Ask any question.... "):

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Running Retriever → Responder → Evaluator..."):
            result = graph.invoke({"question": question})

        st.markdown(result["answer"])
        st.caption(f"Audit report → `{result['report_path']}`")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "report_path": result["report_path"],
    })