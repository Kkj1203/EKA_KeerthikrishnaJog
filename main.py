import sys
import os

from app.ingest import build_vectorstore
from app.graph import build_graph
from app.config import CHROMA_DIR


def ensure_vectorstore():
    if not os.path.exists(CHROMA_DIR):
        print("[Main] No vector store found, running ingestion first...")
        build_vectorstore()
    else:
        print("[Main] Existing vector store found, skipping ingestion.")
        print("[Main] (Delete the chroma_db/ folder to re-ingest from scratch.)")


def main():
    default_question = "How many casual leaves am I entitled to per year?"
    question = sys.argv[1] if len(sys.argv) > 1 else default_question

    ensure_vectorstore()

    app = build_graph()
    print(f"\n[Main] Running graph for question: '{question}'")
    final_state = app.invoke({"question": question})

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Question: {final_state['question']}")
    print(f"\nAnswer:\n{final_state['answer']}")
    print("\nRAGAS scores:")
    for name, value in final_state["scores"].items():
        print(f"  {name:18s} {value:.2f}")
    print(f"\nFull report written to: {final_state['report_path']}")


if __name__ == "__main__":
    main()