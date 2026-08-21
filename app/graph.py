#LangGraph definition: Retriever Agent -> Response Agent -> Evaluator Agent -> END

import asyncio
from datetime import datetime
from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from app.config import (
    CHROMA_DIR, REPORTS_DIR, GROQ_CHAT_MODEL, GROQ_API_KEY,
    OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL,
)
from app.mcp_tools import write_report_via_mcp


class AssistantState(TypedDict):
    question: str
    context: List[str]
    answer: str
    scores: dict
    report_path: str


# Built once and reused by every run
llm = ChatGroq(model=GROQ_CHAT_MODEL, api_key=GROQ_API_KEY, temperature=0)
embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def _log(node: str, direction: str, message: str) -> None:
    """One consistent log format for every node - this is what you screenshot
    to prove the graph's execution path (requirement 5)."""
    arrow = "->" if direction == "in" else "<-"
    print(f"[{node.upper():9s}] {arrow} {message}")


def _interpret(score: float) -> str:
    if score >= 0.8:
        return "Good"
    if score >= 0.5:
        return "Moderate"
    return "Poor"

# Node 1: Retriever Agent
def retriever_node(state: AssistantState) -> dict:
    print("\n" + "=" * 70)
    _log("retriever", "in", f"question = \"{state['question']}\"")
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    docs = vectordb.similarity_search(state["question"], k=3)
    context = [d.page_content for d in docs]
    _log("retriever", "out", f"{len(context)} chunk(s) retrieved from Chroma")
    print("=" * 70)
    return {"context": context}

# Node 2: Response Agent
def response_node(state: AssistantState) -> dict:
    print("\n" + "=" * 70)
    _log("responder", "in", f"{len(state['context'])} context chunk(s) + question")
    context_text = "\n\n".join(state["context"])
    prompt = (
        "You are an internal enterprise knowledge assistant. "
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {state['question']}\n\n"
        "Answer:"
    )
    answer = llm.invoke(prompt).content
    _log("responder", "out", f"answer generated ({len(answer)} chars)")
    print("=" * 70)
    return {"answer": answer}

# Node 3: Evaluator Agent  (RAGAS scoring + the MCP call)
async def _score_all(sample: SingleTurnSample) -> dict:
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    metrics = {
        # Mandatory
        "faithfulness": Faithfulness(llm=ragas_llm),
        "answer_relevancy": ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1),
        # Bonus
        "context_precision": LLMContextPrecisionWithoutReference(llm=ragas_llm),
        "context_recall": LLMContextRecall(llm=ragas_llm),
    }

    results = {}
    for name, metric in metrics.items():
        results[name] = float(await metric.single_turn_ascore(sample))
    return results


def evaluator_node(state: AssistantState) -> dict:
    print("\n" + "=" * 70)
    _log("evaluator", "in", "answer + context -> scoring with RAGAS (4 metrics)")

    sample = SingleTurnSample(
        user_input=state["question"],
        response=state["answer"],
        retrieved_contexts=state["context"],
        reference=state["answer"],
    )

    scores = asyncio.run(_score_all(sample))

    for name, value in scores.items():
        _log("evaluator", "out", f"{name}: {value:.2f} ({_interpret(value)})")

    report = (
        f"# Evaluation Report\n\n"
        f"Generated: {datetime.now().isoformat()}\n\n"
        f"## Question\n{state['question']}\n\n"
        f"## Retrieved Context\n" + "\n---\n".join(state["context"]) + "\n\n"
        f"## Answer\n{state['answer']}\n\n"
        f"## RAGAS Scores\n"
        + "\n".join(f"- {k}: {v:.2f} ({_interpret(v)})" for k, v in scores.items())
        + "\n\n_Note: context_recall uses the generated answer as a proxy "
        "reference (no curated ground truth in this demo)._\n"
    )

    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    _log("evaluator", "out", "writing report to disk via MCP filesystem server...")
    report_path = write_report_via_mcp(REPORTS_DIR, filename, report)
    _log("evaluator", "out", f"report saved at: {report_path}")
    print("=" * 70)

    return {"scores": scores, "report_path": report_path}


def build_graph():
    graph = StateGraph(AssistantState)

    graph.add_node("retriever", retriever_node)
    graph.add_node("responder", response_node)
    graph.add_node("evaluator", evaluator_node)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "responder")
    graph.add_edge("responder", "evaluator")
    graph.add_edge("evaluator", END)

    return graph.compile()