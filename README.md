# Enterprise Knowledge Assistant

An AI assistant that answers employee questions using internal policy
documents. It retrieves the relevant policy text, generates an answer
from it, scores its own answer for quality, and saves an audit report
to disk — all automatically, for every question asked.

Built with LangGraph, RAG, RAGAS, and MCP, using only free tools (Groq
for generation, local Ollama for embeddings).

## How It Works

Step 1 - one-time setup
Policy files (data/*.txt) are split into small chunks, converted into
embeddings, and stored in a local vector database (Chroma).

Step 2 - every time a question is asked
Question -> Retriever -> Responder -> Evaluator -> Done
|
RAGAS scores + report saved to disk

- **Retriever** — finds the most relevant policy chunks for the question.
- **Responder** — writes an answer using only those chunks.
- **Evaluator** — scores the answer's quality with RAGAS and saves a
  report file through an MCP server.

## Requirements

Install these before starting:

- Python 3.11+
- [Ollama](https://ollama.com) (runs the embedding model locally, for free)
- Node.js (needed for `npx`, which runs the MCP server)
- [uv](https://docs.astral.sh/uv/) (used to manage the project)
- A free Groq API key: https://console.groq.com/keys

## Setup — Run Once

**Terminal 1 — start Ollama and leave it running:**
```powershell
ollama serve
```
If Ollama is already running in the background, `ollama list` will work
without errors and you can skip this.

**Terminal 2 — set up the project:**
```powershell
uv venv
uv pip install -r requirements.txt
uv run python patch_ragas.py
ollama pull nomic-embed-text
copy .env.example .env
```
Then open `.env` and paste in your Groq API key.

**What `patch_ragas.py` does:** one of the libraries this project uses
(`ragas`) tries to load a Google Cloud feature this project doesn't use,
and that feature is missing in newer versions of a package it depends
on — which crashes the app before it even starts. This script disables
that one unused feature so the app runs normally. Run it once, right
after installing dependencies. It's safe to run again anytime — it
detects if it's already applied and does nothing.

## Running the Project

Make sure `ollama serve` is still running in Terminal 1, then in
Terminal 2:

```powershell
# Browser chat interface (recommended)
uv run streamlit run app.py

# Or from the command line
uv run python main.py "How many casual leaves do I get in a year?"
```

The first time you run it, it will automatically read the policy files
and build the vector database — this takes a few seconds and only
happens once. If you change any files in `data/`, delete the
`chroma_db` folder so it rebuilds with the new content.

**Sample input:**
How many casual leaves do I get in a year?

**Expected output:** a short answer pulled from the policy document
(e.g. "You have 18 casual leaves per calendar year."), plus a link to a
saved audit report containing the question, the answer, the retrieved
policy text, and the quality scores for that answer.

## RAG Design

| Step | Choice |
|---|---|
| Source documents | `.txt` files in `data/` (HR, IT, and finance policies) |
| Chunking | Documents split into ~500-character pieces, with some overlap so context isn't cut off mid-sentence |
| Embeddings | `nomic-embed-text`, run locally through Ollama (free) |
| Vector database | Chroma, stored locally in `chroma_db/` |
| Retrieval | Top 3 most relevant chunks per question |
| Answer generation | Groq (`openai/gpt-oss-120b`), answers using only the retrieved chunks |

Embeddings run locally through Ollama because Groq is fast and free for
generating answers but doesn't offer an embedding service.

## LangGraph Design

Three agents run one after another for every question:

| Node | What it does |
|---|---|
| Retriever | Searches the vector database for relevant policy text |
| Responder | Writes an answer using only that retrieved text |
| Evaluator | Scores the answer with RAGAS and saves a report via MCP |

Flow: `Retriever → Responder → Evaluator → Done`. Nothing loops back —
it's a straight line, and all three agents share one running record of
the question, context, and answer as it moves through the steps.

## MCP Integration

- **Server used:** the official MCP Filesystem server, started
  automatically via `npx` — no account or setup needed.
- **What it's used for:** after scoring an answer, the Evaluator agent
  uses this server's `write_file` tool to save a Markdown report to the
  `reports/` folder, containing the question, the retrieved policy
  text, the answer, and all four quality scores.

## RAGAS Evaluation

Every answer is automatically scored on four metrics:

| Metric | Required? | What it measures |
|---|---|---|
| Faithfulness | Yes | Does the answer stick to what the policy actually says? |
| Answer Relevancy | Yes | Does the answer actually address the question asked? |
| Context Precision | Bonus | Was the retrieved text actually useful? |
| Context Recall | Bonus | Did retrieval find everything relevant? |

Scores range 0–1 and are labeled **Good** (0.8+), **Moderate** (0.5–0.8),
or **Poor** (below 0.5). The same Groq model used to generate answers is
also used as the judge for scoring.

## Observability

There's no separate dashboard needed — every agent prints what it
received and what it produced directly to the terminal, in order, every
time a question runs. A single terminal screenshot shows all three
agents firing one after another with real data, which is the full
execution trace.

## Troubleshooting

**Error mentioning `llama-server binary not found`:** this is an Ollama
installation problem, not a project bug. Uninstall Ollama completely,
delete any leftover Ollama folders under `AppData`, then reinstall the
latest version from https://ollama.com/download and try again.

**Every answer comes back as "I don't know" and every score is 0.00:**
this means the vector database is empty — usually because an earlier
run was interrupted partway through building it (for example, if Ollama
wasn't running yet). Fix it by deleting the database and letting it
rebuild:
```powershell
Remove-Item -Recurse -Force chroma_db
```
Then run the app again.