#Central place for settings

import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
REPORTS_DIR = "reports"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not found. Copy .env.example to .env and add your "
        "free Groq API key (https://console.groq.com/keys)."
    )

#Ollama pull nomic-embed-text
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")