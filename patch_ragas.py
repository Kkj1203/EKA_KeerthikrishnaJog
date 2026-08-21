"""
uv run python patch_ragas.py
"""

import pathlib
import sys

path = pathlib.Path(".venv/Lib/site-packages/ragas/llms/base.py")

if not path.exists():
    print(f"Could not find {path}")
    print("Make sure you're running this from your project root "
          "(same folder as pyproject.toml) with the venv already installed.")
    sys.exit(1)

text = path.read_text(encoding="utf-8")

if "except ImportError" in text and "class ChatVertexAI" in text:
    print("Already patched — nothing to do. Just rerun your app.")
    sys.exit(0)

old = (
    "from langchain_community.chat_models.vertexai import ChatVertexAI\n"
    "from langchain_community.llms import VertexAI\n"
)

new = (
    "try:\n"
    "    from langchain_community.chat_models.vertexai import ChatVertexAI\n"
    "    from langchain_community.llms import VertexAI\n"
    "except ImportError:\n"
    "    class ChatVertexAI:  # stub: vertexai integration not installed, unused\n"
    "        pass\n"
    "    class VertexAI:  # stub: vertexai integration not installed, unused\n"
    "        pass\n"
)

if old not in text:
    print("The import lines didn't match exactly what was expected.")
    print(f"Open this file manually and fix it by hand: {path}")
    print("Find the lines importing ChatVertexAI and VertexAI from")
    print("langchain_community, and wrap them in try/except ImportError")
    print("with empty stub classes as the fallback (see this script's")
    print("'new' variable for the exact pattern to use).")
    sys.exit(1)

path.write_text(text.replace(old, new), encoding="utf-8")
print(f"Patched: {path}")
print("Now run: uv run streamlit run app.py")