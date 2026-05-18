from pathlib import Path

# Resolve project root: bot/layers/ -> bot/ -> root
_ROOT = Path(__file__).resolve().parent.parent.parent

BASE_URL        = "http://127.0.0.1:1234/v1"
API_KEY         = "lm-studio"
MODELO_LLM      = "dolphin3.0-llama3.1-8b"
MODELO_EMBED    = "text-embedding-bge-small-en-v1.5"
TOP_K           = 3
TEMPERATURA     = 0.2
RUTA_INDICE     = str(_ROOT / "index" / "indice.faiss")
RUTA_CHUNKS     = str(_ROOT / "index" / "chunks.pkl")
RUTA_DOCUMENTOS = str(_ROOT / "docs")
