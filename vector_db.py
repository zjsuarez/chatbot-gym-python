# vector_db.py — Construcción y persistencia del índice FAISS local.
# Ejecutar una sola vez (o cuando cambien los documentos):  python vector_db.py

import os
import pickle
import faiss
import numpy as np
from openai import OpenAI

# ---------------------------------------------------------------------------
# [Fase 1: Configuración Local] - Carga de variables de entorno y preparación
#   del índice FAISS local.
# ---------------------------------------------------------------------------

RUTA_DOCUMENTOS = "documentos"
RUTA_INDICE     = "indice.faiss"   # Artefacto de salida
RUTA_CHUNKS     = "chunks.pkl"     # Artefacto de salida
MODELO_EMBED    = "text-embedding-bge-small-en-v1.5"

print("[INFO] Conectando con LM Studio para embeddings...")
_cliente = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")


def obtener_embeddings(textos: list[str]) -> np.ndarray | None:
    """Vectoriza una lista de textos con LM Studio."""
    try:
        respuesta = _cliente.embeddings.create(input=textos, model=MODELO_EMBED)
        print(f"[OK] Embeddings generados para {len(textos)} fragmento(s).")
        return np.array([item.embedding for item in respuesta.data], dtype=np.float32)
    except Exception as e:
        print(f"[ERROR] Fallo al crear embeddings: {e}")
        return None


def cargar_documentos(ruta_carpeta: str) -> list[str]:
    """Lee todos los .txt de la carpeta (y subcarpetas) y los divide en chunks."""
    textos: list[str] = []
    if not os.path.exists(ruta_carpeta):
        print(f"[AVISO] La carpeta '{ruta_carpeta}' no existe.")
        return textos

    for raiz, _, archivos in os.walk(ruta_carpeta):
        for archivo in archivos:
            if archivo.endswith(".txt"):
                ruta = os.path.join(raiz, archivo)
                with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
                    chunks = f.read().split("\n\n")
                    textos.extend([c.strip() for c in chunks if len(c.strip()) > 10])

    print(f"[INFO] {len(textos)} chunks cargados desde '{ruta_carpeta}'.")
    return textos


def construir_y_guardar_indice(chunks: list[str]) -> None:
    """Genera embeddings, crea el índice FAISS y lo persiste en disco."""
    if not chunks:
        print("[ERROR] No hay chunks para indexar.")
        return

    print(f"[INFO] Construyendo índice FAISS para {len(chunks)} chunks...")
    embeddings = obtener_embeddings(chunks)
    if embeddings is None:
        print("[ERROR] No se pudo construir el índice.")
        return

    dimension = embeddings.shape[1]
    indice = faiss.IndexFlatL2(dimension)
    indice.add(embeddings)

    faiss.write_index(indice, RUTA_INDICE)
    with open(RUTA_CHUNKS, "wb") as f:
        pickle.dump(chunks, f)

    print(f"[OK] Índice guardado → '{RUTA_INDICE}' ({indice.ntotal} vectores)")
    print(f"[OK] Chunks guardados → '{RUTA_CHUNKS}'")


# --- Función de compatibilidad para imports desde main.py (legacy) ---
def recuperar_chunks(pregunta: str, k: int = 3) -> str:
    """
    Mantiene compatibilidad con código que importe directamente desde vector_db.
    En el flujo refactorizado, la recuperación ocurre en bot_logic.recuperar_contexto().
    """
    if not os.path.exists(RUTA_INDICE) or not os.path.exists(RUTA_CHUNKS):
        return ""
    indice = faiss.read_index(RUTA_INDICE)
    with open(RUTA_CHUNKS, "rb") as f:
        chunks = pickle.load(f)

    vector = obtener_embeddings([pregunta])
    if vector is None:
        return ""
    _, indices = indice.search(vector, k)
    return "\n".join(chunks[i] for i in indices[0] if i != -1 and i < len(chunks))


# ---------------------------------------------------------------------------
# Punto de entrada: construye el índice si se ejecuta directamente
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  📦 KAIZEN — Construcción de Índice FAISS")
    print("=" * 50)
    mis_chunks = cargar_documentos(RUTA_DOCUMENTOS)
    construir_y_guardar_indice(mis_chunks)
    print("\n✅ Listo. Ahora puedes ejecutar: python main.py")