# FAISS layer — build, save, load, and search the vector index.

import os
import faiss
from . import obtener_embedding, obtener_embeddings_batch, guardar_chunks, cargar_chunks
from .config import RUTA_INDICE, RUTA_CHUNKS, TOP_K


def cargar_indice(ruta_indice: str = RUTA_INDICE, ruta_chunks: str = RUTA_CHUNKS):
    if not os.path.exists(ruta_indice) or not os.path.exists(ruta_chunks):
        print(
            f"[AVISO] No se encontraron '{ruta_indice}' / '{ruta_chunks}'.\n"
            "Ejecuta vector_db.py primero para construir el índice."
        )
        return None, []
    indice = faiss.read_index(ruta_indice)
    chunks = cargar_chunks(ruta_chunks)
    print(f"[FAISS] Índice cargado: {indice.ntotal} vectores | {len(chunks)} chunks.")
    return indice, chunks


def buscar(indice, chunks: list[str], pregunta_procesada: str, top_k: int = TOP_K) -> str:
    if indice is None or not chunks:
        return ""

    vector = obtener_embedding(pregunta_procesada)
    if vector is None:
        return ""

    _, indices = indice.search(vector, top_k)
    fragmentos = [chunks[i] for i in indices[0] if i != -1 and i < len(chunks)]
    print(f"[FAISS] {len(fragmentos)} fragmentos recuperados.\n")
    return "\n---\n".join(fragmentos)


def construir_y_guardar_indice(
    chunks: list[str],
    ruta_indice: str = RUTA_INDICE,
    ruta_chunks: str = RUTA_CHUNKS,
) -> None:
    if not chunks:
        print("[ERROR] No hay chunks para indexar.")
        return

    print(f"[INFO] Construyendo índice FAISS para {len(chunks)} chunks...")
    embeddings = obtener_embeddings_batch(chunks)
    if embeddings is None:
        print("[ERROR] No se pudo construir el índice.")
        return

    dimension = embeddings.shape[1]
    indice = faiss.IndexFlatL2(dimension)
    indice.add(embeddings)

    faiss.write_index(indice, ruta_indice)
    guardar_chunks(chunks, ruta_chunks)
    print(f"[OK] Índice guardado → '{ruta_indice}' ({indice.ntotal} vectores)")
