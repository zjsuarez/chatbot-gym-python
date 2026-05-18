# pipeline.py — Chains the individual layers into the full RAG pipeline.

from bot.layers import (
    preprocesar_input,
    cargar_indice,
    buscar,
    construir_system_prompt,
    llamar_llm_json,
    simular_procesado_backend,
)
from bot.layers.config import RUTA_INDICE, RUTA_CHUNKS

indice_faiss, base_chunks = cargar_indice(RUTA_INDICE, RUTA_CHUNKS)


def recargar_indice() -> None:
    global indice_faiss, base_chunks
    indice_faiss, base_chunks = cargar_indice(RUTA_INDICE, RUTA_CHUNKS)


def procesar_peticion(pregunta: str, datos_usuario: dict) -> dict:
    pregunta_limpia = preprocesar_input(pregunta)
    contexto        = buscar(indice_faiss, base_chunks, pregunta_limpia)
    print(f"[DEBUG - Contexto FAISS]:\n{contexto}\n")
    system_prompt   = construir_system_prompt(datos_usuario, contexto)
    resultado       = llamar_llm_json(system_prompt, pregunta)
    simular_procesado_backend(resultado)
    return resultado
