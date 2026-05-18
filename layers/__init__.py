import importlib as _il

# Map every public symbol to its numbered layer file.
_SYMBOL_MAP = {
    # 1. User
    "USUARIO_MOCK":               "layers.1_user",
    # 2. Input
    "leer_entrada":               "layers.2_input_layer",
    "mostrar_respuesta":          "layers.2_input_layer",
    # 3. Preprocessing
    "preprocesar_input":          "layers.3_preprocessing",
    "VOCABULARIO_CORE":           "layers.3_preprocessing",
    "EXPANSION_QUERY":            "layers.3_preprocessing",
    # 4. Embedding
    "obtener_embedding":          "layers.4_embedding",
    "obtener_embeddings_batch":   "layers.4_embedding",
    # 5. FAISS
    "cargar_indice":              "layers.5_faiss_layer",
    "buscar":                     "layers.5_faiss_layer",
    "construir_y_guardar_indice": "layers.5_faiss_layer",
    # 6. Chunks
    "cargar_documentos":          "layers.6_chunks",
    "guardar_chunks":             "layers.6_chunks",
    "cargar_chunks":              "layers.6_chunks",
    # 7. Prompt
    "construir_system_prompt":    "layers.7_prompt",
    # 8. LLM
    "llamar_llm_json":            "layers.8_llm",
    "simular_procesado_backend":  "layers.8_llm",
}


def __getattr__(name: str):
    if name in _SYMBOL_MAP:
        mod = _il.import_module(_SYMBOL_MAP[name])
        val = getattr(mod, name)
        globals()[name] = val  # cache so __getattr__ isn't called twice
        return val
    raise AttributeError(f"module 'layers' has no attribute {name!r}")
