import importlib as _il

# Relative paths — location-independent regardless of where the package lives.
_SYMBOL_MAP = {
    # 1. User
    "USUARIO_MOCK":               ".1_user",
    # 3. Preprocessing
    "preprocesar_input":          ".3_preprocessing",
    "VOCABULARIO_CORE":           ".3_preprocessing",
    "EXPANSION_QUERY":            ".3_preprocessing",
    # 4. Embedding
    "obtener_embedding":          ".4_embedding",
    "obtener_embeddings_batch":   ".4_embedding",
    # 5. FAISS
    "cargar_indice":              ".5_faiss_layer",
    "buscar":                     ".5_faiss_layer",
    "construir_y_guardar_indice": ".5_faiss_layer",
    # 6. Chunks
    "cargar_documentos":          ".6_chunks",
    "guardar_chunks":             ".6_chunks",
    "cargar_chunks":              ".6_chunks",
    # 7. Prompt
    "construir_system_prompt":    ".7_prompt",
    # 8. LLM
    "llamar_llm_json":            ".8_llm",
    "simular_procesado_backend":  ".8_llm",
}


def __getattr__(name: str):
    if name in _SYMBOL_MAP:
        mod = _il.import_module(_SYMBOL_MAP[name], package=__name__)
        val = getattr(mod, name)
        globals()[name] = val  # cache so __getattr__ isn't called twice
        return val
    raise AttributeError(f"module 'layers' has no attribute {name!r}")
