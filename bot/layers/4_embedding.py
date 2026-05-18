# Embedding layer — vectorise text via the LM Studio embeddings API.

import numpy as np
from openai import OpenAI
from .config import BASE_URL, API_KEY, MODELO_EMBED

_cliente = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def obtener_embedding(texto: str) -> "np.ndarray | None":
    try:
        respuesta = _cliente.embeddings.create(input=[texto], model=MODELO_EMBED)
        return np.array([respuesta.data[0].embedding], dtype=np.float32)
    except Exception as e:
        print(f"[ERROR Embedding] {e}")
        return None


def obtener_embeddings_batch(textos: list[str]) -> "np.ndarray | None":
    try:
        respuesta = _cliente.embeddings.create(input=textos, model=MODELO_EMBED)
        print(f"[OK] Embeddings generados para {len(textos)} fragmento(s).")
        return np.array([item.embedding for item in respuesta.data], dtype=np.float32)
    except Exception as e:
        print(f"[ERROR Embedding batch] {e}")
        return None
