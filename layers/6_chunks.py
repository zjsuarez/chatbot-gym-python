# Chunks layer — load documents from disk and persist/restore the chunk list.

import os
import pickle


def cargar_documentos(ruta_carpeta: str) -> list[str]:
    textos: list[str] = []
    if not os.path.exists(ruta_carpeta):
        print(f"[AVISO] La carpeta '{ruta_carpeta}' no existe.")
        return textos

    for raiz, _, archivos in os.walk(ruta_carpeta):
        for archivo in archivos:
            if archivo.endswith((".txt", ".md")):
                ruta = os.path.join(raiz, archivo)
                with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
                    raw_chunks = f.read().split("\n\n")
                    textos.extend([c.strip() for c in raw_chunks if len(c.strip()) > 10])

    print(f"[INFO] {len(textos)} chunks cargados desde '{ruta_carpeta}'.")
    return textos


def guardar_chunks(chunks: list[str], ruta: str) -> None:
    with open(ruta, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[OK] Chunks guardados → '{ruta}'")


def cargar_chunks(ruta: str) -> list[str]:
    if not os.path.exists(ruta):
        return []
    with open(ruta, "rb") as f:
        return pickle.load(f)
