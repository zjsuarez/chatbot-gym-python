import os
import faiss
import numpy as np
from openai import OpenAI

print("[INFO] Conectando con LM Studio para embeddings...")
# Configuramos el cliente OpenAI igual que en main.py
cliente_ai = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="api-key-no-necesaria")

def obtener_embeddings(textos):
    # Llama a LM Studio para obtener los vectores
    try:
        respuesta = cliente_ai.embeddings.create(
            input=textos,
            model="text-embedding-bge-small-en-v1.5" # LM Studio suele usar el modelo que esté cargado
        )
        print(f"[ÉXITO] Embeddings generados correctamente para {len(textos)} fragmento(s).")
        # Extraemos los embeddings y los convertimos en un array de numpy
        return np.array([item.embedding for item in respuesta.data], dtype=np.float32)
    except Exception as e:
        print(f"\n[ERROR] Falló la creación de embeddings con LM Studio.")
        print("¿Está iniciada la API en LM Studio y tienes un modelo de Text Embedding cargado?")
        print(f"Detalle del error: {e}\n")
        return None

def cargar_documentos(ruta_carpeta):
    textos = []
    # Verificamos si la carpeta existe para evitar errores
    if not os.path.exists(ruta_carpeta):
        print(f"⚠️ La carpeta '{ruta_carpeta}' no existe. Por favor, créala.")
        return textos
        
    for archivo in os.listdir(ruta_carpeta):
        if archivo.endswith(".txt"):
            with open(os.path.join(ruta_carpeta, archivo), 'r', encoding='utf-8') as f:
                chunks = f.read().split('\n\n')
                textos.extend([c.strip() for c in chunks if len(c.strip()) > 10])
    return textos

def crear_indice_faiss(chunks):
    if not chunks: # Si no hay documentos, devolvemos vacío
        return None, []
        
    print(f"[INFO] Creando índice vectorial para {len(chunks)} documentos...")
    embeddings = obtener_embeddings(chunks)
    
    if embeddings is None:
        print("[ERROR] No se pudo crear el índice de FAISS porque no hay embeddings.")
        return None, []
        
    dimension = embeddings.shape[1]
    indice = faiss.IndexFlatL2(dimension)
    indice.add(embeddings)
    return indice, chunks

# --- INICIALIZACIÓN AUTOMÁTICA ---
# Cuando importemos este archivo, cargará los documentos automáticamente
mis_chunks = cargar_documentos("documentos")
indice_faiss, base_datos_texto = crear_indice_faiss(mis_chunks)

def recuperar_chunks(pregunta, k=2):
    if not indice_faiss:
        return "" # Si no hay base de datos, no devuelve nada
        
    vector_pregunta = obtener_embeddings([pregunta])
    if vector_pregunta is None:
        print("[ERROR] No se pudo obtener el contexto (fallo en embedding de la pregunta).")
        return ""
        
    distancias, indices = indice_faiss.search(vector_pregunta, k)
    
    contexto = []
    for i in indices[0]:
        if i != -1:
            contexto.append(base_datos_texto[i])
    return "\n".join(contexto)