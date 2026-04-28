import os
import faiss
from sentence_transformers import SentenceTransformer

print("[INFO] Cargando modelo de embeddings de IA...")
modelo_embeddings = SentenceTransformer('all-MiniLM-L6-v2')

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
        
    embeddings = modelo_embeddings.encode(chunks)
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
        
    vector_pregunta = modelo_embeddings.encode([pregunta])
    distancias, indices = indice_faiss.search(vector_pregunta, k)
    
    contexto = []
    for i in indices[0]:
        if i != -1:
            contexto.append(base_datos_texto[i])
    return "\n".join(contexto)