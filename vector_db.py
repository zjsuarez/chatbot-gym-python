# vector_db.py — Index build script. Run once (or after documents change).
# Run with:  python vector_db.py

from layers import cargar_documentos, construir_y_guardar_indice
from layers.config import RUTA_DOCUMENTOS

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  📦 KAIZEN — Construcción de Índice FAISS")
    print("=" * 50)
    chunks = cargar_documentos(RUTA_DOCUMENTOS)
    construir_y_guardar_indice(chunks)
    print("\n✅ Listo. Ahora puedes ejecutar: python main.py")
