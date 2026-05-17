# bot_logic.py — Lógica central del chatbot RAG con router de intención para el Proyecto Kaizen.

import re
import json
import pickle
import os
import numpy as np
import faiss

try:
    import spacy
    from rapidfuzz import process, fuzz
except ImportError:
    spacy = None
    process = None
    fuzz = None

from openai import OpenAI

# ---------------------------------------------------------------------------
# [Fase 1: Configuración Local] - Carga de variables de entorno y preparación
#   del índice FAISS local.
# ---------------------------------------------------------------------------

# Rutas a los artefactos FAISS pre-construidos (generados por vector_db.py)
RUTA_INDICE  = "indice.faiss"
RUTA_CHUNKS  = "chunks.pkl"
TOP_K        = 3          # Número de fragmentos a recuperar del índice
TEMPERATURA  = 0.2        # Temperatura baja → salida determinista
MODELO_LLM   = "dolphin3.0-llama3.1-8b"      # Nombre del modelo en LM Studio
MODELO_EMBED = "text-embedding-bge-small-en-v1.5"

# Cliente único compartido (LM Studio expone una API compatible con OpenAI)
_cliente = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

# --- Carga del índice FAISS persistido ---
def _cargar_indice_faiss():
    """Devuelve (indice, chunks) desde disco, o (None, []) si no existen."""
    if not os.path.exists(RUTA_INDICE) or not os.path.exists(RUTA_CHUNKS):
        print(
            f"[AVISO] No se encontraron '{RUTA_INDICE}' / '{RUTA_CHUNKS}'.\n"
            "Ejecuta vector_db.py primero para construir el índice."
        )
        return None, []
    indice = faiss.read_index(RUTA_INDICE)
    with open(RUTA_CHUNKS, "rb") as f:
        chunks = pickle.load(f)
    print(f"[Fase 1] Índice FAISS cargado: {indice.ntotal} vectores | {len(chunks)} chunks.")
    return indice, chunks

indice_faiss, base_chunks = _cargar_indice_faiss()

# ---------------------------------------------------------------------------
# Vocabulario y configuración de pre-procesado (sin cambios respecto al original)
# ---------------------------------------------------------------------------

VOCABULARIO_CORE = [
    "hypertrophy", "protein", "biceps", "triceps", "exercise",
    "dumbbell", "creatine", "supplement", "squats", "calories",
    "testosterone", "weight", "stretch", "deficit", "muscle",
    "volume", "frequency", "barbell", "failure", "recovery", "nutrition",
    "chest", "legs", "arms", "back", "glutes", "routine", "training", "plan",
    "sets", "reps", "cardio", "strength", "endurance", "flexibility",
]

EXPANSION_QUERY = {
    "chest":       ["chest", "pectorals", "pecs"],
    "legs":        ["legs", "quads", "hamstrings", "calves"],
    "arms":        ["arms", "biceps", "triceps", "forearms"],
    "back":        ["back", "lats", "rhomboids"],
    "glute":       ["glutes", "gluteus"],
    "hypertrophy": ["hypertrophy", "growth", "size", "bulk"],
    "routine":     ["routine", "plan", "program", "schedule", "training"],
}

# Intentamos cargar spaCy una sola vez
try:
    _nlp = spacy.load("en_core_web_sm") if spacy else None
except Exception:
    _nlp = None

# ---------------------------------------------------------------------------
# Utilidades de pre-procesado
# ---------------------------------------------------------------------------

def preprocesar_input(texto: str) -> str:
    """Limpia, corrige ortografía y expande la query del usuario."""
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9 ]", "", texto)
    texto = " ".join(texto.split())

    palabras = texto.split()
    palabras_corregidas = []
    for palabra in palabras:
        if process and fuzz:
            match = process.extractOne(
                palabra, VOCABULARIO_CORE, scorer=fuzz.ratio, score_cutoff=80
            )
            palabras_corregidas.append(match[0] if match else palabra)
        else:
            palabras_corregidas.append(palabra)

    texto_corregido = " ".join(palabras_corregidas)

    if _nlp:
        doc = _nlp(texto_corregido)
        tokens = [t.lemma_ for t in doc if not t.is_stop and t.text.strip()]
        texto_corregido = " ".join(tokens)

    palabras_finales = texto_corregido.split()
    query_expandida: list[str] = []
    for p in palabras_finales:
        query_expandida.append(p)
        if p in EXPANSION_QUERY:
            query_expandida.extend(EXPANSION_QUERY[p])

    resultado_final = " ".join(
        sorted(set(query_expandida), key=query_expandida.index)
    )

    print(f"\n[DEBUG PREPROCESADO]")
    print(f"  Usuario: '{texto}'")
    print(f"  Motor:   '{resultado_final}'\n")
    return resultado_final

# ---------------------------------------------------------------------------
# [Fase 3: Búsqueda Vectorial (Retrieval)] - Embedding del prompt del usuario
#   y extracción del Top-K de FAISS.
# ---------------------------------------------------------------------------

def _obtener_embedding(texto: str) -> np.ndarray | None:
    """Llama a LM Studio para vectorizar un texto."""
    try:
        respuesta = _cliente.embeddings.create(
            input=[texto],
            model=MODELO_EMBED,
        )
        return np.array([respuesta.data[0].embedding], dtype=np.float32)
    except Exception as e:
        print(f"[ERROR Fase 3] Fallo al obtener embedding: {e}")
        return None


def recuperar_contexto(pregunta_procesada: str) -> str:
    """Devuelve los TOP_K fragmentos más relevantes del índice FAISS."""
    if indice_faiss is None or not base_chunks:
        return ""

    vector = _obtener_embedding(pregunta_procesada)
    if vector is None:
        return ""

    distancias, indices = indice_faiss.search(vector, TOP_K)
    fragmentos = [
        base_chunks[i] for i in indices[0] if i != -1 and i < len(base_chunks)
    ]
    contexto = "\n---\n".join(fragmentos)
    print(f"[Fase 3] {len(fragmentos)} fragmentos recuperados de FAISS.\n")
    return contexto

# ---------------------------------------------------------------------------
# [Fase 4: Router y Augmented Prompting] - Construcción del System Prompt
#   instruyendo al LLM a decidir la intención (routine vs chat) basándose en
#   la pregunta, inyectando los datos del usuario y los fragmentos recuperados.
# ---------------------------------------------------------------------------

def construir_system_prompt(datos_usuario: dict, contexto: str) -> str:
    """
    Genera el System Prompt que obliga al LLM a:
      - Detectar la intención (routine | chat).
      - Responder SIEMPRE con el JSON schema definido.
    """
    nombre    = datos_usuario.get("nombre", "Usuario")
    objetivo  = datos_usuario.get("objetivo", "hipertrofia muscular")
    nivel     = datos_usuario.get("nivel", "intermedio")
    dias_disp = datos_usuario.get("dias_disponibles", 4)
    peso_kg   = datos_usuario.get("peso_kg", "N/A")
    altura_cm = datos_usuario.get("altura_cm", "N/A")

    return f"""
You are "Kaizen Bot", an elite AI personal trainer specialized in hypertrophy and fitness.

## USER PROFILE
- Name: {nombre}
- Goal: {objetivo}
- Level: {nivel}
- Available training days per week: {dias_disp}
- Weight: {peso_kg} kg | Height: {altura_cm} cm

## KNOWLEDGE BASE (retrieved context)
{contexto if contexto else "No specific context retrieved. Use your general knowledge."}

## YOUR TASK — INTENT ROUTER
Analyse the user's message and decide which intent it belongs to:
  • "routine" → The user is asking to CREATE, GENERATE or GET a training plan/routine/program.
  • "chat"    → Any other question: theory, nutrition, technique, motivation, general advice.

## OUTPUT RULES — CRITICAL
You MUST reply with a single, valid JSON object and NOTHING ELSE outside the JSON.
The JSON must follow this exact schema:

{{
  "intent": "routine" | "chat",
  "message": "<Detailed response or routine explanation in the same language the user used>",
  "routine_data": {{
    "days": [
      {{
        "day": "<e.g. Monday>",
        "muscle_groups": ["<group1>", "<group2>"],
        "exercises": [
          {{
            "name": "<Exercise name>",
            "sets": <number>,
            "reps": "<e.g. 8-12>",
            "rest_seconds": <number>,
            "notes": "<Optional technique tip>"
          }}
        ]
      }}
    ]
  }} | null
}}

Rules:
1. If intent == "chat"  → routine_data MUST be null.
2. If intent == "routine" → routine_data MUST contain the full plan structure above.
3. Always answer in the SAME language the user wrote in.
4. Base your answer on the retrieved context when available; do NOT invent facts.
5. Keep "message" concise but informative (under 250 words).
6. Do NOT include any text, markdown, or explanation outside the JSON object.
7. OUTPUT RAW JSON ONLY. Do NOT wrap the JSON in ```json markdown blocks. Just output the curly braces.
""".strip()

# ---------------------------------------------------------------------------
# [Fase 5: Generación JSON Condicional] - Llamada al LLM forzando salida JSON.
#   Parseo del resultado y simulación de cómo el backend lo procesaría.
# ---------------------------------------------------------------------------

def llamar_llm_json(system_prompt: str, pregunta_usuario: str) -> dict:
    """
    Llama al LLM con response_format JSON y devuelve el dict parseado.
    En caso de error de parseo devuelve un dict de error estructurado.
    """
    try:
        respuesta = _cliente.chat.completions.create(
            model=MODELO_LLM,
            messages=[
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": pregunta_usuario},
            ],
            temperature=TEMPERATURA,
        )
        contenido_raw = respuesta.choices[0].message.content
        print(f"[Fase 5] RAW del LLM:\n{contenido_raw}\n")
        contenido_limpio = contenido_raw.strip()
        if contenido_limpio.startswith("```"):
            lineas = contenido_limpio.split("\n")
            if lineas[0].startswith("```"):
                lineas = lineas[1:]
            if lineas[-1].startswith("```"):
                lineas = lineas[:-1]
            contenido_limpio = "\n".join(lineas).strip()
        
        resultado = json.loads(contenido_limpio)
        return resultado

    except json.JSONDecodeError as e:
        print(f"[ERROR Fase 5] No se pudo parsear el JSON del LLM: {e}")
        return {
            "intent": "chat",
            "message": "Lo siento, hubo un problema al procesar mi respuesta. Por favor, inténtalo de nuevo.",
            "routine_data": None,
        }
    except Exception as e:
        print(f"[ERROR Fase 5] Fallo en la llamada al LLM: {e}")
        return {
            "intent": "chat",
            "message": f"Error de conexión con LM Studio. ¿Está el servidor local encendido? Detalle: {e}",
            "routine_data": None,
        }


def simular_procesado_backend(resultado: dict) -> None:
    """
    Simula cómo el backend Java (Kaizen) procesaría la respuesta del bot.
    En producción esto sería la lógica del WorkoutService, etc.
    """
    intent = resultado.get("intent", "desconocido")

    print("\n" + "─" * 60)
    print("  [Simulación Backend Kaizen]")
    print("─" * 60)

    if intent == "routine":
        print("  → Intención detectada: ROUTINE")
        print("  → El backend crearía un nuevo TrainingPlan en BD.")
        dias = resultado.get("routine_data", {}).get("days", [])
        for dia in dias:
            print(f"     • {dia.get('day')} — {', '.join(dia.get('muscle_groups', []))}")
        print("  → Respuesta enviada al frontend Android (WorkoutsScreen).")
    else:
        print("  → Intención detectada: CHAT")
        print("  → El backend reenviaría el mensaje al hilo de chat del usuario.")

    print("─" * 60 + "\n")


def procesar_peticion(pregunta: str, datos_usuario: dict) -> dict:
    """
    Punto de entrada principal del módulo.
    Ejecuta el pipeline RAG completo y devuelve el dict de respuesta.
    """
    # Fase 3
    pregunta_limpia = preprocesar_input(pregunta)
    contexto        = recuperar_contexto(pregunta_limpia)
    print(f"[DEBUG - Contexto FAISS]:\n{contexto}\n")

    # Fase 4
    system_prompt = construir_system_prompt(datos_usuario, contexto)

    # Fase 5
    resultado = llamar_llm_json(system_prompt, pregunta)
    simular_procesado_backend(resultado)

    return resultado