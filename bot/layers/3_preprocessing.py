# Preprocessing layer — clean, spell-correct, lemmatize, and expand the user query.

import re

try:
    import spacy
    from rapidfuzz import process, fuzz
except ImportError:
    spacy = None
    process = None
    fuzz = None

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

try:
    _nlp = spacy.load("en_core_web_sm") if spacy else None
except Exception:
    _nlp = None


def preprocesar_input(texto: str) -> str:
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
