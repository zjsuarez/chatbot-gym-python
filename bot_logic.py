import re
try:
    import spacy
    from rapidfuzz import process, fuzz
except ImportError:
    pass

# Intentamos cargar el modelo de NLP. Si falla, el usuario necesita descargarlo.
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None

# Vocabulario de la base de datos para corregir errores con RapidFuzz
VOCABULARIO_CORE = [
    "hypertrophy", "protein", "biceps", "triceps", "exercise", 
    "dumbbell", "creatine", "supplement", "squats", "calories", 
    "testosterone", "weight", "stretch", "deficit", "muscle",
    "volume", "frequency", "barbell", "failure", "recovery", "nutrition",
    "chest", "legs", "arms", "back", "glutes"
]

# Nivel 4: Diccionario para expansión de queries (Query Expansion)
EXPANSION_QUERY = {
    "chest": ["chest", "pectorals", "pecs"],
    "legs": ["legs", "quads", "hamstrings", "calves"],
    "arms": ["arms", "biceps", "triceps", "forearms"],
    "back": ["back", "lats", "rhomboids"],
    "glute": ["glutes", "gluteus"],
    "hypertrophy": ["hypertrophy", "growth", "size", "bulk"],
}

def preprocesar_input(texto):
    # Nivel 1: Minúsculas y limpieza básica de espacios
    texto = texto.lower()
    
    # Nivel 2a: Regex (solo letras, números y espacios)
    texto = re.sub(r"[^a-z0-9 ]", "", texto)
    texto = " ".join(texto.split())
    
    palabras = texto.split()
    palabras_corregidas = []
    
    # Nivel 2b: Corrección ortográfica aproximada con RapidFuzz
    for palabra in palabras:
        try:
            # Buscamos la palabra más parecida (score_cutoff=80 significa que debe parecerse al menos en un 80%)
            match = process.extractOne(palabra, VOCABULARIO_CORE, scorer=fuzz.ratio, score_cutoff=80)
            if match:
                palabras_corregidas.append(match[0])
            else:
                palabras_corregidas.append(palabra)
        except NameError: # Si rapidfuzz no está instalado aún
            palabras_corregidas.append(palabra)
            
    texto_corregido = " ".join(palabras_corregidas)
    
    # Nivel 3: NLP Avanzado con spaCy (Lematización y Stopwords)
    if nlp:
        doc = nlp(texto_corregido)
        tokens_limpios = []
        for token in doc:
            # Quitamos stopwords (is, the, at) y aplicamos lematización (gains -> gain)
            if not token.is_stop and token.text.strip():
                tokens_limpios.append(token.lemma_)
        texto_nlp = " ".join(tokens_limpios)
    else:
        texto_nlp = texto_corregido
        
    # Nivel 4: Expansión de Query (Query Expansion)
    palabras_finales = texto_nlp.split()
    query_expandida = []
    for p in palabras_finales:
        query_expandida.append(p)
        if p in EXPANSION_QUERY:
            query_expandida.extend(EXPANSION_QUERY[p])
            
    # Eliminamos duplicados manteniendo el orden
    resultado_final = " ".join(sorted(set(query_expandida), key=query_expandida.index))
    
    print(f"\n[DEBUG PREPROCESADO]")
    print(f"  Usuario: '{texto}'")
    print(f"  Motor:   '{resultado_final}'\n")
    
    return resultado_final

def es_respuesta_valida(respuesta, contexto):
    if len(contexto) == 0:
        return False
    if "not enough information" in respuesta.lower() or "no tengo información" in respuesta.lower():
        return False
    if len(respuesta) < 20:
        return False
    return True

def escalar_a_humano():
    return "⚠️ [ESCALADO A HUMANO] No he podido encontrar una respuesta segura en mi base de datos. Te transfiero con un entrenador certificado."