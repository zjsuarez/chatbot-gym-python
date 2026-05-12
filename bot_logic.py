def preprocesar_input(texto):
    texto = texto.lower()
    texto = " ".join(texto.split())
    # Diccionario de typos comunes de gym en inglés para profesionalizar el preprocesamiento
    ERRORES = {
        "protien": "protein", 
        "hypertrofy": "hypertrophy", 
        "bicep": "biceps",
        "tricep": "triceps",
        "excersize": "exercise",
        "excercise": "exercise",
        "dumbell": "dumbbell",
        "dumble": "dumbbell",
        "creatin": "creatine",
        "suplement": "supplement",
        "squots": "squats",
        "calorys": "calories",
        "testosteron": "testosterone",
        "wheigt": "weight",
        "strech": "stretch",
        "deficet": "deficit",
        "mussle": "muscle",
        "musle": "muscle"
    }
    
    for mal, bien in ERRORES.items():
        texto = texto.replace(mal, bien)
    return texto

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