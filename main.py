# main.py
from openai import OpenAI
from vector_db import recuperar_chunks
from bot_logic import preprocesar_input, es_respuesta_valida, escalar_a_humano

# Configuramos la conexión a tu LM Studio local
# Fíjate que apunta al puerto 1234 que es el que usa por defecto
cliente_ai = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="api-key-no-necesaria")

def llamar_llm(contexto, pregunta_usuario):
    # Aquí definimos las reglas estrictas para el RAG
    prompt_sistema = f"""
    You are an elite personal trainer specialized in muscle hypertrophy.
    
    Context retrieved from our database:
    {contexto}
    
    RULES:
    1. Answer ONLY based on the context provided above.
    2. If the context does not contain the answer, reply EXACTLY with: "I do not have enough information."
    """
    
    try:
        # Llamada real al modelo que tienes cargado en LM Studio
        respuesta = cliente_ai.chat.completions.create(
            model="dolphin3.0-llama3.1-8b", # LM Studio ignora el nombre y usa el modelo que tengas cargado
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": pregunta_usuario}
            ],
            temperature=0.1 # Temperatura muy baja para que sea estricto y no invente cosas
        )
        return respuesta.choices[0].message.content
        
    except Exception as e:
        print(f"\n[ERROR DE CONEXIÓN] No se pudo hablar con LM Studio. ¿Está el Local Server encendido? Detalle: {e}")
        # Si falla la IA (ej. apagaste LM Studio), forzamos que caiga en la regla de escalado
        return "I do not have enough information."

def manejar_pregunta(pregunta):
    limpio = preprocesar_input(pregunta)
    contexto = recuperar_chunks(limpio)
    # AÑADE ESTA LÍNEA PARA VER QUÉ LEE LA IA REALMENTE
    print(f"\n[DEBUG - LO QUE ENCONTRÓ FAISS]:\n{contexto}\n")
    
    # Llamamos a tu IA de verdad
    respuesta = llamar_llm(contexto, limpio)
    
    # Pasamos la respuesta por tu lógica de validación
    if not es_respuesta_valida(respuesta, contexto):
        return escalar_a_humano()
        
    return respuesta

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🧠 CONECTANDO CON LM STUDIO...")
    print("💪 BOT DE HIPERTROFIA INICIADO (Escribe 'salir' para terminar)")
    print("="*50)
    
    while True:
        usuario = input("\nTú: ")
        if usuario.lower() == 'salir':
            print("Bot: ¡A entrenar duro! Nos vemos.")
            break
            
        respuesta_bot = manejar_pregunta(usuario)
        print(f"Bot: {respuesta_bot}")