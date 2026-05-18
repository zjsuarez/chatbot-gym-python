# LLM layer — call the chat completions API, parse JSON, and simulate backend routing.

import json
from openai import OpenAI
from .config import BASE_URL, API_KEY, MODELO_LLM, TEMPERATURA

_cliente = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def llamar_llm_json(system_prompt: str, pregunta_usuario: str) -> dict:
    try:
        respuesta = _cliente.chat.completions.create(
            model=MODELO_LLM,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": pregunta_usuario},
            ],
            temperature=TEMPERATURA,
        )
        contenido_raw = respuesta.choices[0].message.content
        print(f"[LLM] RAW:\n{contenido_raw}\n")

        contenido_limpio = contenido_raw.strip()
        if contenido_limpio.startswith("```"):
            lineas = contenido_limpio.split("\n")
            if lineas[0].startswith("```"):
                lineas = lineas[1:]
            if lineas[-1].startswith("```"):
                lineas = lineas[:-1]
            contenido_limpio = "\n".join(lineas).strip()

        return json.loads(contenido_limpio)

    except json.JSONDecodeError as e:
        print(f"[ERROR LLM] No se pudo parsear el JSON: {e}")
        return {
            "intent": "chat",
            "message": "Lo siento, hubo un problema al procesar mi respuesta. Por favor, inténtalo de nuevo.",
            "routine_data": None,
        }
    except Exception as e:
        print(f"[ERROR LLM] Fallo en la llamada: {e}")
        return {
            "intent": "chat",
            "message": f"Error de conexión con LM Studio. ¿Está el servidor local encendido? Detalle: {e}",
            "routine_data": None,
        }


def simular_procesado_backend(resultado: dict) -> None:
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
