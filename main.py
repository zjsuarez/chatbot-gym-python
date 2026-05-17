# main.py — Punto de entrada de la simulación local del chatbot RAG "Kaizen".
# Ejecutar con:  python main.py

import json
from bot_logic import procesar_peticion

# ---------------------------------------------------------------------------
# [Fase 2: Simulación de Entrada] - Definición de un DTO de usuario mockeado
#   y la entrada de texto por consola.
# ---------------------------------------------------------------------------

# DTO mockeado que simula los datos que llegarían desde el backend Kaizen
# (en producción estos datos vendrían del UserService/JWT del usuario autenticado)
USUARIO_MOCK = {
    "id":               "usr_kaizen_001",
    "nombre":           "Alex",
    "objetivo":         "hipertrofia muscular",
    "nivel":            "intermedio",
    "dias_disponibles": 4,
    "peso_kg":          78,
    "altura_cm":        181,
    "restricciones":    [],          # p.ej. ["sin sentadilla", "lesión hombro"]
}


def mostrar_respuesta(resultado: dict) -> None:
    """Formatea e imprime la respuesta JSON del bot de forma legible en consola."""
    intent = resultado.get("intent", "desconocido")

    print("\n" + "═" * 60)
    print(f"  🤖 KAIZEN BOT  |  intent: [{intent.upper()}]")
    print("═" * 60)
    print(f"\n💬 Mensaje:\n  {resultado.get('message', '')}\n")

    if intent == "routine" and resultado.get("routine_data"):
        dias = resultado["routine_data"].get("days", [])
        print("📋 Rutina generada:")
        for dia in dias:
            print(f"\n  🗓️  {dia.get('day', '?')} — {', '.join(dia.get('muscle_groups', []))}")
            for ej in dia.get("exercises", []):
                print(
                    f"     • {ej.get('name')} — "
                    f"{ej.get('sets')} series × {ej.get('reps')} reps  "
                    f"(descanso: {ej.get('rest_seconds', '?')}s)"
                )
                if ej.get("notes"):
                    print(f"       ℹ️  {ej['notes']}")

    print("\n📦 JSON completo (para el backend):")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    print("═" * 60 + "\n")


# ---------------------------------------------------------------------------
# Bucle principal de simulación
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🏋️  KAIZEN BOT — Simulación Local RAG")
    print(f"  Usuario: {USUARIO_MOCK['nombre']} | Nivel: {USUARIO_MOCK['nivel']}")
    print(f"  Objetivo: {USUARIO_MOCK['objetivo']}")
    print("  Escribe 'salir' para terminar.")
    print("=" * 60)

    while True:
        try:
            entrada = input("\nTú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Sesión terminada. ¡A entrenar!")
            break

        if not entrada:
            continue

        if entrada.lower() in {"salir", "exit", "quit"}:
            print("Bot: ¡A entrenar duro! Nos vemos. 💪")
            break

        # Ejecutar el pipeline RAG completo (Fases 3-5 en bot_logic.py)
        resultado = procesar_peticion(entrada, USUARIO_MOCK)
        mostrar_respuesta(resultado)