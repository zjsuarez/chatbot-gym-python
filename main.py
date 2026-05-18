# main.py — Entry point. Handles the User layer and the Input layer.
# Run with:  python main.py

from layers import USUARIO_MOCK, leer_entrada, mostrar_respuesta
from bot_logic import procesar_peticion

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🏋️  KAIZEN BOT — Simulación Local RAG")
    print(f"  Usuario: {USUARIO_MOCK['nombre']} | Nivel: {USUARIO_MOCK['nivel']}")
    print(f"  Objetivo: {USUARIO_MOCK['objetivo']}")
    print("  Escribe 'salir' para terminar.")
    print("=" * 60)

    while True:
        try:
            entrada = leer_entrada()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Sesión terminada. ¡A entrenar!")
            break

        if not entrada:
            continue

        if entrada.lower() in {"salir", "exit", "quit"}:
            print("Bot: ¡A entrenar duro! Nos vemos. 💪")
            break

        resultado = procesar_peticion(entrada, USUARIO_MOCK)
        mostrar_respuesta(resultado)
