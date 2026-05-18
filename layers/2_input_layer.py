# Input layer — raw stdin reading and formatted console output.

import json


def leer_entrada() -> str:
    return input("\nTú: ").strip()


def mostrar_respuesta(resultado: dict) -> None:
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
