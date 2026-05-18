# app.py — Flask web interface for Kaizen Bot.
# Run with:  python app.py

from flask import Flask, render_template, request, jsonify
from bot_logic import procesar_peticion
from layers import USUARIO_MOCK

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", usuario=USUARIO_MOCK)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    mensaje = data.get("message", "").strip()
    if not mensaje:
        return jsonify({"error": "Empty message"}), 400
    resultado = procesar_peticion(mensaje, USUARIO_MOCK)
    return jsonify(resultado)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
