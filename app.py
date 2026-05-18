# app.py — Flask web interface for Kaizen Bot. Run with:  python app.py

from flask import Flask, render_template, request, jsonify
from bot.pipeline import procesar_peticion, recargar_indice
from bot.layers import USUARIO_MOCK, cargar_documentos, construir_y_guardar_indice
from bot.layers.config import RUTA_DOCUMENTOS

app = Flask(__name__, template_folder="web/templates")


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


@app.route("/build-index", methods=["POST"])
def build_index():
    try:
        chunks = cargar_documentos(RUTA_DOCUMENTOS)
        construir_y_guardar_indice(chunks)
        recargar_indice()
        return jsonify({"ok": True, "chunks": len(chunks)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
