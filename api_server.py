
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from chatbot import Chatbot

app = Flask(__name__)
CORS(app)  # em produção, restrinja: CORS(app, origins=["https://seusite.com.br"])
bot = Chatbot()


@app.get("/")
def index():
    return send_from_directory(".", "widget.html")


@app.get("/greeting")
def greeting():
    return jsonify(reply=bot.greeting())


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    text = str(data.get("message", "")).strip()[:1000]
    if not text:
        return jsonify(error="mensagem vazia"), 400
    sid = "web-" + str(data.get("session_id", "anon"))[:64]
    return jsonify(reply=bot.respond(text, sid))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
