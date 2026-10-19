# Servidor web local para Atenea
# Uso:  python servidor_web.py   y abrir http://localhost:5000 en el navegador
# Requiere: pip install flask
import io
import re
import wave
import base64
import threading

from flask import Flask, request, jsonify, send_from_directory

# Importar Atenea carga los modelos (Whisper, Piper, LLM, skills) pero no arranca
# el bucle de la wakeword, que solo corre al ejecutar Atenea.py directamente
import Atenea

#________________________________________________________________________________
#Configuración
HOST = "127.0.0.1"   # Solo accesible desde esta computadora
PUERTO = 5000

app = Flask(__name__, static_folder=None)
# El LLM no admite varias peticiones a la vez: se atienden de una en una
candado = threading.Lock()

#________________________________________________________________________________
#Funciones

def sintetizar(mensaje):
    """Convierte el texto a voz con Piper y devuelve el WAV en base64."""
    mensaje = re.sub(r"[*#`]", "", mensaje).replace("_", " ")
    if not mensaje.strip():
        return None
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        Atenea.voice.synthesize_wav(mensaje, wav_file)
    return base64.b64encode(buffer.getvalue()).decode("ascii")

def responder(texto_usuario):
    print("Usuario (web):", texto_usuario)
    respuesta = Atenea.procesar_orden(texto_usuario)
    print("Asistente (web):", respuesta)
    try:
        audio = sintetizar(respuesta)
    except Exception as e:
        print("Error al generar la voz:", e)
        audio = None
    return {"usuario": texto_usuario, "respuesta": respuesta, "audio": audio}

#________________________________________________________________________________
#Rutas

@app.route("/")
def inicio():
    return send_from_directory(Atenea.DIRECTORIO + "/web", "index.html")

@app.route("/texto", methods=["POST"])
def texto():
    texto_usuario = (request.get_json(silent=True) or {}).get("texto", "").strip()
    if not texto_usuario:
        return jsonify({"error": "No escribiste nada."}), 400
    with candado:
        return jsonify(responder(texto_usuario))

@app.route("/audio", methods=["POST"])
def audio():
    archivo = request.files.get("audio")
    if archivo is None:
        return jsonify({"error": "No llegó el audio."}), 400
    with candado:
        # faster-whisper decodifica el audio del navegador (webm/ogg) por sí mismo
        try:
            segmentos, _ = Atenea.model_whisper.transcribe(
                io.BytesIO(archivo.read()), language="es"
            )
            texto_usuario = "".join(s.text for s in segmentos).strip()
        except Exception as e:
            print("Error al transcribir:", e)
            return jsonify({"error": "No pude leer el audio."}), 400
        if not texto_usuario:
            return jsonify({"error": "No entendí lo que dijiste."}), 400
        return jsonify(responder(texto_usuario))

#________________________________________________________________________________
if __name__ == "__main__":
    print(f"Atenea web en http://localhost:{PUERTO}")
    app.run(host=HOST, port=PUERTO, threaded=True)
