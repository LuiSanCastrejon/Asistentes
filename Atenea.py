#LLM
from llama_cpp import Llama
#Texto a voz
from piper.voice import PiperVoice
import wave
import soundfile as sf
#Voz a texto
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
#Wakeword
from openwakeword.model import Model
#Skills
from Skills.hora import obtener_hora
import json
#VAD
import webrtcvad

skills = {
    "hora": obtener_hora
}

def hablar (mensaje):
    output_filename = "Respuesta_modelo.wav"
    texto = mensaje

    with wave.open(output_filename, "wb") as wav_file:
        voice.synthesize_wav(texto, wav_file)

    #Leer archivo
    data, fs = sf.read(output_filename)
    sd.play(data, fs, device=11)
    sd.wait()

# Capturar audio con WebRTC VAD
def capturar_audio():
    fs = 16000
    blocksize = 320          # 20 ms
    silencio_limite = 40     # 10 bloques = 200 ms
    print("Habla...")
    audio_total = []
    silencios = 1
    hablando = False
    with sd.InputStream(
        samplerate=fs,
        channels=1,
        dtype=np.int16,
        blocksize=blocksize
    ) as stream:

        while True:
            audio, overflow = stream.read(blocksize)
            if overflow:
                print("Overflow")

            audio = np.squeeze(audio)
            voz = vad.is_speech(audio.tobytes(),sample_rate=fs)
            if voz:
                hablando = True
                silencios = 0
                audio_total.append(audio)

            elif hablando:
                audio_total.append(audio)
                silencios += 1
                if silencios >= silencio_limite:
                    break

    if len(audio_total) == 0:
        return None

    audio_total = np.concatenate(audio_total)
    # Convertir a float32 para Whisper
    audio_total = audio_total.astype(np.float32) / 32768.0
    return audio_total

def escuchar(audio, idioma="es"):
    if audio is None:
        return ""

    segments, _ = model_whisper.transcribe(audio,language=idioma)
    texto_usuario = "".join(s.text for s in segments).strip()
    return texto_usuario

def generar_respuesta(messages):
    respuesta = llm.create_chat_completion(
        messages=messages,
        max_tokens=128,
        temperature=0.7
    )

    texto = respuesta["choices"][0]["message"]["content"]

    if "</think>" in texto:
        texto = texto.split("</think>")[-1].strip()

    return texto

def esperar_wakeword():
    samplerate = 16000
    blocksize = 1280
    print("Esperando wake word...")

    with sd.InputStream(
        samplerate=samplerate,
        channels=1,
        dtype=np.int16,
        blocksize=blocksize
    ) as stream:

        while True:
            audio, overflow = stream.read(blocksize)
            if overflow:
                print("Overflow de audio")

            audio = np.squeeze(audio)
            pred = model_wakeword.predict(audio)
            print(f"Alexa: {pred['alexa']:.4f}")

            if pred["alexa"] > 0.6:
                print("Wake word detectada")
                model_wakeword.reset()
                return
#________________________________________________________________________________--
#Crear VAD
vad = webrtcvad.Vad(2)

#Inicializar el modelo de voz a texto
model_whisper = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

fs_whisper = 16000
seconds = 3

#Cargar modelo de texto a voz
model_path = "/home/luis/Documents/LLM/voices/es_ES-carlfm-x_low.onnx"
config_path = "/home/luis/Documents/LLM/voices/es_ES-carlfm-x_low.onnx.json"
voice = PiperVoice.load(model_path, config_path=config_path)

#Inicializar el modelo de wakeword
model_wakeword = Model()

#Inicializar LLM
llm = Llama(
    model_path="models/qwen3-4b-q4_k_m.gguf",
    n_ctx=4096,
    n_threads=8,
    chat_format="chatml",
    verbose=False
)

#Parametros LLM
messages = [
{
"role":"system",
"content":
"/no_think\n"
"Responde en español de forma breve y directa."

"""
Eres un asistente de voz.

Si el usuario pide una acción disponible,
responde SOLO con JSON.

Skills disponibles:
- hora

Ejemplo:
Usuario: ¿Qué hora es?
Respuesta:
{"skill":"hora"}

Si no hay skill:
{"skill":"ninguna","respuesta":"texto"}
"""
}
]
#------------------
#Ciclo principal
#-------------------
while True:
    #Esperar a que el usuario diga la wakeword
    esperar_wakeword()
    #Escuchar la orden del usuario
    audio = capturar_audio()
    texto_usuario = escuchar(audio)
    print("Usuario:", texto_usuario)
    if not texto_usuario:
        continue

    messages.append({
        "role": "user",
        "content": texto_usuario
    })

    #El modelo genera la respuesta
    respuesta_modelo=generar_respuesta(messages)

    #Filtrar la respuesta para ver si es una skill
    try:
        datos = json.loads(respuesta_modelo)

        if datos["skill"] != "ninguna":

            funcion = skills[datos["skill"]]
            respuesta_modelo = funcion()

        else:
            respuesta_modelo = datos["respuesta"]

    except:
        pass

    messages.append({
        "role": "assistant",
        "content": respuesta_modelo
    })

    print(respuesta_modelo)
    hablar(respuesta_modelo)