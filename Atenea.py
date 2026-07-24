#LLM
from llama_cpp import Llama
#Texto a voz
from piper.voice import PiperVoice
import wave
import sounddevice as sd
import soundfile as sf
#Voz a texto
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np

def hablar (mensaje):
    output_filename = "Prueba_piper.wav"
    texto = mensaje

    with wave.open(output_filename, "wb") as wav_file:
        voice.synthesize_wav(texto, wav_file)

    #Leer archivo
    data, fs = sf.read(output_filename)
    sd.play(data, fs, device=11)
    sd.wait()

def escuchar (idioma="es"):
    fs_whisper = 16000
    seconds = 3

    #Grabar el audio del usuario con el microfono
    print("Habla...")
    audio = sd.rec(int(seconds * fs_whisper), samplerate=fs_whisper, channels=1)
    sd.wait()
    audio = np.squeeze(audio)

    #Transcribir el audio
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
        "role": "system",
        "content":
        "/no_think\n"
        "Eres un asistente de voz. "
        "Responde en español de forma breve y directa."
    }
]

while True:
    texto_usuario = escuchar()
    if not texto_usuario:
        continue

    messages.append({
        "role": "user",
        "content": texto_usuario
    })

    respuesta_modelo=generar_respuesta(messages)

    messages.append({
        "role": "assistant",
        "content": respuesta_modelo
    })

    print(respuesta_modelo)
    hablar(respuesta_modelo)