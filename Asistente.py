import ollama
from piper.voice import PiperVoice
import numpy as np
import wave
from playsound import playsound 
import sounddevice as sd
import soundfile as sf

#Cargar Moldeo TTS y Ollama
client = ollama.Client()
model_path = "voices/en_US-amy-low.onnx"
config_path = "voices/en_US-amy-low.onnx.json"
voice = PiperVoice.load(model_path, config_path=config_path)

#Ollama
model = "llama2"
prompt = "Hello, What is your name? "

response = client.generate(model=model, prompt=prompt)
texto = response["response"]
print(texto)

output_filename = "Ollama.wav"

with wave.open(output_filename, "wb") as wav_file:
    voice.synthesize_wav(texto, wav_file)

data, fs = sf.read(output_filename)
sd.play(data, fs, device=11)
sd.wait()