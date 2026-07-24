import ollama
from piper.voice import PiperVoice
import numpy as np
import wave
from playsound import playsound 
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.io.wavfile import write

#Cargar Moldeo Olama
client = ollama.Client()

#Cargar Moldeo TTS
model_path = "voices/en_US-amy-low.onnx"
config_path = "voices/en_US-amy-low.onnx.json"
voice = PiperVoice.load(model_path, config_path=config_path)

#Cargar whisper
fs_whisper = 16000
seconds = 3
model_whisper = WhisperModel("base", device="cpu")

#Hablar
print("Habla...")
audio = sd.rec(int(seconds * fs_whisper), samplerate=fs_whisper, channels=1)
sd.wait()
audio = np.squeeze(audio)


segments, _ = model_whisper.transcribe(audio,language="en")
texto_usuario = "Hello"
texto_usuario = "".join(s.text for s in segments).strip()
print("Texto usuario: ", texto_usuario)

#Ollama
model_LLM = "llama2"
response = client.generate(model=model_LLM, prompt=texto_usuario)
texto = response["response"]
print(texto)

#Audio
output_filename = "Ollama.wav"
with wave.open(output_filename, "wb") as wav_file:
    voice.synthesize_wav(texto, wav_file)

data, fs = sf.read(output_filename)
sd.play(data, fs, device=11)
sd.wait()