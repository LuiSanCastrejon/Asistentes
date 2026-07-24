import ollama
from piper.voice import PiperVoice
import numpy as np
import wave
from playsound import playsound 
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.io.wavfile import write
import re

#Cargar Moldeo Olama
client = ollama.Client()

#Cargar Moldeo TTS
model_path = "voices/en_US-amy-low.onnx"
config_path = "voices/en_US-amy-low.onnx.json"
voice = PiperVoice.load(model_path, config_path=config_path)

#Cargar whisper
fs_whisper = 16000
seconds = 2
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
stream = client.generate(
    model="llama2",
    prompt=texto_usuario,
    stream=True
)
buffer = ""

#piper

for chunk in stream:
    text = chunk["response"]
    print(text, end="", flush=True)

    buffer += text

    if re.search(r"[.!?]\s", buffer):
        with wave.open("tmp.wav", "wb") as wav_file:
            voice.synthesize_wav(buffer, wav_file)
        audio, sr = sf.read("tmp.wav")
        sd.play(audio, sr)
        sd.wait()

        buffer = ""