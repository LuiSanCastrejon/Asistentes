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
#VAD
import webrtcvad
#Skills
import json
import re
import os
import inspect
import importlib
from collections import deque

#________________________________________________________________________________
#Configuración
DIRECTORIO = os.path.dirname(os.path.abspath(__file__))
RUTA_SKILLS = os.path.join(DIRECTORIO, "skills.json")

RUTA_LLM = "models/qwen3-4b-q4_k_m.gguf"
RUTA_VOZ = "/home/luis/Documents/LLM/voices/es_ES-carlfm-x_low.onnx"
RUTA_VOZ_CONFIG = "/home/luis/Documents/LLM/voices/es_ES-carlfm-x_low.onnx.json"

DISPOSITIVO_SALIDA = 11   # Índice de la salida de audio (ver sd.query_devices())
WAKEWORD = "alexa"
UMBRAL_WAKEWORD = 0.6

SILENCIO_FIN = 0.7        # Segundos de silencio que marcan el fin de la orden
ESPERA_MAX = 5.0          # Segundos máximos esperando a que empieces a hablar
DURACION_MAX = 15.0       # Duración máxima de una orden
VOZ_MINIMA = 0.25         # Segundos de voz mínimos para aceptar una orden (filtra ruidos)
PRE_BUFFER = 0.3          # Audio previo que se conserva para no cortar la primera sílaba

MAX_HISTORIAL = 10        # Mensajes que recuerda el asistente (sin contar el system)

#________________________________________________________________________________
#Skills: se definen en skills.json, no hay que tocar este código para añadirlas

def a_numero(valor):
    if isinstance(valor, bool):
        raise ValueError("no es un número")
    if isinstance(valor, (int, float)):
        return valor
    texto = str(valor).strip().replace(",", ".")
    numero = float(texto)
    return int(numero) if numero.is_integer() else numero

def a_entero(valor):
    numero = a_numero(valor)
    if float(numero).is_integer():
        return int(numero)
    raise ValueError("no es un entero")

def a_booleano(valor):
    if isinstance(valor, bool):
        return valor
    texto = str(valor).strip().lower()
    if texto in ("true", "si", "sí", "1", "verdadero"):
        return True
    if texto in ("false", "no", "0", "falso"):
        return False
    raise ValueError("no es sí/no")

# Tipos que se pueden indicar en skills.json con "tipo"
CONVERSORES = {
    "texto": lambda v: v if isinstance(v, str) else str(v),
    "numero": a_numero,
    "entero": a_entero,
    "booleano": a_booleano,
    "lista": lambda v: v if isinstance(v, list) else [v]
}
# Cómo se muestra cada tipo en el "Formato" del prompt del router
MARCADORES = {"texto": "...", "numero": 0, "entero": 0, "booleano": True, "lista": ["..."]}

def validar_firma(funcion, parametros):
    """Comprueba que los parámetros del JSON coinciden con los de la función.
    Devuelve un mensaje de error o None si todo está bien."""
    try:
        firma = inspect.signature(funcion)
    except (TypeError, ValueError):
        return None  # No se puede inspeccionar; se confía en el JSON

    nombrables = [
        p for p in firma.parameters.values()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    acepta_kwargs = any(
        p.kind == p.VAR_KEYWORD for p in firma.parameters.values()
    )

    if not acepta_kwargs:
        aceptados = {p.name for p in nombrables}
        sobrantes = [p for p in parametros if p not in aceptados]
        if sobrantes:
            return f"la función no acepta los parámetros {sobrantes}"

    for p in nombrables:
        if p.default is p.empty:
            if p.name not in parametros:
                return f"la función requiere '{p.name}' y no está declarado en el JSON"
            # Si la función lo exige, es obligatorio aunque el JSON diga lo contrario
            parametros[p.name]["obligatorio"] = True

    return None


def cargar_skills(ruta):
    """Lee skills.json, importa cada función y valida sus parámetros."""
    with open(ruta, encoding="utf-8") as f:
        config = json.load(f)

    skills = {}
    for nombre, info in config.items():
        nombre = nombre.strip().lower()
        try:
            modulo = importlib.import_module(info["modulo"])
            funcion = getattr(modulo, info["funcion"])
        except Exception as e:
            print(f"[skills] '{nombre}' no se cargó: {e}")
            continue

        # Un parámetro puede ser solo un texto (descripción) o un objeto completo
        parametros = {}
        for param, datos in info.get("parametros", {}).items():
            if isinstance(datos, str):
                datos = {"descripcion": datos}
            tipo = str(datos.get("tipo", "texto")).lower()
            if tipo not in CONVERSORES:
                print(f"[skills] '{nombre}': tipo '{tipo}' desconocido en '{param}', se usa texto")
                tipo = "texto"
            parametros[param] = {
                "descripcion": datos.get("descripcion", ""),
                "obligatorio": datos.get("obligatorio", True),
                "tipo": tipo
            }

        error = validar_firma(funcion, parametros)
        if error:
            print(f"[skills] '{nombre}' no se cargó: {error}")
            continue

        skills[nombre] = {
            "funcion": funcion,
            "descripcion": info.get("descripcion", ""),
            "parametros": parametros,
            "ejemplos": info.get("ejemplos", [])
        }
        print(f"[skills] '{nombre}' cargada")

    return skills


def construir_prompt_router(skills):
    """Genera el system prompt del router a partir de las skills cargadas."""
    lineas = [
        "/no_think",
        "Eres el router de un asistente de voz.",
        "Tu única función es decidir qué skill debe ejecutarse según lo que pide el usuario.",
        "Responde SOLO con un objeto JSON válido en una sola línea, sin texto adicional.",
        "Incluye solo los parámetros de la skill elegida y omite los opcionales que el usuario no mencione.",
        "",
        "Skills disponibles:"
    ]

    for nombre, skill in skills.items():
        lineas.append("")
        lineas.append(f"- {nombre}: {skill['descripcion']}")
        formato = {"skill": nombre}
        for param, datos in skill["parametros"].items():
            necesidad = "obligatorio" if datos["obligatorio"] else "opcional"
            lineas.append(f"    {param} ({datos['tipo']}, {necesidad}): {datos['descripcion']}")
            formato[param] = MARCADORES.get(datos["tipo"], "...")
        lineas.append(f"  Formato: {json.dumps(formato, ensure_ascii=False)}")
        for ejemplo in skill["ejemplos"]:
            salida = {"skill": nombre, **ejemplo.get("argumentos", {})}
            lineas.append(
                f'  Ejemplo: "{ejemplo["frase"]}" -> {json.dumps(salida, ensure_ascii=False)}'
            )

    lineas += [
        "",
        "Si ninguna skill es necesaria (saludos, conversación o preguntas generales):",
        '{"skill": "ninguna"}',
        'Ejemplo: "hola, ¿cómo estás?" -> {"skill": "ninguna"}'
    ]
    return "\n".join(lineas)


def ejecutar_skill(nombre, datos):
    """Ejecuta una skill con cero o varios argumentos tomados del JSON del router."""
    skill = skills[nombre]

    # A veces el modelo anida los argumentos: {"skill": "x", "parametros": {...}}
    for clave in ("parametros", "argumentos", "args"):
        if isinstance(datos.get(clave), dict) and clave not in skill["parametros"]:
            datos = {**datos, **datos[clave]}

    argumentos = {}
    for param, info in skill["parametros"].items():
        valor = datos.get(param)
        if valor is None or (isinstance(valor, str) and valor.strip() in ("", "...")):
            continue
        try:
            argumentos[param] = CONVERSORES[info["tipo"]](valor)
        except (ValueError, TypeError):
            return f"No entendí el valor de {param}: {valor}."

    faltantes = [
        p for p, d in skill["parametros"].items()
        if d["obligatorio"] and p not in argumentos
    ]
    if faltantes:
        return "Me falta información para hacerlo: " + ", ".join(faltantes) + "."

    resultado = skill["funcion"](**argumentos)
    if resultado is None or str(resultado).strip() == "":
        return "Listo."
    return str(resultado)

#________________________________________________________________________________
#Funciones del asistente

def hablar(mensaje):
    if not mensaje:
        return
    # Quitar símbolos de markdown que el TTS leería en voz alta
    mensaje = re.sub(r"[*#`]", "", mensaje).replace("_", " ")
    output_filename = "Respuesta_modelo.wav"

    with wave.open(output_filename, "wb") as wav_file:
        voice.synthesize_wav(mensaje, wav_file)

    #Leer archivo
    data, fs = sf.read(output_filename)
    sd.play(data, fs, device=DISPOSITIVO_SALIDA)
    sd.wait()

# Capturar audio con WebRTC VAD
def capturar_audio():
    fs = 16000
    blocksize = 320          # 20 ms (WebRTC VAD acepta bloques de 10, 20 o 30 ms)
    duracion_bloque = blocksize / fs
    bloques_silencio = max(1, round(SILENCIO_FIN / duracion_bloque))
    bloques_espera = max(1, round(ESPERA_MAX / duracion_bloque))
    bloques_max = max(1, round(DURACION_MAX / duracion_bloque))
    bloques_voz_min = max(1, round(VOZ_MINIMA / duracion_bloque))

    print("Habla...")
    pre_buffer = deque(maxlen=max(1, round(PRE_BUFFER / duracion_bloque)))
    audio_total = []
    silencios = 0
    bloques_voz = 0
    bloques_leidos = 0
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
            bloques_leidos += 1
            voz = vad.is_speech(audio.tobytes(), sample_rate=fs)

            if not hablando:
                pre_buffer.append(audio)
                if voz:
                    hablando = True
                    bloques_voz = 1
                    audio_total.extend(pre_buffer)
                elif bloques_leidos >= bloques_espera:
                    print("No se detectó voz")
                    return None
                continue

            audio_total.append(audio)
            if voz:
                silencios = 0
                bloques_voz += 1
            else:
                silencios += 1
                if silencios >= bloques_silencio:
                    break

            if len(audio_total) >= bloques_max:
                print("Duración máxima alcanzada")
                break

    if bloques_voz < bloques_voz_min:
        print("Solo se detectó ruido")
        return None

    audio_total = np.concatenate(audio_total)
    # Convertir a float32 para Whisper
    audio_total = audio_total.astype(np.float32) / 32768.0
    return audio_total

def escuchar(audio, idioma="es"):
    if audio is None:
        return ""

    segments, _ = model_whisper.transcribe(audio, language=idioma)
    texto_usuario = "".join(s.text for s in segments).strip()
    return texto_usuario

def generar_respuesta(messages, max_tokens=128, temperature=0.7):
    respuesta = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )

    texto = respuesta["choices"][0]["message"]["content"] or ""
    # Quitar el bloque de razonamiento de Qwen3, aunque esté vacío o incompleto
    if "</think>" in texto:
        texto = texto.split("</think>")[-1]
    elif "<think>" in texto:
        texto = texto.split("<think>")[0]

    return texto.strip()

def extraer_json(texto):
    """Extrae el primer objeto JSON del texto, aunque venga con ```json o texto extra."""
    decodificador = json.JSONDecoder()
    inicio = texto.find("{")
    while inicio != -1:
        try:
            datos, _ = decodificador.raw_decode(texto, inicio)
            if isinstance(datos, dict):
                return datos
        except json.JSONDecodeError:
            pass
        inicio = texto.find("{", inicio + 1)
    return None

def conversar(texto_usuario):
    mensajes = messages_asistente + [{"role": "user", "content": texto_usuario}]
    respuesta = generar_respuesta(mensajes)
    if not respuesta:
        respuesta = "No sé qué responder a eso."

    # Solo se guarda en el historial si la generación salió bien
    messages_asistente.append({"role": "user", "content": texto_usuario})
    messages_asistente.append({"role": "assistant", "content": respuesta})

    # Conservar el system prompt y los últimos mensajes para no llenar el contexto
    if len(messages_asistente) > MAX_HISTORIAL + 1:
        del messages_asistente[1:-MAX_HISTORIAL]

    return respuesta

def esperar_wakeword():
    samplerate = 16000
    blocksize = 1280         # 80 ms, el tamaño que espera openWakeWord
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
            puntuacion = max(
                (v for k, v in pred.items() if WAKEWORD in k),
                default=0
            )

            if puntuacion > UMBRAL_WAKEWORD:
                print(f"Wake word detectada ({puntuacion:.2f})")
                model_wakeword.reset()
                return
#________________________________________________________________________________
#Inicialización

#Crear VAD
vad = webrtcvad.Vad(2)

#Inicializar el modelo de voz a texto
model_whisper = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

#Cargar modelo de texto a voz
voice = PiperVoice.load(RUTA_VOZ, config_path=RUTA_VOZ_CONFIG)

#Inicializar el modelo de wakeword (solo el que se usa)
model_wakeword = Model(wakeword_models=[WAKEWORD])

#Inicializar LLM
llm = Llama(
    model_path=RUTA_LLM,
    n_ctx=4096,
    n_threads=8,
    chat_format="chatml",
    verbose=False
)

#Cargar skills y generar el prompt del router
skills = cargar_skills(RUTA_SKILLS)

mensaje_router = {
    "role": "system",
    "content": construir_prompt_router(skills)
}

messages_asistente = [
    {
        "role": "system",
        "content":
        "/no_think\n"
        "Eres un asistente de un laboratorio de electrónica.\n"
        "Responde en español, de forma rápida, sencilla y directa.\n"
        "No des explicaciones innecesariamente largas.\n"
        "Tu respuesta se leerá en voz alta: no uses emojis, listas ni formato markdown.\n"
    }
]
#________________________________________________________________________________
#Bucle principal
try:
    while True:
        #Esperar a que el usuario diga la wakeword
        esperar_wakeword()
        #Escuchar la orden del usuario
        audio = capturar_audio()
        texto_usuario = escuchar(audio)
        print("Usuario:", texto_usuario)

        if not texto_usuario:
            continue

        try:
            #El router no necesita historial: solo el system prompt y la orden actual
            respuesta_router = generar_respuesta(
                [mensaje_router, {"role": "user", "content": texto_usuario}],
                max_tokens=128,
                temperature=0
            )
            print("Router:", repr(respuesta_router))

            #Filtrar la respuesta para ver si es una skill
            datos = extraer_json(respuesta_router)
            skill = str(datos.get("skill", "ninguna")).strip().lower() if datos else "ninguna"

            if skill in skills:
                respuesta_modelo = ejecutar_skill(skill, datos)
            else:
                if skill != "ninguna":
                    print(f"Skill desconocida '{skill}', responde el asistente")
                respuesta_modelo = conversar(texto_usuario)

        except Exception as e:
            print("Error procesando la orden:", e)
            respuesta_modelo = "Lo siento, ocurrió un error al procesar tu petición."

        print("Asistente:", respuesta_modelo)
        try:
            hablar(respuesta_modelo)
        except Exception as e:
            print("Error al reproducir la respuesta:", e)

except KeyboardInterrupt:
    print("\nAsistente detenido.")
