import spotipy
from spotipy.oauth2 import SpotifyOAuth

#__________________________
def reproduce():
        spotify.start_playback()

def pausa():
    spotify.pause_playback()

def siguiente():
    spotify.next_track()

def anterior():
    spotify.previous_track()

def cancion_actual():
    data = spotify.current_playback()
    if data and data["item"]:
        song = data["item"]
        return {
            "cancion": song["name"],
            "artista": song["artists"][0]["name"],
            "album": song["album"]["name"]
        }
    return None

def playlists():
    playlists = spotify.current_user_playlists(limit=50)
    for playlist in playlists["items"]:
        print(f"{playlist['name']} -> {playlist['id']}")

def dispositivos():
    dispositivos = spotify.devices()

    for dispositivo in dispositivos["devices"]:
        print(
            f"Nombre: {dispositivo['name']}"
            f"\nTipo: {dispositivo['type']}"
            f"\nID: {dispositivo['id']}"
            f"\nActivo: {dispositivo['is_active']}"
            f"\nPuede reproducir: {dispositivo['is_restricted']}"
            f"\n"
        )

def estado_reproduccion():
    estado = spotify.current_playback()

    if estado is None:
        return "Apagado"

    if estado["is_playing"]:
        return "Reproduciendo"
    else:
        return "Pausado"
#__________________________

SCOPE = (
    "playlist-read-private "
    "playlist-read-collaborative "
    "user-read-playback-state "
    "user-modify-playback-state "
    "user-read-currently-playing"
)

spotify = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id="eb9a997c0e3e462fb8986278f947b3fd",
        client_secret="f5917580b40b439b8c467e6278d59023",
        redirect_uri="http://127.0.0.1:8888/callback",
        scope=SCOPE
    )
)

#____________________________
estado=estado_reproduccion()
if estado=="Reproduciendo":
    pausa()
elif estado=="Pausado":
    reproduce()

cancion=cancion_actual()
print(cancion)