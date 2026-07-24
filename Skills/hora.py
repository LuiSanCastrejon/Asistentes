from datetime import datetime

def obtener_hora():
    from datetime import datetime
    hora = datetime.now().strftime("%H:%M")
    return f"Son las {hora}"