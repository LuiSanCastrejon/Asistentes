import wikipedia

def buscar_wikipedia(consulta):
    wikipedia.set_lang("es")

    try:
        busqueda = wikipedia.search(consulta)
        if not busqueda:
            return "No encontré información sobre ese tema."

        respuesta = wikipedia.summary(busqueda[0], sentences=3)
        return respuesta

    except wikipedia.exceptions.DisambiguationError as e:
        opciones = ", ".join(e.options[:5])
        return f"El término es ambiguo. Algunas opciones son: {opciones}"

    except wikipedia.exceptions.PageError:
        return "No encontré una página específica sobre ese tema."

    except Exception as e:
        print("Error en Wikipedia:", e)
        return "Ocurrió un error al consultar Wikipedia."