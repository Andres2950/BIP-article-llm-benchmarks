def build_prompt(context, question):
    return f"""

    CONTEXTO:
    {context}

    PREGUNTA:
    {question}

    OBJETIVO:
    Eres un asistente cuya responsabilidad es responder preguntas sobre el contexto dado.
    Debes usar exclusivamente la información del contexto proporcionado para responder la pregunta.
    Si no sabes la respuesta, simplemente di que no lo sabes. No intentes inventar una respuesta.
    Si la pregunta no está relacionada con el contexto, di que no estás seguro de la respuesta.
    Si la pregunta no está clara, simplemente di que no estás seguro de la respuesta.
    Si la pregunta no está relacionada con el contexto, di que no estás seguro de la respuesta.
    Hay tres tipos de preguntas que se te pueden hacer: sí o no, respuesta corta y respuesta abierta.
    Si detectas que es de sí o no, responde únicamente con "sí" o "no".
    Si detectas que es de respuesta corta, responde únicamente con una frase corta y concisa.
    Si detectas que es de respuesta abierta, responde detalladamente, pero no superes las 200 palabras.
    Da tu respuesta en español.
    Responde la pregunta basada en el contexto proporcionado.
    """
