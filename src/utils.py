def build_prompt(context, question, question_type):
   type_instructions = {
        "yes_no": "Responde únicamente con la palabra exacta 'sí' o 'no' (minúscula, sin acentos, sin puntuación, sin texto adicional). Ejemplo: 'sí'",
        "short_answer": "Responde con una frase corta y concisa de máximo 15 palabras.",
        "open_ended": "Responde detalladamente pero sin superar las 200 palabras.",
    }

    return f"""
CONTEXTO:
{context}

PREGUNTA:
{question}

OBJETIVO:
Eres un asistente que responde preguntas basándose EXCLUSIVAMENTE en el contexto anterior.
- Si no encuentras la respuesta en el contexto, di "No lo sé".
- Si la pregunta no está relacionada, di "No estoy seguro".
- Si la pregunta no es clara, di "No estoy seguro".

TIPO DE PREGUNTA: {question_type}
{type_instructions.get(question_type, "")}

IMPORTANTE: Tu respuesta debe estar en español, sin mezclar con inglés.
RESPONDE AHORA:
"""
