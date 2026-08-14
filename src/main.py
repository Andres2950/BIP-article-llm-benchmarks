import argparse
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader

def parse_arguments():
    parser = argparse.ArgumentParser(description="Models Benchmarking")

    parser.add_argument("--model", type=str, default="qwen2.5-coder:3b", help="Model to use")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature to use")
    parser.add_argument("--context-docs", type=str, default="./context_docs", help="Path to the context documents directory")
    parser.add_argument("--question", type=str, default="Haz un resumen del contexto dado", help="Question to ask the model")
    return parser.parse_args()

def load_context(dir):
    context = []
    routes = sorted(p for p in Path(dir).iterdir() if p.is_file() and p.suffix == ".pdf")
    for route in routes:
        loader = PyPDFLoader(str(route))
        pages = loader.load()
        text = "\n".join([p.page_content for p in pages])
        context.append(f"### Document {route.name}\n{text}")
    
    return "\n\n".join(context)


def build_prompt(context, question):
    return f"""
    Eres un asistente que puede responder preguntas sobre el contexto dado.
    Debes usar exclusivamente la información del contexto proporcionado para responder la pregunta.
    Si no sabes la respuesta, simplemente di que no lo sabes. No intentes inventar una respuesta.
    Si la pregunta no está relacionada con el contexto, di que no estás seguro de la respuesta.
    Si la pregunta no está clara, simplemente di que no estás seguro de la respuesta.
    Si la pregunta no está relacionada con el contexto, di que no estás seguro de la respuesta.

    CONTEXTO: 
    {context}

    PREGUNTA:
    {question}

    RESPUESTA:
    Responde la pregunta basada en el contexto proporcionado.
    """

if __name__ == "__main__":

    args = parse_arguments()

    llm = ChatOllama(model=args.model, temperature=args.temperature)
    context = load_context(args.context_docs)
    prompt = build_prompt(context, args.question)

    response = llm.invoke(prompt)
    
    print("\n" + "=" * 20)
    print("Respuesta:\n")
    print(response.content)

    print("\n" + "=" * 20)
    print("Modelo: ", response.response_metadata["model"])
    print("Duración total: ", response.response_metadata["total_duration"])
    print("Duración de carga: ", response.response_metadata["load_duration"])
    print("Tokens de input: ", response.response_metadata["input_tokens"])
    print("Tokens de output: ", response.response_metadata["output_tokens"])
    print("Tokens totales: ", response.response_metadata["total_tokens"])


