import argparse
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader

from benchmark import Benchmark


def parse_arguments():
    parser = argparse.ArgumentParser(description="Models Benchmarking")

    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature to use")
    parser.add_argument("--context-docs", type=str, default="./context_docs", help="Path to the context documents directory")
    parser.add_argument("--question", type=str, default="Haz un resumen del contexto dado", help="Question to ask the model")
    parser.add_argument("--show-response", action="store_true", help="Show the response of the model")

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

if __name__ == "__main__":

    models = [
        "qwen2.5-coder:3b",
        "llama3.2:3b"
    ]

    args = parse_arguments()
    context = load_context(args.context_docs)

    for model in models:
        print("#"*10 + f" MODEL: {model} " + "#"*10)
        print("\n")
        benchmark = Benchmark(model, args.temperature, context, args.question, args.show_response)
        benchmark.print_question_result()