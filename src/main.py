import argparse
from pathlib import Path
import pandas as pd
import random
import secrets
from datetime import datetime
import gc
import torch

from langchain_community.document_loaders import PyPDFLoader

from benchmark import Benchmark
from evaluator import extraer_si_no
from wrappers import load_model_from_config, unload_model
from model_config import MODELS


# El CSV usa "yes/no"; el resto del pipeline (prompt, métricas) usa "yes_no".
TYPE_ALIASES = {
    "yes/no": "yes_no",
    "yes_no": "yes_no",
    "short_answer": "short_answer",
    "open_ended": "open_ended",
}

BAG_SIZE_PER_TYPE = 100 # 100 preguntas de cada tipo por bag
NUM_BAGS = 3 # 3 bags por modelo
SEED = 1234224714 # misma semilla en todas las corridas paralelas; None genera una al azar


RECORD_COLUMNS = [
    "bag_id", "model", "question_id", "expected_answer", "response",
    "total_duration_sec", "input_tokens", "output_tokens", "total_tokens",
    "question_type",
    "cpu_avg", "cpu_min", "cpu_max",
    "memory_avg", "memory_min", "memory_max",
    "gpu_util_avg", "gpu_util_min", "gpu_util_max",
    "gpu_mem_avg", "gpu_mem_min", "gpu_mem_max",
]

GGUF_BASE_PATH = "/data/edelgado/models"


def normalize_question_type(raw_type):
    key = str(raw_type).strip().lower()
    if key not in TYPE_ALIASES:
        raise ValueError(f"Tipo de pregunta desconocido en la columna Type: {raw_type!r}")
    return TYPE_ALIASES[key]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Models Benchmarking")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature to use")
    parser.add_argument("--context-docs", type=str, default="./context_docs",
                        help="Path to the context documents directory")
    parser.add_argument("--dataset", type=str, default="./datasets/dataset_normativas.csv",
                        help="Path to the CSV dataset file with columns: ID, Type, Question, Answer")
    parser.add_argument("--question_id", type=int, default=None,
                        help="Question ID in the csv dataset file (if not provided, all questions will be benchmarked)")
    return parser.parse_args()


# Esto ya no se esta usando, pero lo dejo por si es necesario hacer rollback
def load_context(dir_path):
    context = []
    routes = sorted(p for p in Path(dir_path).iterdir() if p.is_file() and p.suffix == ".pdf")
    for route in routes:
        loader = PyPDFLoader(str(route))
        pages = loader.load()
        text = "\n".join([p.page_content for p in pages])
        # No se como vaya si se trunca el texto, hay mucho contenido que se pierde
        # text = text[:15000] + "[truncated]..." if len(text) > 15000 else text
        context.append(f"### Document {route.name}\n{text}")
    return "\n\n".join(context)


def load_dataset(path, question_id=None):
    df = pd.read_csv(path)
    if "Type" not in df.columns:
        raise ValueError("El dataset debe incluir la columna Type")

    df["ID"] = df["ID"].astype(int)
    df["question_type"] = df["Type"].map(normalize_question_type)

    if question_id is not None:
        df = df[df["ID"] == question_id]
        if df.empty:
            raise ValueError(f"No hay pregunta con ID {question_id}")
        return df.to_dict("records")

    groups = {}
    for tipo, group in df.groupby("question_type", sort=False):
        groups[tipo] = group.to_dict("records")
    return groups


def resolve_seed(seed):
    if seed is None:
        seed = secrets.randbits(32)
    random.seed(seed)
    return seed


def build_bags(question_groups):
    bags = []
    for _ in range(NUM_BAGS):
        sampled_questions = []
        for q_list in question_groups.values():
            sampled_questions.extend(random.choices(q_list, k=BAG_SIZE_PER_TYPE))
        bags.append(sampled_questions)
    return bags


if __name__ == "__main__":
    args = parse_arguments()
    question_groups = load_dataset(args.dataset, args.question_id)

    # Si es modo de una sola pregunta
    if isinstance(question_groups, list):
        print("Single questin mode")
        row = question_groups[0]
        question_type = row["question_type"]
        for model_cfg in MODELS:
            print(f"Loading model {model_cfg.name}")
            wrapper = load_model_from_config(model_cfg, args.temperature)
            print(f"Benchmarking model {model_cfg.name} with question {row['ID']}")
            question = row["Question"]
            benchmark = Benchmark(model_cfg.name, wrapper, args.context_docs, question, question_type)
            result = benchmark.run_question()
            benchmark.print_question_result(result)
            unload_model(wrapper)
        exit()

    # Modo completo: mismas bags para todos los modelos
    seed = resolve_seed(SEED)
    bags = build_bags(question_groups)
    print(f"Seed: {seed}")

    csv_path = f"./out/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(columns=RECORD_COLUMNS).to_csv(csv_path, index=False)
    print(f"Saving results incrementally to {csv_path}")

    # Iterar sobre los modelos definidos en model_config (ya ordenados por VRAM)
    for model_cfg in MODELS:
        print(f"Loading model {model_cfg.name}")
        wrapper = load_model_from_config(model_cfg, args.temperature)

        for bag_id, sampled_questions in enumerate(bags):
            for row in sampled_questions:
                question = row["Question"]
                question_type = row["question_type"]
                expected_answer = row["Answer"]
                question_id = row["ID"]

                benchmark = Benchmark(model_cfg.name, wrapper, args.context_docs, question, question_type)
                result = benchmark.run_question()

                response_text = result["response"]
                
                if question_type == "yes_no":
                    extracted = extraer_si_no(response_text)
                    if extracted is not None:
                        response_text = "sí" if extracted else "no"


                record = {
                    "bag_id": bag_id,
                    "model": model_cfg.name,
                    "question_id": question_id,
                    "expected_answer": expected_answer,
                    "response": response_text,
                    "total_duration_sec": result["total_duration"] / 1e9,
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "total_tokens": result["total_tokens"],
                    "question_type": question_type
                }

                for resource in ['cpu', 'memory', 'gpu_util', 'gpu_mem']:
                    if result.get(resource) is not None:
                        record[f"{resource}_avg"] = result[resource]["avg"]
                        record[f"{resource}_min"] = result[resource]["min"]
                        record[f"{resource}_max"] = result[resource]["max"]
                    else:
                        record[f"{resource}_avg"] = None
                        record[f"{resource}_min"] = None
                        record[f"{resource}_max"] = None

                pd.DataFrame([record], columns=RECORD_COLUMNS).to_csv(
                    csv_path, mode="a", header=False, index=False
                )
                print(f"Question {question_id} | bag {bag_id} | model {model_cfg.name} | saved to {csv_path}")

        unload_model(wrapper)
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Raw data saved to {csv_path}")
    #print("Evaluando Métricas") # Evaluacion se deja para despues porque parece que no van a caber todos los modelos en una sola corrida
    #ejecutar_evaluacion(csv_path)