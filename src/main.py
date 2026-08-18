import argparse
from pathlib import Path
import pandas as pd
import random
from datetime import datetime


from langchain_community.document_loaders import PyPDFLoader

from benchmark import Benchmark, load_hf_model
from evaluator import ejecutar_evaluacion


TYPES_RANGES = {
    "yes_no": (1, 3), # real: 1 a 100 / smoke test: 1 a 3
    "short_answer": (101, 103), # real: 101 a 200 / smoke test: 101 a 103
    "open_ended": (201, 203) # real: 201 a 300 / smoke test: 201 a 203
}

BAG_SIZE_PER_TYPE = 1
NUM_BAGS = 2

RECORD_COLUMNS = [
    "bag_id",
    "model",
    "question_id",
    "expected_answer",
    "response",
    "total_duration_sec",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "question_type",
    "cpu_avg", "cpu_min", "cpu_max",
    "memory_avg", "memory_min", "memory_max",
    "gpu_util_avg", "gpu_util_min", "gpu_util_max",
    "gpu_mem_avg", "gpu_mem_min", "gpu_mem_max",
]

# Función auxiliar (ponerla fuera del main)
def get_question_type(q_id):
    for tipo, (inicio, fin) in TYPES_RANGES.items():
        if inicio <= q_id <= fin:
            return tipo
    return 'unknown'

def parse_arguments():
    parser = argparse.ArgumentParser(description="Models Benchmarking")

    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature to use")
    parser.add_argument("--context-docs", type=str, default="./context_docs", help="Path to the context documents directory")
    parser.add_argument("--dataset", type=str, default="./datasets/dataset-normativas.csv", help="Path to the CSV dataset file with columns: ID, Question, Answer")
    parser.add_argument("--question_id", type=int, default=None, help="Question ID in the csv dataset file (if not provided, all questions will be benchmarked)")

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

def load_dataset(path, question_id=None):
    df = pd.read_csv(path)
    df["ID"] = df["ID"].astype(int)

    if question_id is not None:
        df = df[df["ID"] == question_id]
        return df.to_dict("records")

    groups = {}

    for i, (start, end) in TYPES_RANGES.items():
        group = df[(df["ID"] >= start) & (df["ID"] <= end)]
        groups[i] = group.to_dict("records")

    return groups


if __name__ == "__main__":
    args = parse_arguments()
    context = load_context(args.context_docs)
    question_groups = load_dataset(args.dataset, args.question_id)

    models = [
        "Qwen/Qwen3.5-2B", # Luego cambiar a los modelos reales cuando se ejecute en el servidor
    ]

    if isinstance(question_groups, list):
        print("Single question mode")
        row = question_groups[0]
        for model in models:
            print(f"Loading model {model}")
            model_wrapper = load_hf_model(model, args.temperature)

            print(f"Benchmarking model {model} with question {row['ID']}")
            question = row["Question"]
            benchmark = Benchmark(model, model_wrapper, context, question)
            result = benchmark.run_question()
            benchmark.print_question_result(result)
            unload_hf_model(model_wrapper)
        exit()

    csv_path = f"./out/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    pd.DataFrame(columns=RECORD_COLUMNS).to_csv(csv_path, index=False)
    print(f"Saving results incrementally to {csv_path}")

    for model in models:
        print(f"Loading model {model}")
        model_wrapper = load_hf_model(model, args.temperature)

        for bag_id in range(NUM_BAGS):
            sampled_questions = []

            for q_type, q_list in question_groups.items():
                sample = random.choices(q_list, k=BAG_SIZE_PER_TYPE)
                sampled_questions.extend(sample)

            for row in sampled_questions:
                question = row["Question"]
                expected_answer = row["Answer"]
                question_id = row["ID"]

                benchmark = Benchmark(model, model_wrapper, context, question)

                result = benchmark.run_question()

                record = {
                    "bag_id": bag_id,
                    "model": model,
                    "question_id": question_id,
                    "expected_answer": expected_answer,
                    "response": result["response"],
                    "total_duration_sec": result["total_duration"] / 1e9,
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "total_tokens": result["total_tokens"],
                    "question_type": get_question_type(question_id)
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
                print(f"Question {question_id} | bag {bag_id} | model {model} | saved to {csv_path}")

        unload_hf_model(model_wrapper)

    print(f"Raw data saved to {csv_path}")

    print("Evaluando Métricas")

    ejecutar_evaluacion(csv_path)
