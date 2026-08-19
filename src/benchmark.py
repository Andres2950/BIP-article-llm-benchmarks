import time
import psutil

from resource_monitor import ResourceMonitor
from utils import build_prompt
from wrappers import unload_model


class Benchmark:
    def __init__(self, model_name, model_wrapper, context, question):
        self.model_name = model_name
        self.model_wrapper = model_wrapper
        self.context = context
        self.question = question
        self.prompt = build_prompt(context, question)

    def run_question(self):
        pid = psutil.Process().pid
        monitor = ResourceMonitor(pid)

        input_tokens = self.model_wrapper.count_tokens(self.prompt)

        monitor.start()
        start_time = time.perf_counter()
        response_text = self.model_wrapper.generate(self.prompt)
        total_duration = time.perf_counter() - start_time
        monitor.stop()

        resources = monitor.get_resource_usage()
        output_tokens = self.model_wrapper.count_tokens(response_text)

        result = {
            "model": self.model_name,
            "response": response_text,
            "total_duration": total_duration * 1e9,  # ns
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cpu": resources["cpu"],
            "memory": resources["memory"],
            "gpu_util": resources["gpu_util"],
            "gpu_mem": resources["gpu_mem"]
        }
        return result

    def print_question_result(self, result):
        # (sin cambios, igual que antes)
        cpu_cores = psutil.cpu_count(logical=False)
        print("\n" + "=" * 10)
        print("Modelo: ", result["model"])
        print("Respuesta: ", result["response"])
        print("\n" + "=" * 10)
        print("Duración total: ", result["total_duration"])
        print("Tokens de input: ", result["input_tokens"])
        print("Tokens de output: ", result["output_tokens"])
        print("Tokens totales: ", result["total_tokens"])
        print("\n" + "=" * 10)
        print("Recursos:")
        if result["cpu"] is not None:
            print(f"CPU: {result['cpu']['avg']}% (min: {result['cpu']['min']}%, max: {result['cpu']['max']}%)")
            print(f"CPU por núcleo ({cpu_cores} núcleos): {result['cpu']['avg'] / cpu_cores}% (min: {result['cpu']['min'] / cpu_cores}%, max: {result['cpu']['max'] / cpu_cores}%)")
        if result["memory"] is not None:
            print(f"RAM: {result['memory']['avg']}MB (min: {result['memory']['min']}MB, max: {result['memory']['max']}MB)")
        if result["gpu_util"] is not None:
            print(f"GPU: {result['gpu_util']['avg']}% (min: {result['gpu_util']['min']}%, max: {result['gpu_util']['max']}%)")
        if result["gpu_mem"] is not None:
            print(f"GPU Mem: {result['gpu_mem']['avg']}MB (min: {result['gpu_mem']['min']}MB, max: {result['gpu_mem']['max']}MB)")
        print("\n" + "=" * 10)