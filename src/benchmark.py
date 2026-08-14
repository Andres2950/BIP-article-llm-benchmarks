import psutil

from langchain_ollama import ChatOllama

from resource_monitor import ResourceMonitor, find_ollama_pid
from utils import build_prompt

class Benchmark:
    def __init__(self, model, temperature, context, question):
        self.model = model
        self.temperature = temperature
        self.context = context
        self.question = question
        self.llm = ChatOllama(model=model, temperature=temperature)
        self.prompt = build_prompt(context, question)

    def run_question(self):
        ollama_pid = find_ollama_pid()
        if not ollama_pid:
            raise Exception("No se encontró el proceso de Ollama, no se puede monitorear el uso de recursos.")

        monitor = ResourceMonitor(ollama_pid)

        monitor.start()
        response = self.llm.invoke(self.prompt)
        monitor.stop()

        resources = monitor.get_resource_usage()

        result = {
            "model": self.model,
            "response": response.content,
            "total_duration": response.response_metadata["total_duration"],
            "load_duration": response.response_metadata["load_duration"],
            "input_tokens": response.usage_metadata["input_tokens"],
            "output_tokens": response.usage_metadata["output_tokens"],
            "total_tokens": response.usage_metadata["total_tokens"],
            "cpu": resources["cpu"],
            "memory": resources["memory"],
            "gpu_util": resources["gpu_util"],
            "gpu_mem": resources["gpu_mem"]
        }

        return result
    
    def print_question_result(self, result):
        cpu_cores = psutil.cpu_count(logical=False)

        print("\n" + "=" * 10)
        print("Modelo: ", result["model"])
        print("Respuesta: ", result["response"])

        print("\n" + "=" * 10)
        print("Duración total: ", result["total_duration"])
        print("Duración de carga: ", result["load_duration"])
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