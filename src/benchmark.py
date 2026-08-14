import psutil

from langchain_ollama import ChatOllama

from resource_monitor import ResourceMonitor, find_ollama_pid
from utils import build_prompt

class Benchmark:
    def __init__(self, model, temperature, context, question, show_response=False):
        self.model = model
        self.temperature = temperature
        self.context = context
        self.question = question
        self.show_response = show_response
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

        result = {
            "response": response.content,
            "model": response.response_metadata["model"],
            "total_duration": response.response_metadata["total_duration"],
            "load_duration": response.response_metadata["load_duration"],
            "input_tokens": response.usage_metadata["input_tokens"],
            "output_tokens": response.usage_metadata["output_tokens"],
            "total_tokens": response.usage_metadata["total_tokens"],
            "resources": monitor.get_resource_usage()
        }

        return result
    
    def print_question_result(self):
        result = self.run_question()
        cpu_cores = psutil.cpu_count(logical=False)

        if self.show_response:
            print("\n" + "=" * 10)
            print("Respuesta:\n")
            print(result["response"])

        print("\n" + "=" * 10)
        print("Modelo: ", result["model"])
        print("Duración total: ", result["total_duration"])
        print("Duración de carga: ", result["load_duration"])
        print("Tokens de input: ", result["input_tokens"])
        print("Tokens de output: ", result["output_tokens"])
        print("Tokens totales: ", result["total_tokens"])

        if "resources" in result:
            print("\n" + "=" * 10)
            print("Recursos:")
            if result["resources"]["cpu"] is not None:
                print(f"CPU: {result['resources']['cpu']['avg']}% (min: {result['resources']['cpu']['min']}%, max: {result['resources']['cpu']['max']}%)")
                print(f"CPU por núcleo ({cpu_cores} núcleos): {result['resources']['cpu']['avg'] / cpu_cores}% (min: {result['resources']['cpu']['min'] / cpu_cores}%, max: {result['resources']['cpu']['max'] / cpu_cores}%)")
            if result["resources"]["memory"] is not None:
                print(f"RAM: {result['resources']['memory']['avg']}MB (min: {result['resources']['memory']['min']}MB, max: {result['resources']['memory']['max']}MB)")
            if result["resources"]["gpu_util"] is not None:
                print(f"GPU: {result['resources']['gpu_util']['avg']}% (min: {result['resources']['gpu_util']['min']}%, max: {result['resources']['gpu_util']['max']}%)")
            if result["resources"]["gpu_mem"] is not None:
                print(f"GPU Mem: {result['resources']['gpu_mem']['avg']}MB (min: {result['resources']['gpu_mem']['min']}MB, max: {result['resources']['gpu_mem']['max']}MB)")
        print("\n" + "=" * 10)