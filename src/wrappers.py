import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


class HFModelWrapper:
    def __init__(self, model_name, temperature=0.2, max_new_tokens=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.pipeline = hf_pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            return_full_text=False,
        )

    def generate(self, prompt: str) -> str:
        outputs = self.pipeline(prompt)
        generated_text = outputs[0]['generated_text']
        if generated_text.startswith(prompt):
            return generated_text[len(prompt):].lstrip()
        return generated_text

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def unload(self):
        del self.model
        del self.pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class GGUFModelWrapper:
    def __init__(self, model_path, temperature=0.2, max_new_tokens=512,
                 n_ctx=4096, n_gpu_layers=-1):
        if Llama is None:
            raise ImportError("llama-cpp-python no está instalado. "
                              "Instálalo con: CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install llama-cpp-python")
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            temperature=temperature,
            verbose=False
        )
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        response = self.model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        return response['choices'][0]['message']['content']

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenize(text.encode('utf-8')))

    def unload(self):
        del self.model
        

def load_hf_model(model_name, temperature=0.2, max_new_tokens=512):
    return HFModelWrapper(model_name, temperature, max_new_tokens)

def load_gguf_model(model_path, temperature=0.2, max_new_tokens=512,
                    n_ctx=4096, n_gpu_layers=-1):
    return GGUFModelWrapper(model_path, temperature, max_new_tokens, n_ctx, n_gpu_layers)

def unload_model(wrapper):
    if hasattr(wrapper, 'unload'):
        wrapper.unload()