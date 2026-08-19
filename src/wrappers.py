import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

# ----------------------------------------------
# WRAPPER PARA MODELOS HUGGING FACE
# ----------------------------------------------
class HFModelWrapper:
    def __init__(self, model_name, temperature=0.2, max_new_tokens=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
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


# ----------------------------------------------
# WRAPPER PARA MODELOS GGUF
# ----------------------------------------------
class HFGGUFModelWrapper:
    def __init__(self, repo_id, filename, temperature=0.2, max_new_tokens=512):
        self.tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            gguf_file=filename,
            use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            gguf_file=filename,
            torch_dtype=torch.float16,  
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        self.pipeline = hf_pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def generate(self, prompt: str) -> str:
        outputs = self.pipeline(prompt)
        generated_text = outputs[0]['generated_text']
        if generated_text.startswith(prompt):
            return generated_text[len(prompt):].lstrip()
        return generated_text

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def unload(self):
        del self.model
        del self.pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ----------------------------------------------
# FUNCIONES DE CARGA
# ----------------------------------------------
def load_hf_model(model_name, temperature=0.2, max_new_tokens=512):
    return HFModelWrapper(model_name, temperature, max_new_tokens)

def load_hf_gguf_model(repo_id, filename, temperature=0.2, max_new_tokens=512):
    return HFGGUFModelWrapper(repo_id, filename, temperature, max_new_tokens)

def load_model_from_config(model_cfg, temperature=0.2, max_new_tokens=512):
    if model_cfg.format == "hf":
        return load_hf_model(model_cfg.repo_id, temperature, max_new_tokens)
    elif model_cfg.format == "gguf":
        return load_hf_gguf_model(
            repo_id=model_cfg.repo_id,
            filename=model_cfg.filename,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
    else:
        raise ValueError(f"Formato no soportado: {model_cfg.format}")

def unload_model(wrapper):
    if hasattr(wrapper, 'unload'):
        wrapper.unload()