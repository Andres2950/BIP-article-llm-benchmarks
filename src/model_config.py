from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str 
    repo_id: str                    # repo id de huggingface, solo si es hf
    format: str
    filename: str  # nombre del gguf
    context_size: int = 4096
    gpu_layers: int = -1            # solo gguf



MODELS = [
    ## 4.6 GB
    #ModelConfig(
    #    name = "Qwen3.5-4B",
    #    repo_id="Qwen/Qwen3.5-4B",
    #    format="hf",
    #    filename="",
    #    context_size=16384,
    #    gpu_layers=-1,
    #),
    # 4.6 GB
    ModelConfig(
        name = "Llama-3.1-8B-Instruct-Q4_K_M",
        repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        format="gguf",
        filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        context_size=16384,
        gpu_layers=-1,
    ),
    ## 5 GB
    #ModelConfig(
    #    name = "DeepSeek-R1-Distill-Qwen-14B-Q2_K",
    #    repo_id="bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
    #    format="gguf",
    #    filename="DeepSeek-R1-Distill-Qwen-14B-Q2_K.gguf",
    #    context_size=16384,
    #    gpu_layers=-1,
    #),
    ## 8.7 GB
    #ModelConfig(
    #    name = "Llama-3.1-8B-Instruct-Q8_0",
    #    repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    #    format="gguf",
    #    filename="Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
    #    context_size=16384,
    #    gpu_layers=-1,
    #),
    ##  11.3 GB 
    #ModelConfig(
    #    name="DeepSeek-R1-Distill-Qwen-14B-Q6_K",
    #    repo_id="bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
    #    format="gguf",
    #    filename="DeepSeek-R1-Distill-Qwen-14B-Q6_K.gguf",
    #    context_size=16384,
    #    gpu_layers=-1,
    #),
    ## 18.4 GB
    #ModelConfig(
    #    name = "Qwen3.5-35B-A3B-Q4_K_M",
    #    repo_id="lmstudio-community/Qwen3.5-35B-A3B-GGUF",
    #    format="gguf",
    #    filename="Qwen3.5-35B-A3B-Q4_K_M.gguf",
    #    context_size=16384,
    #    gpu_layers=-1,
    #),
    ##  	29.2 GB
    #ModelConfig(
    #    name="DeepSeek-R1-Distill-Qwen-14B-F16",
    #    repo_id="bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
    #    format="gguf",
    #    filename="DeepSeek-R1-Distill-Qwen-14B-f16.gguf",
    #    context_size=16384,
    #    gpu_layers=-1,
    #),
    ## 36.4 GB
    #ModelConfig(
    #    name = "Qwen3.5-35B-A3B-Q8_0",
    #    repo_id="lmstudio-community/Qwen3.5-35B-A3B-GGUF",
    #    format="gguf",
    #    filename="Qwen3.5-35B-A3B-Q8_0.gguf",
    #    context_size=16384,
    #    gpu_layers=-1,
    #),
    ## 36.4 GB - 42.52 GB
    #ModelConfig(
    #    name = "Llama-3.3-70B-Instruct-Q4_K_M",
    #    repo_id="bartowski/Llama-3.3-70B-Instruct-GGUF",
    #    format="gguf",
    #    filename="Llama-3.3-70B-Instruct-Q4_K_M.gguf",
    #    context_size=16384,
    #    gpu_layers=-1,
    #),
    ## ???   
    #ModelConfig(
    #    name = "Llama-3.1-70B-LatamGPT",
    #    repo_id="latam-gpt/Llama-3.1-70B-LatamGPT-SFT-1.0",
    #    format="hf",
    #    filename="",
    #    context_size=16384,
    #    gpu_layers=-1,
    #),
]
