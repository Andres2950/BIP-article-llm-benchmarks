# BIP-article-llm-benchmarks
Benchmarking de LLMs para tareas relacionadas con el procesamiento y consulta de documentación normativa del Centro Nacional de Detección Temprana de Cáncer Gastrointestinal (CNDTCG).

Este repositorio tiene datasetes, documentos de contexto, resultados y código utilizado para los experimientos hechos para un artículo subido al BIP.

## Descripción

El proyecto evalúa distintos modelos de lenguaje en preguntas sobre documentación del CNDTCG.

Los experimientos buscan considerar la calidad de respuesta y el rendimiento computacional de los modelos. Esto para analizar el comportamiento de distintos modelos con distintos niveles de cuantización.

Los benchmarks fueron diseñadps para ejecutarse localmente (probado solo con modelos pequeños) o en infraestructura de computación de alto rendimiento como el [supercomputador Kabré](https://kabre.cenat.ac.cr/).

## Objetivos

Los principales objetivos de este proyecto son:

- Evaluar distintos modelos sobre un dataset de preguntas basado en documentos del CNDTCG
- Comparar calidad de respuestas
- Evaluar consumo de recursos computacionales
- Analizar efecto de diferentes configuraciones

## Modelos evaluados

El repositorio contiene los  resultados correspondientes a los modelos que fueron evaluadios:

- DeepSeek R1 Distill Qwen F16
- DeepSeek R1 Distill Qwen Q6
- Llama Q4 + DeepSeek R1 Distill Qwen Q2, Llama Q8
- Qwen 3.5 4B

## Dataset

Las preguntas utilizadas para el benchmark se encuentran en:

```
datasets/
├── dataset.json
└── dataset_normativas.csv
```

El formato soportado por el programa es el csv. 
El arquivo JSON es el mismo dataset, solamente usando un formato distinto.

Los documentos utilizados como contexto para el benchmark se encuentran en:

```
context_docs/
```

Estos incluyen documentación normatica y administratica relacionada con el CNDTCG.
El dataset cuenta con 300 preguntas, 100 de cada tipo de pregunta (sí o no, respuesta corta, respuesta abierta).

## Metodología

De forma general, el proceso experimental consiste en:

- Hacer preguntas para el dataset a partir de los documentos de contexto
- Configurar el modelo
- Utilizar RAG para cargar los documentos, se carga ncomo un vector store
- Ejecutar benchmark para cada uno de los modelos
  - Obtener pregunta
  - Obtener los TOP K=5 chunks de los documentos más similares a la pregunta, utilizando FAISS
  - Generar respuesta
  - Medir rendimiento
- Evaluar resultados


## Guía para correrlo

El programa usa **Python 3.11**. En Kabré ese intérprete se crea con el módulo `miniconda/3`. Los modelos GGUF corren en una GPU de la partición `nukwa` (nodos Nvidia L40S o V100).

### 1. Clonar el repositorio

Conéctese al nodo de login. Ahí no se ejecuta el benchmark: solo se preparan archivos y se envían trabajos.

```bash
ssh -p 22022 usuario@kabre.cenat.ac.cr
cd /work/$USER
git clone https://github.com/Andres2950/BIP-article-llm-benchmarks
cd BIP-article-llm-benchmarks
```

### 2. Crear el entorno con Miniconda

En un nodo Nukwa, `module` no existe hasta cargar el sistema de módulos de AlmaLinux 9:

```bash
. /opt/Modules/3.2.10/init/sh
module load miniconda/3
conda create -n llm-env python=3.11 -y
source activate llm-env
```

El punto y el espacio del primer comando son obligatorios. Dentro de un job de SLURM use `source activate llm-env`. `conda activate` falla si el shell del nodo no inicializó conda.

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

`llama-cpp-python` carga los GGUF y no está en `requirements.txt`. No lo compile. Instale una rueda CUDA ya construida, de una versión de CUDA igual o menor a la que reporta el driver:

```bash
nvidia-smi
```

En la primera línea aparece `CUDA Version: 12.x`. Use el índice `cu12x` correspondiente (`12.4` → `cu124`). Una rueda más vieja también sirve si el driver es más nuevo. En las V100 no use CUDA 13: esas ruedas piden compute capability 7.5 y la V100 es 7.0. La L40S sí la acepta.

```bash
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
  --only-binary=llama-cpp-python
```

Cambie `cu124` por `cu118`, `cu121`, `cu122`, `cu123`, `cu125`, `cu130` o `cu132` según la línea de `nvidia-smi`. `--only-binary` hace que pip falle si no hay rueda, en lugar de compilar.

### 4. Elegir los modelos

En `src/model_config.py`, descomente en `MODELS` solo los modelos que quepan en la GPU del nodo. El programa los corre uno tras otro. `context_size` y `gpu_layers` de cada ficha no se aplican: un GGUF se carga con contexto 16384 y todas las capas en GPU.

Cree las carpetas de caché y de salida:

```bash
mkdir -p /data/$USER/huggingface_cache logs out
```

### 5. Enviar el trabajo con SLURM

Prueba de una pregunta (`--question_id 54`), en la cola de depuración de Nukwa (máximo 4 horas):

```bash
#!/bin/bash
#SBATCH --job-name=llm_benchmark_smoke
#SBATCH --output=./logs/smoke_%j.out
#SBATCH --error=./logs/smoke_%j.err
#SBATCH --partition=nukwa-debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca

set -euo pipefail

. /opt/Modules/3.2.10/init/sh
module load miniconda/3
source activate llm-env

export HF_HOME=/data/$USER/huggingface_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/$USER/BIP-article-llm-benchmarks
mkdir -p logs out

nvidia-smi
python src/main.py --question_id 54
```

Benchmark completo, cola `nukwa` (máximo 24 horas). Se excluyen `nukwa-00` a `nukwa-03`: tienen 32 GB de RAM y una V100, y no alcanzan para los modelos grandes.

```bash
#!/bin/bash
#SBATCH --job-name=llm_benchmark
#SBATCH --output=./logs/benchmark_%j.out
#SBATCH --error=./logs/benchmark_%j.err
#SBATCH --partition=nukwa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=nukwa-00.cnca,nukwa-01.cnca,nukwa-02.cnca,nukwa-03.cnca

set -euo pipefail

. /opt/Modules/3.2.10/init/sh
module load miniconda/3
source activate llm-env

export HF_HOME=/data/$USER/huggingface_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/$USER/BIP-article-llm-benchmarks
mkdir -p logs out

nvidia-smi
python src/main.py
```

Guarde cada script en `slurm/` y envíelo desde el directorio del proyecto:

```bash
mkdir -p slurm
sbatch slurm/smoke_test.slurm
sbatch slurm/benchmark.slurm
```

Para ver la cola:

```bash
watch -n 5 squeue -u $USER
```

El CSV queda en `out/`. La semilla usada se imprime al inicio de la corrida (`Seed: ...`).

