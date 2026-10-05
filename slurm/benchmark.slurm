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

module load miniconda/3
conda activate llm-env

export HF_HOME=/data/$USER/huggingface_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/$USER/BIP-article-llm-benchmarks
mkdir -p logs out /data/$USER/huggingface_cache

nvidia-smi
python src/main.py
