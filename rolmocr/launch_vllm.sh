#!/bin/bash
#SBATCH --partition=A100-PCI,H200-DA,RTX3090-MLT,
#SBATCH --job-name=vllm-server
#SBATCH --export=ALL,HF_HUB_CACHE=/ds/models/hf-cache-slt/
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1-00:00:00
#SBATCH --output=vllm_server.out
#SBATCH --error=vllm_server.err

# Load conda and activate your environment
source ~/.bashrc
conda activate vLLM-Starter

# Set VLLM env and launch the server
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=DEBUG

# Run the server, bind to 0.0.0.0 if accessing from another node
vllm serve reducto/RolmOCR --device cuda --port 8000 --host 0.0.0.0
