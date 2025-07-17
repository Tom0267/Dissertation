#!/bin/bash
#SBATCH --job-name=LLM-test
#SBATCH --output=LLMLogs/%x-%j.out
#SBATCH --error=LLMLogs/%x-%j.err
#SBATCH --partition=a2000-48h
#SBATCH --nodes=1
#SBATCH --ntasks=1

# activate your virtual environment
source ~/Dissertation/venv/bin/activate

# move to project directory
cd /mnt/nfs/homes/ditchfit/Dissertation/

# log basic info
echo "Running on $(hostname)"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

echo "Starting testing"
#python Llama-3.2-Vision.py
python Deepseek-Janus-Pro-7B.py

#python evaluateLLM.py

echo "Job finished with exit code $? at $(date)"