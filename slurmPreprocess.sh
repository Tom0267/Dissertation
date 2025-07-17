#!/bin/bash
#SBATCH --job-name=preprocessing
#SBATCH --output=preprocessingLogs/%x-%j.out
#SBATCH --error=preprocessingLogs/%x-%j.err
#SBATCH --partition=a2000-6h
#SBATCH --nodes=1
#SBATCH --ntasks=1

# activate your virtual environment
source ~/Dissertation/venv/bin/activate

# move to project directory
cd /mnt/nfs/homes/ditchfit/Dissertation/

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# log basic info
echo "Running on $(hostname)"
echo "preprocessing"

python preprocessing.py

echo "Job finished with exit code $? at $(date)"