#!/bin/bash
#SBATCH --job-name=timesformer-train
#SBATCH --output=timesformerLogs/%x-%j.out
#SBATCH --error=timesformerLogs/%x-%j.err
#SBATCH --partition=a2000-48h
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
 
#activate virtual environment 
source ~/Dissertation/venv/bin/activate 

#move to project directory
cd /mnt/nfs/homes/ditchfit/Dissertation/

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

#log basic info
echo "Running on $(hostname)"

echo "Starting timesformer"
python timesformer.py

echo "Job finished with exit code $? at $(date)"