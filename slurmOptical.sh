#!/bin/bash
#SBATCH --job-name=opticalFlow-test
#SBATCH --output=OpticalLogs/%x-%j.out
#SBATCH --error=OpticalLogs/%x-%j.err
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

#echo "preparing data"
#python timesformerPreprocessing.py

echo "Starting analysis"
# run your training script
python opticalFlow.py

echo "Job finished with exit code $? at $(date)"