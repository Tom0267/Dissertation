#!/bin/bash
#SBATCH --job-name=timesformer-train
#SBATCH --output=timesformerLogs/%x-%j.out
#SBATCH --error=timesformerLogs/%x-%j.err
#SBATCH --partition=a5000-6h
#SBATCH --nodes=1
#SBATCH --ntasks=1

# activate your virtual environment
source ~/Dissertation/venv/bin/activate

# move to project directory
cd /mnt/nfs/homes/ditchfit/Dissertation/

# log basic info
echo "Running on $(hostname)"

#echo "preparing data"
#python timesformerPreprocessing.py

echo "Starting timesformer"
# run your training script
python timesformer.py

echo "Job finished with exit code $? at $(date)"