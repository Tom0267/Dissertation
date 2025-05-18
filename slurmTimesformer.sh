#!/bin/bash
#SBATCH --job-name=timesformer-train
#SBATCH --output=timesformerLogs/%x-%j.out
#SBATCH --error=timesformerLogs/%x-%j.err
#SBATCH --partition=a5000-16c64g-6h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ditchfit@lancaster.ac.uk

# activate your virtual environment
source ~/Dissertation/venv/bin/activate

# move to project directory
cd /mnt/nfs/homes/ditchfit/Dissertation/

# log basic info
echo "Running on $(hostname)"

#echo "preparing data"
#python timesformerPreprocessing.py

echo "Starting training"
# run your training script
python LLaVA-Video.py

echo "Job finished with exit code $? at $(date)"