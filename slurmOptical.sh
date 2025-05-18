#!/bin/bash
#SBATCH --job-name=opticalFlow-test
#SBATCH --output=OpticalLogs/%x-%j.out
#SBATCH --error=OpticalLogs/%x-%j.err
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

echo "Starting analysis"
# run your training script
python opticalFlow.py

echo "Job finished with exit code $? at $(date)"