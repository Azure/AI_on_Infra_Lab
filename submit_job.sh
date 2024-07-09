#!/bin/bash
#SBATCH --job-name=llm_benchmark       # Job name
#SBATCH --output=llm_benchmark.out     # Output file
#SBATCH --error=llm_benchmark.err      # Error file
#SBATCH --time=00:10:00                # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                # Partition name
#SBATCH --gres=gpu:1                   # Number of GPUs
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --mem=16G                      # Memory per node

#Pre-requisites
pip install numpy==1.22.4 torch transformers datasets

# Load modules or activate environment if needed
module load python/3.8
module load cuda/12.4

# Run the Python script
python3 distilbert-base-uncased.py --sample_text "This is a great movie with a thrilling plot"

