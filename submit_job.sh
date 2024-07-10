#!/bin/bash
#SBATCH --job-name=bert_inference       # Job name
#SBATCH --output=bert_inference.out     # Output file
#SBATCH --error=bert_inference.err      # Error file
#SBATCH --time=00:10:00                # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                # Partition name
#SBATCH --gres=gpu:1                   # Number of GPUs
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --mem=16G                      # Memory per node

#Pre-requisites
pip install numpy==1.22.4 torch transformers datasets

# Run the Python script
python3 distilbert-base-uncased.py --sample_text "This is a great movie with a thrilling plot"

