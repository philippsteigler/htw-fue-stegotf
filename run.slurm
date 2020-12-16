#!/bin/bash
#SBATCH --time=01:20:00
#SBATCH --partition=ml
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8GB
#SBATCH -o HLR_run.out
#SBATCH -e HLR_run.err

module load modenv/ml
module load TensorFlow
module load Pillow

python3 main.py