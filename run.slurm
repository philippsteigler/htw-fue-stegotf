#!/bin/bash
#SBATCH --time=01:20:00
#SBATCH --partition=ml		# -p
#SBATCH --nodes=1		    # -N
#SBATCH --tasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1	# -c
#SBATCH --mem-per-cpu=2GB
#SBATCH -o HLR_run.out
#SBATCH -e HLR_run.err

module load modenv/ml
module load TensorFlow
module load Pillow

python3 main.py