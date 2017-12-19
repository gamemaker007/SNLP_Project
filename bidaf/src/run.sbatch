#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=fast
#SBATCH --mail-type=END
#SBATCH --mail-user=kv730@nyu.edu
#SBATCH --output=snlp_bidaf6.out
python bidaf.py