#!/bin/bash

#SBATCH --job-name=scores_extract
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=10G
#SBATCH --time=5:00:00

source ~/.pyenv/versions/anaconda3-2019.10/bin/activate
conda activate stklia_env

python3 binary_classification.py ../data/v2t/xvector_test.txt.b.standardized ../data/v2t/sample_test.txt.b ../data/gender_labels_v2.txt exp 0.5 -g
