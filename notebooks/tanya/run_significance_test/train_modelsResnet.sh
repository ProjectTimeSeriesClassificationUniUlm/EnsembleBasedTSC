#!/bin/bash
#SBATCH --ntasks=40
#SBATCH --time=03:00:00
#SBATCH --mem=10000
#SBATCH --gres=gpu:4
source ../../../../ts_ensemble/bin/activate
python train_modelsResnet.py