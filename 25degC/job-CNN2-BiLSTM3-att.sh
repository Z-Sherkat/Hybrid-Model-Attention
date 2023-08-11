#!/bin/bash
#SBATCH --account=def-makarenk
#SBATCH --array=1-14
#SBATCH --gres=gpu:1       # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=47G        # memory per node
#SBATCH --time=00-23:00      # time (DD-HH:MM)
#SBATCH --job-name=25CNN2BLSTM3-att-verbose
#SBATCH --output=%x-%j.out  # %N for node name, %j for jobID


python ./4-CNN-2-BiLSTM-att-3.py --array $SLURM_ARRAY_TASK_ID