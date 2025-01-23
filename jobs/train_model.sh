#!/bin/bash

# SLURM SUBMIT SCRIPT

#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_2080_ti:4
#SBATCH --time=40:00:00
# save output in "slurm-{jobid}-{jobname}.out" file
#SBATCH --output=jobs/logs/slurm-%j-%x.out
# Send keyboard interupt signal 5 minutes before time limit
#SBATCH --signal=SIGINT@300

# Go to repository
cd /cluster/scratch/lgentner/repositories/deeprec/

# Load modules
module load stack/2024-06 python_cuda/3.11.6 eth_proxy

# Activate venv
source .venv/bin/activate

# debugging flags
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1

# run script from above
srun python scripts/4-model-train.py --config config/paper-experiments/train_1-era-rdcd.yaml