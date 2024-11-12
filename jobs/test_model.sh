#!/bin/bash

# SLURM SUBMIT SCRIPT

#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_2080_ti:1
#SBATCH --time=2:00:00
# save output in "slurm-{jobid}-{jobname}.out" file
#SBATCH --output=jobs/logs/slurm-%j-%x.out
# Send keyboard interupt signal 5 minutes before time limit
#SBATCH --signal=SIGINT@300

# Go to repository
cd /cluster/home/lgentner/repositories/deep-waters/

# Load modules
module load stack/2024-06 python_cuda/3.11.6 eth_proxy

# Activate venv
source .venv/bin/activate

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# run script from above
srun python scripts/5.3-model-test.py lgentner/deepwaters_global dad9gfb5