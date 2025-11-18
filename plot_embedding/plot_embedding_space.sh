#!/bin/bash

# ======== SLURM Job Config ========
#SBATCH --time=0-04:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=A100
#SBATCH --exclude=compute-0-7

# ======== Path Config ========
JOB=/share/nas2_3/yhuang/myrepo/mphys_project/plot_embedding/plot_embedding_space.py

echo ">>> Job started at $(date)"

# ======== Environment Setup ========
source /share/nas2_3/yhuang/byol/runscripts/venv/bin/activate

# ======== Run ========
echo ">>> Running $JOB"
python "$JOB"

echo ">>> Job finished at $(date)"
