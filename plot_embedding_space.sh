#!/bin/bash

# ======== SLURM Job Config ========
#SBATCH --job-name=plot_embedding       # 作业名
#SBATCH --output=plot_%j.out            # 输出日志文件（%j 会自动变成 job ID）
#SBATCH --error=plot_%j.err             # 错误日志文件

#SBATCH --time=0-04:00                  # 最长运行时间 (天-时:分)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=A100               # 要 GPU 节点的话留着
#SBATCH --exclude=compute-0-7           # 可选：排除某节点

# ======== Path Config ========
JOB=/share/nas2_3/yhuang/myrepo/mphys_project/Plot\ embedding\ space.py

echo ">>> Job started at $(date)"

# ======== Environment Setup ========
source /share/nas2_3/yhuang/byol/runscripts/venv/bin/activate

# ======== Run ========
echo ">>> Running $JOB"
python "$JOB"

echo ">>> Job finished at $(date)"
