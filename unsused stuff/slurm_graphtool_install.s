#!/bin/bash
#
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=100:00:00
#SBATCH	--mem=10G
#SBATCH --job-name=myCommunityRecovery
#SBATCH --output=slurm_%j.out


export PATH=/pkgs/anaconda3/bin:$PATH
conda create -p /scratch/gobi2/ghnafis/gt -c conda-forge graph-tool
mkdir -p /scratch/gobi2/ghnafis/gt
export PYTHONPATH=/scratch/gobi2/ghnafis/gt:$PYTHONPATH
