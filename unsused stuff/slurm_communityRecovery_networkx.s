#!/bin/bash
#
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=100:00:00
#SBATCH	--mem=10G
#SBATCH --job-name=myCommunityRecovery
#SBATCH --output=slurm_%j.out


cd 'Community Detection & Coarsening'
python3 coarsendSBM_communityRecovery_fromGraph.py