#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=HCP_NVC_dyn_fc
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --mem=64G
#SBATCH --cpus-per-task=24
#SBATCH --chdir=/network/iss/home/ivan.mindlin/Repos/fastHDMF/matlab
#SBATCH --output=outputs/HCP_NVC_dyn_fc.out
#SBATCH --error=outputs/HCP_NVC_dyn_fc.err

ml matlab/R2022b
matlab -nodisplay -batch "run('Bayesian_dyn_fc_fit.m');"