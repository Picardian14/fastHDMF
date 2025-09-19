#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=HCP_NVC_dyn_fcd_fit
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --mem=64G
#SBATCH --cpus-per-task=24
#SBATCH --chdir=/network/iss/home/ivan.mindlin/Repos/fastHDMF/matlab
#SBATCH --output=../slurm/outputs/HCP_NVC_dyn_fcd_fit.out
#SBATCH --error=../slurm/errors/HCP_NVC_dyn_fcd_fit.err

ml matlab/R2022b
matlab -nodisplay -batch "run('Bayesian_dyn_fcd_fit.m'); exit;" 