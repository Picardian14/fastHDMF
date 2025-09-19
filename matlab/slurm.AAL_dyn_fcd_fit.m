#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=AAL_NVC_dyn_fcd
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --mem=64G
#SBATCH --cpus-per-task=24
#SBATCH --chdir=/network/iss/home/ivan.mindlin/Repos/fastHDMF/matlab
#SBATCH --output=../slurm/outputs/AAL_NVC_dyn_fcd.out
#SBATCH --error=../slurm/errors/AAL_NVC_dyn_fcd.err

ml matlab/R2022b
matlab -nodisplay -batch "run('AAL_dyn_fcd_fit.m');"