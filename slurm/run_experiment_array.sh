#!/bin/bash
#SBATCH --partition=medium
#SBATCH --job-name=array_simulation
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --chdir=/network/iss/home/ivan.mindlin/Repos/fastHDMF/
#SBATCH --output=slurm/outputs/HDMF_%A_%a.out
#SBATCH --error=slurm/errors/HDMF_%A_%a.err

# Load Singularity module
module load singularity

# Define paths
SINGULARITY_IMAGE=/network/iss/home/ivan.mindlin/ubuntu_focal_conda.sif
EXPERIMENT_ID=$1
JOB_COUNT=$2
CONFIG_FILENAME=configs/experiments/${EXPERIMENT_ID}.yaml
# Parse optional --cpus argument after job count
CPUS=1
shift 2
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpus)
            CPUS="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done
# Run Python script inside Singularity container with job info
singularity exec $SINGULARITY_IMAGE bash -c "source /opt/anaconda/3/2023.07-2/base/etc/profile.d/conda.sh && conda activate fic && python3 fastHDMF/run_experiment.py $CONFIG_FILENAME --job-id $SLURM_ARRAY_TASK_ID --job-count $JOB_COUNT --cpus $CPUS"