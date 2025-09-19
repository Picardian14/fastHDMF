#!/bin/bash
#SBATCH --time=${TIME:-24:00:00}
#SBATCH --job-name=${JOB_NAME:-fit_job}
#SBATCH --mail-type=END
#SBATCH --mail-user=${MAIL_USER:-}
#SBATCH --mem=${MEM:-64G}
#SBATCH --cpus-per-task=${CPUS:-24}
#SBATCH --chdir=/network/iss/home/ivan.mindlin/Repos/fastHDMF/matlab
#SBATCH --output=../slurm/outputs/%x.out
#SBATCH --error=../slurm/errors/%x.err

# Example usage:
# sbatch slurm.run_fit.sh \
#   --export=ALL,PARCELLATION=AAL,DATASET=coma24,FIT_MODE=dyn_fc,CPUS=24,MEM=64G,JOB_NAME=AAL_dyn_fc
# or define environment variables beforehand.

ml matlab/R2022b

# Set defaults if vars unset
PARCELLATION=${PARCELLATION:-AAL}
DATASET=${DATASET:-coma24}
FIT_MODE=${FIT_MODE:-dyn_fc}
OBJ_RATE=${OBJ_RATE:-3.44}
HOURS=${HOURS:-12}
WITH_PLASTICITY=${WITH_PLASTICITY:-}
WITH_DECAY=${WITH_DECAY:-}
TR_OVERRIDE=${TR_OVERRIDE:-}
BURNOUT=${BURNOUT:-}
WSIZE=${WSIZE:-30}
OVERLAP=${OVERLAP:-29}
CPUS=${CPUS:-24}
RESUME=${RESUME:-false}
OUT_ROOT=${OUT_ROOT:-../results}

MATLAB_ARGS=("parcellation" "${PARCELLATION}" \
             "dataset" "${DATASET}" \
             "fit_mode" "${FIT_MODE}" \
             "out_root" "${OUT_ROOT}" \
             "obj_rate" ${OBJ_RATE} \
             "hours" ${HOURS} \
             "resume" ${RESUME} \
             "wsize" ${WSIZE} \
             "overlap" ${OVERLAP} \
             "cpus" ${CPUS})

if [[ -n "$WITH_PLASTICITY" ]]; then MATLAB_ARGS+=("with_plasticity" ${WITH_PLASTICITY}); fi
if [[ -n "$WITH_DECAY" ]]; then MATLAB_ARGS+=("with_decay" ${WITH_DECAY}); fi
if [[ -n "$TR_OVERRIDE" ]]; then MATLAB_ARGS+=("TR" ${TR_OVERRIDE}); fi
if [[ -n "$BURNOUT" ]]; then MATLAB_ARGS+=("burnout" ${BURNOUT}); fi

# Build MATLAB invocation string
MATLAB_CALL="run_fit(";
for ((i=0; i<${#MATLAB_ARGS[@]}; i+=2)); do
  KEY=${MATLAB_ARGS[i]}
  VAL=${MATLAB_ARGS[i+1]}
  if [[ $VAL =~ ^[0-9.]+$ || $VAL == true || $VAL == false ]]; then
    MATLAB_CALL+="'${KEY}',${VAL}"
  else
    MATLAB_CALL+="'${KEY}','${VAL}'"
  fi
  if (( i+2 < ${#MATLAB_ARGS[@]} )); then MATLAB_CALL+=" , "; fi
 done
MATLAB_CALL+=")";

echo "Running MATLAB with: $MATLAB_CALL"
matlab -nodisplay -batch "$MATLAB_CALL" || exit 1
