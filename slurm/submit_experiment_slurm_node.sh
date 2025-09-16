#!/bin/bash

set -euo pipefail

# Determine experiments directory (prefer ./experiments, fallback to ./configs/experiments)
EXPERIMENTS_DIR="experiments"
if [ ! -d "$EXPERIMENTS_DIR" ]; then
    if [ -d "configs/experiments" ]; then
        EXPERIMENTS_DIR="configs/experiments"
    fi
fi

if [ ! -d "$EXPERIMENTS_DIR" ]; then
    echo "Could not find experiments directory (looked for 'experiments' and 'configs/experiments')." >&2
    exit 1
fi

# Build list of experiment YAMLs
mapfile -t EXP_FILES < <(find "$EXPERIMENTS_DIR" -maxdepth 1 -type f -name '*.yaml' -printf '%f\n' | sort)

if [ ${#EXP_FILES[@]} -eq 0 ]; then
    echo "No experiment YAML files found in $EXPERIMENTS_DIR" >&2
    exit 1
fi

# Strip extensions for IDs
EXP_IDS=()
for f in "${EXP_FILES[@]}"; do
    EXP_IDS+=("${f%.yaml}")
done

print_menu() {
    echo "Available experiments (from $EXPERIMENTS_DIR):"
    local idx=1
    for id in "${EXP_IDS[@]}"; do
        echo "  $idx) $id"
        idx=$((idx+1))
    done
    echo
}

EXPERIMENT_INPUT="${1-}"

if [ -z "$EXPERIMENT_INPUT" ]; then
    print_menu
    read -rp "Select experiment by number or type experiment_id: " EXPERIMENT_INPUT
fi

# Decide if numeric selection or direct id
if [[ "$EXPERIMENT_INPUT" =~ ^[0-9]+$ ]]; then
    SEL_INDEX=$EXPERIMENT_INPUT
    if [ "$SEL_INDEX" -lt 1 ] || [ "$SEL_INDEX" -gt ${#EXP_IDS[@]} ]; then
        echo "Invalid selection index: $SEL_INDEX" >&2
        exit 1
    fi
    EXPERIMENT_ID="${EXP_IDS[$((SEL_INDEX-1))]}"
else
    EXPERIMENT_ID="$EXPERIMENT_INPUT"
    # Validate it exists
    if [[ ! " ${EXP_IDS[*]} " =~ [[:space:]]${EXPERIMENT_ID}[[:space:]] ]]; then
        echo "Experiment id '$EXPERIMENT_ID' not found among available IDs." >&2
        echo
        print_menu
        exit 1
    fi
fi

# Add day-month-hour-minute tier to experiment ID to ensure uniqueness
#TIMESTAMP=$(date +"%d%m%H%M")
#EXPERIMENT_ID="${EXPERIMENT_ID}_$TIMESTAMP"

CONFIG_FILENAME="${EXPERIMENTS_DIR}/${EXPERIMENT_ID}.yaml"

echo "Submitting SLURM node job for experiment: $EXPERIMENT_ID (config: $CONFIG_FILENAME)"

echo "Calculating grid size..."
# Calculate grid size and show values
GRID_INFO=$(python utils/calculate_grid_size.py "$CONFIG_FILENAME")
echo "$GRID_INFO"
# Extract grid size
GRID_SIZE=$(echo "$GRID_INFO" | grep "Total grid combinations:" | awk '{print $4}')

echo
echo "Grid size: $GRID_SIZE combinations"

# Ask user for number of CPUs per task
read -rp "Enter number of CPUs for this job (or press Enter for default: 1): " USER_CPUS
if [ -z "$USER_CPUS" ]; then
    CPUS=1
else
    CPUS=$USER_CPUS
fi
# Validate CPUs
if ! [[ "$CPUS" =~ ^[0-9]+$ ]] || [ "$CPUS" -lt 1 ]; then
    echo "Invalid CPUs: $CPUS" >&2
    exit 1
fi

echo "Using CPUs: $CPUS"

echo
echo "Final configuration:"
echo "  Experiment ID: $EXPERIMENT_ID"
echo "  Config file: $CONFIG_FILENAME"
echo "  Grid size: $GRID_SIZE"
echo "  CPUs: $CPUS"
echo

# Submit the node job and capture the job ID
JOB_ID=$(sbatch --parsable --cpus-per-task=$CPUS slurm/run_experiment_node.sh "$EXPERIMENT_ID" --cpus $CPUS)

echo "Node job submitted with ID: $JOB_ID"
