#!/bin/bash

# Test the grid size calculation with the current config
echo "Testing grid size calculation..."

CONFIG_FILE="configs/experiments/Homeostatic_Grid.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Config file: $CONFIG_FILE"
echo

python utils/calculate_grid_size.py "$CONFIG_FILE"