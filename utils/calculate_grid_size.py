#!/usr/bin/env python3
"""
Utility to calculate grid size and suggest optimal SLURM array configuration
"""
import sys
import yaml
import numpy as np
from pathlib import Path

def calculate_grid_size(config_path):
    """Calculate total grid size from config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    grid = config.get("grid", {})
    if not grid:
        return 1
    
    total_combinations = 1
    
    for param_name, spec in grid.items():
        if "fun" in spec:
            # Custom function - need to evaluate to get size
            fun_name = spec["fun"]
            args = spec.get("args", [])
            kwargs = spec.get("kwargs", {})
            
            try:
                func = eval(fun_name)
                values = func(*args, **kwargs)
                param_size = len(values)
            except Exception as e:
                print(f"Error evaluating {fun_name}: {e}")
                return None
        else:
            # Traditional start/end/step
            start = spec["start"]
            end = spec["end"] 
            step = spec["step"]
            values = np.arange(start, end, step)
            param_size = len(values)
        
        total_combinations *= param_size
        print(f"{param_name}: {param_size} values")
        # print actual values for sanity check
        try:
            val_list = values.tolist()
        except Exception:
            val_list = list(values)
        print(f"  values: {val_list}")
    
    # Handle optional "over" parameters (e.g., SC matrices selection)
    over = config.get("over", {})
    if over:
        print("\nOver parameters:")
        total_over = 1
        for param_name, spec in over.items():
            if "fun" in spec:
                fun_name = spec["fun"]
                args = spec.get("args", [])
                kwargs = spec.get("kwargs", {})
                try:
                    func = eval(fun_name)
                    values = func(*args, **kwargs)
                except Exception as e:
                    print(f"Error evaluating {fun_name}: {e}")
                    return None
            else:
                start = spec["start"]
                end = spec["end"]
                step = spec["step"]
                values = np.arange(start, end, step)
            over_size = len(values)
            total_over *= over_size
            print(f"{param_name}: {over_size} values")
            try:
                over_list = values.tolist()
            except Exception:
                over_list = list(values)
            print(f"  values: {over_list}")
        print(f"\nTotal SC matrices (over combinations): {total_over}")
    return total_combinations

def suggest_array_size(grid_size, max_jobs=None):
    """Suggest optimal array size based on grid size"""
    if max_jobs is None:
        # Default suggestions based on grid size
        if grid_size <= 25:
            return min(grid_size, 5)
        elif grid_size <= 100:
            return min(grid_size, 10)
        elif grid_size <= 500:
            return min(grid_size, 25)
        elif grid_size <= 2000:
            return min(grid_size, 50)
        else:
            return min(grid_size, 100)
    else:
        return min(grid_size, max_jobs)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calculate_grid_size.py <config_file>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    grid_size = calculate_grid_size(config_path)
    
    if grid_size is None:
        print("Error calculating grid size")
        sys.exit(1)
    
    print(f"\nTotal grid combinations: {grid_size}")
    
    suggested = suggest_array_size(grid_size)
    print(f"Suggested SLURM array size: {suggested}")
    print(f"Tasks per job: {grid_size // suggested} (remainder: {grid_size % suggested})")