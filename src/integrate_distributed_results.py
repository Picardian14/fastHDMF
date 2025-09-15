#!/usr/bin/env python3
"""
Command-line script to run HDMF experiments
Usage: python run_experiment.py <config_name> [experiment_id]
"""
import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from modeling.HDMF.src.experiment_manager import ExperimentManager

def main():
    parser = argparse.ArgumentParser(description='Integrate results already ran as a SLURM job')
    parser.add_argument('id', help='Name of the experiment to be integrated')    
    
    print(project_root)
    args = parser.parse_args()
    print(args)

    print(f"Integrating experiment: {args.id}")
    
    try:
        experiment_manager = ExperimentManager(project_root)
        _ = experiment_manager.integrate_slurm_results(args.id)     
        print("\n✅Results got integrated")

    except Exception as e:
        print(f"❌ Experiment failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
