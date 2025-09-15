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
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experiment_manager import ExperimentManager
from src.simulation_runner import HDMFSimulationRunner
def main():
    parser = argparse.ArgumentParser(description='Run HDMF experiment from config file')
    parser.add_argument('config', help='Config file name (e.g., "default_hdmf" or "experiments/high_coupling")')
    parser.add_argument('--id', help='Custom experiment ID (optional)', default=None)
    parser.add_argument('--list-configs', action='store_true', help='List available config files')
    print(project_root)
    args = parser.parse_args()
    print(args)
    if args.list_configs:
        configs_dir = project_root / "configs"
        print("Available configurations:")
        print("\nMain configs:")
        for config_file in configs_dir.glob("*.yaml"):
            print(f"  {config_file.stem}")
        
        print("\nExperiment configs:")
        exp_dir = configs_dir / "experiments"
        if exp_dir.exists():
            for config_file in exp_dir.glob("*.yaml"):
                print(f"  experiments/{config_file.stem}")
        return
    
    print(f"Running experiment with config: {args.config}")
    if args.id:
        print(f"Using custom experiment ID: {args.id}")    
    try:
        experiment_manager = ExperimentManager(project_root)
        experiment_dir, experiment_id = experiment_manager.setup_experiment(
            config_path=args.config, experiment_id=args.id,
            job_id=os.getenv('SLURM_ARRAY_TASK_ID'), job_count_str=os.getenv('SLURM_ARRAY_TASK_COUNT')
        )

        # ExperimentManager now stores the ObservablesPipeline; runner will consume it.
        runner = HDMFSimulationRunner(project_root, experiment_manager)
        runner.run_experiment(experiment_id)
        print(f"\n✅ Experiment completed successfully!")
        print(f"Experiment ID: {experiment_id}")
        print(f"Results directory: {experiment_dir}")

    except Exception as e:
        print(f"❌ Experiment failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
