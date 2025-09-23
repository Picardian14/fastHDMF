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

from .experiment_manager import ExperimentManager
from .simulation_runner import HDMFSimulationRunner
def main():
    parser = argparse.ArgumentParser(description='Run HDMF experiment from config file')
    parser.add_argument('config', help='Config file name (e.g., "default_hdmf" or "experiments/high_coupling")')
    parser.add_argument('--id', help='Custom experiment ID (optional)', default=None)
    parser.add_argument('--job-id', help='SLURM array job ID', type=int, default=None)
    parser.add_argument('--job-count', help='Total number of SLURM array jobs', type=int, default=None)
    parser.add_argument('--list-configs', action='store_true', help='List available config files')
    parser.add_argument('--cpus', type=int, default=None, help='Max CPUs/workers (overrides default 32)')
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
    if args.job_id is not None:
        print(f"SLURM array job ID: {args.job_id}")
    if args.job_count is not None:
        print(f"Total SLURM jobs: {args.job_count}")
        
    try:
        experiment_manager = ExperimentManager(
            project_root,
            config_path=args.config,
            results_dir=Path("/network/iss/cohen/data/Ivan/fastHDMF/"),
            job_id=args.job_id if args.job_id is not None else os.getenv('SLURM_ARRAY_TASK_ID'),
            job_count=args.job_count if args.job_count is not None else os.getenv('SLURM_ARRAY_TASK_COUNT'),
        )
        # Store user override for parallel CPUs
        experiment_manager.max_cpus = args.cpus
        # after init, get experiment_dir and id
        experiment_dir = experiment_manager.experiment_dir
        experiment_id = experiment_manager.current_experiment

        # ExperimentManager now stores the ObservablesPipeline; runner will consume it.
        runner = HDMFSimulationRunner(project_root, experiment_manager)
        runner.run_experiment()
        print(f"\n✅ Experiment completed successfully!")
        print(f"Experiment ID: {experiment_id}")
        print(f"Results directory: {experiment_dir}")

    except Exception as e:
        print(f"❌ Experiment failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
