"""
Experiment management system for HDMF simulations
Handles configuration loading, logging, and result organization
"""
import yaml
import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import re
import psutil

from src.observables import ObservablesPipeline


class ExperimentManager:
    """Manages HDMF experiments with configuration, logging, and result storage"""
    
    def __init__(self, project_root: Path, config_path: Optional[str] = None, results_dir: Optional[Path] = None, job_id: Optional[int] = None, job_count: Optional[int] = None):
        self.project_root = Path(project_root)
        self.configs_dir = self.project_root / "configs"
        # unified results directory
        self.results_dir = Path(results_dir) if results_dir else (self.project_root / "results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # placeholders for current experiment
        self.current_experiment = None
        self.experiment_dir = None
        self.logger = None
        self.current_conf = None
        self.current_config_path: Optional[str] = None
        self.observables = None
        
        # if config_path provided, immediately set up experiment
        if config_path:
            self.setup_experiment(
                config_path,
                job_id=str(job_id) if job_id is not None else None,
                job_count_str=str(job_count) if job_count is not None else None
            )
    
    def integrate_slurm_results(self, experiment_id: str) -> Path:
        base_dir = self.results_dir / experiment_id
        if not base_dir.exists() or not base_dir.is_dir():
            raise ValueError(f"Base experiment directory not found: {base_dir}")

        job_folders = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('job_')]
        if not job_folders:
            raise ValueError(f"No job folders found inside base experiment directory: {base_dir}")

        # Numeric sort ensures global task order
        job_folders.sort(key=lambda x: int(x.name.split('_')[-1]))

        # Logging
        self.experiment_dir = base_dir
        self._setup_logging(experiment_id)
        self.logger.info(f"Integrating {len(job_folders)} job folders from {base_dir}")

        merged = None
        all_failed = set()

        for jf in job_folders:
            rp = jf / "full_results.pkl"
            if not rp.exists():
                self.logger.warning(f"Missing results in {jf}; skipping.")
                continue
            with open(rp, 'rb') as f:
                jr = pickle.load(f)

            # Collect failures if present
            all_failed.update(jr.get('failed_simulations', []))

            if merged is None:
                # First job: take structure as-is
                merged = jr
                continue

            # Concatenate observables along task axis (axis 0)
            tgt_obs = merged.get('observables', {})
            src_obs = jr.get('observables', {})
            for k, src in src_obs.items():
                if k in tgt_obs and getattr(tgt_obs[k], 'size', 0) > 0 and getattr(src, 'size', 0) > 0:
                    tgt_obs[k] = np.concatenate([tgt_obs[k], src], axis=0)
                elif getattr(src, 'size', 0) > 0:
                    tgt_obs[k] = src
            merged['observables'] = tgt_obs

            # Patients should be identical across jobs; keep the first
            # Axis values should be identical across jobs; keep the first

            # Update simple counters
            m = merged.setdefault('meta', {})
            m['num_tasks_integrated'] = m.get('num_tasks_integrated', 0) + jr.get('meta', {}).get('local_task_count', 0)

        # Finalize metadata
        merged = merged or {}
        merged.setdefault('meta', {})
        merged['meta'].update({
            'status': 'completed',
            'integrated_from_jobs': len(job_folders),
        })
        merged['failed_simulations'] = sorted(all_failed)

        # Save integrated
        outp = base_dir / "full_results.pkl"
        with open(outp, 'wb') as f:
            pickle.dump(merged, f)

        # Update / write metadata.json at base level
        meta_path = base_dir / "metadata.json"
        base_meta = {
            'experiment_id': experiment_id,
            'status': 'completed',
            'end_time': datetime.now().isoformat(),
            'total_patients': len(merged.get('patients', [])),
            'integrated_from_jobs': len(job_folders),
            'original_job_folders': [d.name for d in job_folders],
            'failed_simulations': merged.get('failed_simulations', []),
        }
        with open(meta_path, 'w') as f:
            json.dump(base_meta, f, indent=2)

        # Optionally archive job folders
        jobs_dir = base_dir / ".jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for jf in job_folders:
            try:
                shutil.move(str(jf), str(jobs_dir / jf.name))
            except Exception as e:
                self.logger.warning(f"Failed to move {jf.name}: {e}")

        self.logger.info(f"Integrated results saved to {outp}")
        return base_dir

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load experiment configuration from YAML file"""
        if not config_path.endswith('.yaml'):
            config_path += '.yaml'
            
        # Try different possible locations
        possible_paths = [
            self.project_root / config_path,
            self.configs_dir / config_path,
            self.configs_dir / "experiments" / config_path,
            Path(config_path)  # Absolute path
        ]
        
        for path in possible_paths:
            if path.exists():
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
                self.current_config_path = str(path)
                self.logger_info(f"Loaded config from: {path}")
                return config
                
        raise FileNotFoundError(f"Config file not found: {config_path}")

    def setup_experiment(self, config_path: str, job_id: Optional[str] = None, job_count_str: Optional[str] = None) -> Tuple[Optional[Path], str]:
        """Setup a new experiment with logging and result directories.           
        """
        config = self.load_config(config_path)
        self.current_conf = config    
        # Build and store the observables pipeline for this experiment
        try:
            self.observables = ObservablesPipeline.from_config(config.get("output", {}))
        except Exception:
            # Fallback to a safe default if config is malformed
            self.observables = ObservablesPipeline.default()
        # Determine experiment ID strictly from the config filename (stem)
        if self.current_config_path:
            cfg_path = Path(self.current_config_path)
        else:
            cfg_path = Path(config_path)
        experiment_id = cfg_path.stem

        # Determine if we will save anything; only then create directories
        out = (self.current_conf or {}).get('output', {})
        will_save = bool(out.get('save_full_outputs') or out.get('save_metrics_only') or out.get('save_plots'))

        self.experiment_dir = None
        if will_save:
            self.experiment_dir = Path(self.results_dir) / experiment_id
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
        # New directory layout for SLURM array jobs:
        #   Base experiment folder: <experiment_name>_<timestamp>
        #   Each job has subfolder: job_<id>
        # Single (non-array) runs keep old flat structure.
        if job_id is not None and will_save:
            #base_experiment_id = f"{experiment_id}_{timestamp}"
            self.base_experiment_id = experiment_id  # informational
            self.job_id = int(job_id)
            self.job_count = int(job_count_str) if job_count_str is not None else None

            base_dir = Path(self.results_dir) / experiment_id
            base_dir.mkdir(parents=True, exist_ok=True)

            job_folder = base_dir / f"job_{job_id}"
            job_folder.mkdir(parents=True, exist_ok=True)
            self.experiment_dir = job_folder

            # Maintain full experiment_id with job suffix for metadata/logging
            self.experiment_id = f"{experiment_id}_job{job_id}"

        # Setup logging (console-only if not saving)
        self._setup_logging(experiment_id)

    
        metadata = {
            'experiment_id': experiment_id,
            'original_config_path': self.current_config_path or str(config_path),
            'start_time': datetime.now().isoformat(),
            'config': self.current_conf,
            'status': 'initialized',
            'observables_spec': self.observables.spec() if self.observables else []
        }
        # Only persist metadata if saving
        if self.experiment_dir is not None:
            metadata_path = self.experiment_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        self.current_experiment = experiment_id
        self.logger.info(f"Experiment '{experiment_id}' initialized")
        desc = (config.get('experiment') or {}).get('description')
        if desc:
            self.logger.info(f"Config: {desc}")
        if self.experiment_dir is not None:
            self.logger.info(f"Results will be saved to: {self.experiment_dir}")
        else:
            self.logger.info("No save flags set; results will not be written to disk.")

        return self.experiment_dir, experiment_id
    
    def _setup_logging(self, experiment_id: str):
        """Setup logging for the experiment. If no experiment_dir (not saving), use console-only."""

        # Create logger
        self.logger = logging.getLogger(f"hdmf_experiment_{experiment_id}")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Optional file handler if saving to a directory
        if self.experiment_dir is not None:
            log_file = self.experiment_dir / "experiment.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
    def logger_info(self, message: str):
        """Log info message (creates basic logger if none exists)"""
        if self.logger:
            self.logger.info(message)
        else:
            print(f"INFO: {message}")
    
    def save_results(self, results: Dict[str, Any], config: Dict[str, Any], failed_simulations: Optional[List] = None):
        """Save experiment results"""
        out = (config or {}).get('output', {})
        will_save = bool(out.get('save_full_outputs') or out.get('save_metrics_only') or out.get('save_plots'))
        if not will_save:
            # Nothing to persist; just log and return
            self.logger_info("Run completed (no save flags set); skipping filesystem writes.")
            return self.experiment_dir
        if not self.experiment_dir:
            # Create directory on-demand if missing (should not happen if setup_experiment followed flags)
            exp_id = (self.current_config_path and Path(self.current_config_path).stem) or datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_dir = Path(self.results_dir) / exp_id
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            # Ensure logging to file is enabled now
            self._setup_logging(exp_id)
        
        # Save full results if requested
        if config['output']['save_full_outputs']:
            results_path = self.experiment_dir / "full_results.pkl"
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
            self.logger.info(f"Saved full results to: {results_path}")
        
        # Determine patient count from new structured results if present
        if isinstance(results, dict) and 'patients' in results:
            total_patients = len(results['patients'])
        else:
            # Fallback: try common patterns or length heuristic
            total_patients = len(results)

        # Update experiment metadata
        self._update_experiment_status('completed', {
            'total_patients': total_patients,
            'failed_simulations': failed_simulations or []
        })
        
        self.logger.info(f"Experiment completed successfully!")
        return self.experiment_dir
    
    def _update_experiment_status(self, status: str, additional_data: Dict[str, Any] = None):
        """Update experiment metadata"""
        if not self.experiment_dir:
            return
        metadata_path = self.experiment_dir / "metadata.json"
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        metadata['status'] = status
        metadata['end_time'] = datetime.now().isoformat()
        
        if additional_data:
            metadata.update(additional_data)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def list_experiments(self) -> pd.DataFrame:
        """List all completed experiments"""
        experiments = []
        
        for exp_dir in self.results_dir.iterdir():
            if exp_dir.is_dir():
                metadata_path = exp_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    experiments.append({
                        'experiment_id': metadata.get('experiment_id'),
                        'name': metadata.get('config', {}).get('experiment', {}).get('name'),
                        'description': metadata.get('config', {}).get('experiment', {}).get('description'),
                        'status': metadata.get('status'),
                        'start_time': metadata.get('start_time'),
                        'total_patients': metadata.get('total_patients'),
                        'directory': str(exp_dir)
                    })
        
        return pd.DataFrame(experiments)

    def load_experiment_results(self, experiment_id: str, is_done: bool) -> Dict[str, Any]:
        """Load results from a completed experiment. As with setup_experiment it sets the current results directory"""
        self.experiment_dir = Path(self.results_dir) / experiment_id if not is_done else Path(self.results_dir) / "done" / experiment_id

        if not self.experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {self.experiment_dir}")

        results = {}
        
        # Load metadata
        metadata_path = self.experiment_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                results['metadata'] = json.load(f)
        # Load configuration from the original YAML file rather than from metadata JSON
        original_config_path = results['metadata'].get('original_config_path')        
        self.current_conf = self.load_config(original_config_path)
        # Rebuild observables pipeline for any follow-up processing
        try:
            self.observables = ObservablesPipeline.from_config(self.current_conf.get("output", {}))
        except Exception:
            self.observables = ObservablesPipeline.default()
        
        # Load full results if available
        full_results_path = self.experiment_dir / "full_results.pkl"
        if full_results_path.exists():
            with open(full_results_path, 'rb') as f:
                results['full_results'] = pickle.load(f)
        
        return results
