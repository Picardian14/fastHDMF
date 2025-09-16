"""
HDMF simulation runner with experiment management
"""
import sys
from pathlib import Path
import numpy as np
import time
import pickle
import tempfile
import os
import shutil
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import product, islice
from typing import Any, Dict, Tuple, Optional, Union
from collections import defaultdict

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules
from utils.data_loading import load_metadata, load_all_sc_matrices
from src.experiment_manager import ExperimentManager
from src.helper_functions import filter_bold
from src.observables import ObservablesPipeline

# Import HDMF (assuming it's available in the environment)
try:
    import fastdyn_fic_dmf as dmf
except ImportError:
    print("Warning: fastdyn_fic_dmf not available. Make sure you're in the correct environment.")
    dmf = None

class HDMFSimulationRunner:
    """Runs HDMF simulations with experiment management"""

    def __init__(self, project_root: Path, experiment_manager: ExperimentManager, observables: Optional[ObservablesPipeline] = None):
        self.project_root = project_root
        self.experiment_manager = experiment_manager
        # Prefer pipeline from ExperimentManager; allow explicit override
        if observables is not None:
            self.observables = observables
        elif getattr(self.experiment_manager, "observables", None) is not None:
            self.observables = self.experiment_manager.observables
        else:
            self.observables = ObservablesPipeline.default()

        # Load patient data once
        self.metadata = load_metadata(
            metadata_file=self.experiment_manager.current_conf.get('data', {}).get('metadata', None),
            sc_root=self.experiment_manager.current_conf.get('data', {}).get('sc_root', 'SCs')
        )
        self.all_ipps = self.metadata['IPP'].tolist()        

    def _generate_parameter_values(self, spec: dict) -> np.ndarray:
        """Generate parameter values from spec with optional custom function"""
        if "fun" in spec:
            fun_name = spec["fun"]
            args = spec.get("args", [])
            kwargs = spec.get("kwargs", {})
            
            try:
                func = eval(fun_name)
                values = func(*args, **kwargs)
                return np.array(values, dtype=float)
            except Exception as e:
                raise ValueError(f"Error calling function '{fun_name}': {e}")
        else:
            # Default: start, end, step
            values = np.arange(spec["start"], spec["end"], spec["step"], dtype=float)
            return np.round(values, 12)

    def _contiguous_block(self, total: int, parts: int, k: int) -> tuple[int, int]:
        base, rem = divmod(total, parts)
        if k < rem:
            start = k * (base + 1)
            end   = start + (base + 1)
        else:
            start = rem * (base + 1) + (k - rem) * base
            end   = start + base
        return start, end

    def define_tasks_from_config(self, config: dict):
        grid = config.get("grid")
        sim_defaults = config.get("simulation", {})

        if not grid:
            # Single task, no grid.
            self._axis_names = ['task']
            self._axis_values = [np.array([0.0])]
            self._global_total_combos = 1
            yield sim_defaults.copy()
            return

        axis_names = list(grid.keys())
        # Generate parameter values using the new utility function
        axis_values = [self._generate_parameter_values(spec) for spec in grid.values()]

        self._axis_names = axis_names
        self._axis_values = axis_values
        self._global_total_combos = int(np.prod([len(v) for v in axis_values]))

        job_id   = getattr(self.experiment_manager, 'job_id', None)
        job_count = getattr(self.experiment_manager, 'job_count', None)

        combo_iter = product(*axis_values)
        if job_id is not None and job_count is not None:
            start, end = self._contiguous_block(self._global_total_combos, job_count, job_id)
            combo_iter = islice(combo_iter, start, end)

        for combo in combo_iter:
            task = sim_defaults.copy()
            task.update({name: float(val) for name, val in zip(axis_names, combo)})
            yield task



    def prepare_hdmf_params(self, task, seed=1):
        """Prepare HDMF parameters from config and SC matrix"""
               
        # Base parameters
        params = dmf.default_params(C=task['sc_matrix'])
        params['N'] = task['sc_matrix'].shape[0]
        if 'seed' in task:
            params['seed'] = int(task['seed'])
        if task is None:
            print( "No task provided, returning default params only.")
            return params
        # Configure from config file
        params['obj_rate'] = task['obj_rate']
        # NVC sigmoid option
        params['nvc_sigmoid'] = True
        params['nvc_r0']         =  params['obj_rate']  # baseline firing-rate
        params['nvc_u50']         = 12.0   # half-saturation (Hz): compression starts in this range
        params['nvc_match_slope'] = False # if using sigmoid, match slope at obj_rate
        params['nvc_k']          = 0.20    # maximum vasodilatory signal gain                    
        
        
        params["with_decay"] = task['with_decay']
        params["with_plasticity"] = task['with_plasticity']
    
        if 'lrj' in task:
            LR = task['lrj']
            if 'taoj' not in task:
                # Load homeostatic parameters
                fit_res_path = Path(__file__).parent.parent / "data" / "fit_res_3-44.npy"
                fit_res = np.load(str(fit_res_path))
                b = fit_res[0]
                a = fit_res[1]
                DECAY = np.exp(a + np.log(LR) * b) if task['with_decay'] else 0
            else:
                DECAY = task['taoj']

            # Makes decay and lr heterogenizable, as J is.
            params['taoj_vector'] = np.ones(params['N']) * DECAY
            params['lr_vector'] = np.ones(params['N']) * LR
            
        # Global coupling
        params['G'] = task['G']
        if 'alpha' in task:
            params['alpha'] = task['alpha']
            params['J'] = params['alpha'] * params['G'] * params['C'].sum(axis=0).squeeze() + 1
        params['TR'] = task['TR']
        
        # Neuromodulation
        if 'wgaine' in task:
            params['wgaine'] = task['wgaine']
            # By defualt, set inhibitory gain to match excitatory gain
            params['wgaini'] = task['wgaine']
        if 'wgaini' in task:
            params['wgaini'] = task['wgaini']
        

        # Return settings
        # If bold in any observable signal         
        params["return_bold"] = self.observables.needs("bold")
        params["return_fic"] = self.observables.needs("fic")
        params["return_rate"] = self.observables.needs("rates")
        
        
        return params
    
    def define_items_over(self, config: dict):
        """
        Define items over to process with a given task. Can be or not parallelized.
        Returns a list of (ipp, item) tuples where item contains sc_matrix and parameter values.
        """
        sim = config.get("simulation", {})
        over_config = sim.get('over')
        sc_root = config.get('data', {}).get('sc_root', 'SCs')
        
        # Load SC matrices based on test mode
        if config['data']['test_mode']:
            test_ipps = self.all_ipps[:config['data'].get('max_subjects_test', 2)]
            sc_matrices = load_all_sc_matrices(test_ipps, sc_root=sc_root)
        else:
            sc_matrices = load_all_sc_matrices(self.all_ipps, sc_root=sc_root)
        
        items = []
        
        if over_config is None:
            # No 'over' specified, just iterate over patients
            for ipp, sc_matrix in sc_matrices.items():
                item = {'sc_matrix': sc_matrix}
                items.append((ipp, item))
        else:

            # Assume that you iterate over 1 parameter (x amount of SC_matrices)
            param_name = list(over_config.keys())[0]
            param_values = self._generate_parameter_values(over_config[param_name])

            # Cartesian product between patients and parameter values
            for ipp, sc_matrix in sc_matrices.items():
                for param_value in param_values:
                    item = {
                        'sc_matrix': sc_matrix,
                        'param_key': param_name,
                        'param_value': param_value

                    }
                    items.append((ipp, item))
        
        return items
        

    def run_one_simulation(self, task: dict, config: dict, nb_steps: int, seed: int = 1) -> dict:
        """Run a single HDMF simulation with given SC matrix and task parameters"""
        params = self.prepare_hdmf_params(task, seed=seed+1)
        # Run the simulation
        rates_dyn, _, bold_dyn, fic_t_dyn = dmf.run(params, nb_steps)
        outputs = {}
        # Minimal processing on outputs
        if params.get('return_rate', True):
            rates_dyn = rates_dyn[:, int(config['simulation']['burnout'] * (params['TR'] / params['dtt'])):]            
            outputs['rates'] = rates_dyn
        if params.get('return_bold', True):
            bold_dyn = bold_dyn[:, config['simulation']['burnout']:]
            bold_dyn = filter_bold(bold_dyn, flp=0.008, fhp=0.09, tr=params['TR'])            
            outputs['bold'] = bold_dyn
        if params.get('return_fic', True):
            fic_t_dyn = fic_t_dyn[:, int(config['simulation']['burnout'] * (params['TR'] / params['dtt'])):]
            outputs['fic'] = fic_t_dyn
        # Once desired variables are ready to be observed, compute observables
        obs_dict = self.observables.compute(outputs, params, config)
        return obs_dict

    def run_experiment(self):
        """
        Simple + in-place integration:
        - Precompute patients and tasks
        - For each task, run all patients (optionally in parallel)
        - Write each observable directly into its preallocated grid at [task_idx, patient_idx]
        - Save once
        """
        em = self.experiment_manager
        log = em.logger
        config = em.current_conf
        sim = config.get("simulation", {})
        nb_steps = int(sim.get("nb_steps", 1000))
        parallel = bool(sim.get("parallel", False))
        items = self.define_items_over(config)
        n_items = len(items)        
        log.info(f"Loaded {n_items} patients.")

        # --- Build tasks (Cartesian product + optional job slicing) ---
        tasks = list(self.define_tasks_from_config(config))  # keep as list; we need local_task_count
        local_task_count = len(tasks)
        log.info(f"Local task count: {local_task_count}")

        axis_names = getattr(self, "_axis_names", [])
        axis_values = getattr(self, "_axis_values", [])
        global_total = int(getattr(self, "_global_total_combos", local_task_count))

        # --- Observable grids: lazily create per key, fill in-place ---
        # shape per grid: (local_task_count, n_items), dtype=object
        observable_grids: Dict[str, np.ndarray] = {}
        failed_simulations = set()

        # --- Main loop: fill grids directly ---        
        def get_grid(obs_key: str) -> np.ndarray:
            """Create the grid on first use; then reuse (no temp rows)."""
            g = observable_grids.get(obs_key)
            if g is None:
                g = np.empty((local_task_count, n_items), dtype=object)
                g[:] = None  # explicit init
                observable_grids[obs_key] = g
            return g

        # Determine parallel job cap: use user-specified override if provided, else default 32
        max_jobs = getattr(self.experiment_manager, 'max_jobs', None)
        cap = max_jobs if (isinstance(max_jobs, int) and max_jobs > 0) else 32
        n_jobs = min(len(items), min(cap, os.cpu_count() or 1))
        # --- Helpers ---
        def _run_one(current_task: Dict[str, Any], i: int, ipp: str, item_value: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
            try:
                # If there is a param to iterate over, set it here
                thread_task = current_task.copy()  # capture current task from outer loop
                thread_task['sc_matrix'] = item_value['sc_matrix']
                if item_value.get('param_key') is not None:
                    thread_task[item_value['param_key']] = item_value['param_value']
                out = self.run_one_simulation(
                    task=thread_task,   # captured from loop below
                    config=config,
                    nb_steps=nb_steps,
                    seed=i + 1,
                )
                # expected: dict {obs_key: value}
                return i, out
            except Exception as e:
                log.error(f"[{ipp}] simulation failed: {e}")
                return ipp, None


        for t_idx, current_task in enumerate(tasks):
            log.info(f"[Task {t_idx+1}/{local_task_count}] {current_task}")

            # Run all patients for this task
            if parallel and n_jobs > 1:
                pairs = Parallel(n_jobs=n_jobs, prefer="processes")(
                    delayed(_run_one)(current_task, i, ipp, item) for i, (ipp, item) in enumerate(items)
                )
            else:
                pairs = [_run_one(current_task, i, ipp, item) for i, (ipp, item) in enumerate(items)]

            # Write results directly into grids
            for item_idx, out in pairs:
                ipp = items[item_idx][0]
                if out is None:
                    failed_simulations.add(ipp)
                    # leave cells as None for all observables
                    continue
                for obs_key, value in out.items():
                    grid = get_grid(obs_key)
                    grid[t_idx, item_idx] = value

        # --- Package + save once ---
        axis_values_dict = {name: np.array(vals) for name, vals in zip(axis_names, axis_values)}
        meta = {
            "job_id": getattr(em, "job_id", None),
            "job_count": getattr(em, "job_count", None),
            "local_task_count": local_task_count,
            "global_total_combos": global_total,
            "partition_strategy": "contiguous_by_job_id",
            "observables_spec": list(observable_grids.keys()),
            "items": items,
        }

        results = {
            "param_axes": axis_names,
            "axis_values": axis_values_dict,            
            "observables": observable_grids,  # dict[str, np.ndarray] shape (tasks, items)
            "failed_simulations": sorted(failed_simulations),
            "meta": meta,
        }

        em.save_results(results=results, config=config, failed_simulations=results["failed_simulations"])
        log.info("Experiment finished and results saved.")
        return results