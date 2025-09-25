import os
from pathlib import Path
import types
import numpy as np
import pytest
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastHDMF.simulation_runner import HDMFSimulationRunner
from fastHDMF.experiment_manager import ExperimentManager
from fastHDMF.observables import ObservablesPipeline

class DummyLogger:
    def __init__(self):
        self.infos = []
        self.errors = []
        self.warnings = []

    def info(self, msg):
        self.infos.append(str(msg))

    def error(self, msg):
        self.errors.append(str(msg))

    def warning(self, msg):
        self.warnings.append(str(msg))


class DummyExperimentManager:
    def __init__(self, config, project_root, data_dir=None, job_id=None, job_count=None, max_jobs=None):
        self.current_config = config
        self.project_root = Path(project_root)
        self.data_dir = Path(data_dir) if data_dir else self.project_root / "data"
        self.logger = DummyLogger()
        self.job_id = job_id
        self.job_count = job_count
        self.max_jobs = max_jobs
        self.saved = []
        # Mock sc_matrices based on test_mode
        self.sc_matrices = self._create_dummy_sc_matrices()
        
    def _create_dummy_sc_matrices(self):
        """Create dummy SC matrices for testing"""
        N = 3
        if self.current_config['data']['test_mode']:
            return {"P1": np.eye(N), "P2": np.eye(N)}
        else:
            return {"P1": np.eye(N), "P2": np.eye(N), "P3": np.eye(N)}
    
    @property
    def all_ipps(self):
        """Return list of all subject IDs"""
        return list(self.sc_matrices.keys())

    def save_results(self, results, config, failed_simulations):
        # record the save call
        self.saved.append((results, config, failed_simulations))


class FakeSeries(list):
    def tolist(self):
        return list(self)


class FakeMeta(dict):
    def __init__(self, ipps):
        super().__init__({"IPP": FakeSeries(ipps)})


def make_dummy_dmf(N=3, T_rate=1000, T_bold=100, T_fic=1000, dtt=0.1):
    """
    Create a dummy 'dmf' module-like object with default_params and run().
    """
    def default_params(C):
        return {
            "C": C,
            "dtt": dtt,
            # TR will be set from task in prepare_hdmf_params
        }

    def run(params, nb_steps):
        N_local = params["N"]
        # rates and fic simulated at dtt resolution; bold at TR resolution
        rates = np.random.RandomState(0).randn(N_local, T_rate)
        bold = np.random.RandomState(1).randn(N_local, T_bold)
        fic = np.random.RandomState(2).randn(N_local, T_fic)
        dummy_second = None
        return rates, dummy_second, bold, fic

    m = types.SimpleNamespace()
    m.default_params = default_params
    m.run = run
    return m


@pytest.fixture
def base_config():
    # minimal config supplying all fields used by the runner
    return {
        "data": {"test_mode": True, "sc_root": "AAL"},
        "simulation": {
            # values used as defaults for tasks
            "obj_rate": 3.0,
            "with_decay": True,
            "with_plasticity": True,
            "lrj": 0.01,
            "G": 1.5,
            "alpha": 0.2,
            "TR": 2.0,
            "return_bold": True,
            "return_fic": True,
            "return_rate": True,
            # runner controls
            "nb_steps": 1000,
            "burnout": 10,
            "parallel": False,
        },
        # No pre-FC transform; let observables compute FC
        "output": {
            "observables": [
                {"name": "raw", "signal": "bold"}
            ]
        }
    }


def install_mocks(monkeypatch, tmp_path, dmf=None, ipps=None, N=3):
    """
    Install monkeypatches for:
    - dmf module (default_params, run)
    - filter_bold (identity)
    """
    if ipps is None:
        ipps = ["P1", "P2"]

    # dmf
    if dmf is None:
        dmf = make_dummy_dmf(N=N)
    monkeypatch.setattr(
        "fastHDMF.simulation_runner.dmf",
        dmf,
        raising=True,
    )

    # filter_bold -> identity
    monkeypatch.setattr(
        "fastHDMF.simulation_runner.filter_bold",
        lambda x, flp, fhp, tr: x,
        raising=True,
    )

    return {ipp: np.eye(N) for ipp in ipps}


def test_define_tasks_no_grid(monkeypatch, tmp_path, base_config):
    # Remove grid for this test
    config = dict(base_config)
    config.pop("grid", None)

    # Mocks
    install_mocks(monkeypatch, tmp_path)

    em = DummyExperimentManager(config=config, project_root=tmp_path)
    runner = HDMFSimulationRunner(experiment_manager=em)

    tasks = list(runner.define_tasks_from_config(config))
    assert isinstance(tasks, list)
    assert len(tasks) == 1
    assert runner._axis_names == ["task"]
    assert runner._axis_values == [np.array([0.0])]
    assert runner._global_total_combos == 1
    # Check that the task contains the simulation defaults
    assert tasks[0]['obj_rate'] == config['simulation']['obj_rate']


def test_define_tasks_with_grid_full(monkeypatch, tmp_path, base_config):
    # Add a small grid: 2x3 = 6 combos
    config = dict(base_config)
    config["grid"] = {
        "G": {"fun": "np.arange", "args": [1.0, 3.0, 1.0]},      # 1.0, 2.0
        "alpha": {"fun": "np.arange", "args": [0.1, 0.4, 0.1]},  # 0.1, 0.2, 0.3
    }

    install_mocks(monkeypatch, tmp_path)

    em = DummyExperimentManager(config=config, project_root=tmp_path, job_id=None, job_count=None)
    runner = HDMFSimulationRunner(experiment_manager=em)

    tasks = list(runner.define_tasks_from_config(config))
    assert len(tasks) == 8  # 2 G values * 4 alpha values = 8
    # Ensure each task has G and alpha overridden from the grid
    G_vals = sorted({t["G"] for t in tasks})
    alpha_vals = sorted({t["alpha"] for t in tasks})
    assert G_vals == [1.0, 2.0]
    assert np.allclose(alpha_vals, [0.1, 0.2, 0.3, 0.4])
    assert runner._global_total_combos == 8

def test_define_tasks_with_grid_partition(monkeypatch, tmp_path, base_config):
    # Grid with 12 combos: 3x4
    config = dict(base_config)
    config["grid"] = {
        "G": {"fun": "np.arange", "args": [1.0, 4.0, 1.0]},      # 1,2,3
        "alpha": {"fun": "np.arange", "args": [0.1, 0.5, 0.1]},  # 0.1,0.2,0.3,0.4
    }
    install_mocks(monkeypatch, tmp_path)

    # Partition across 3 jobs; pick job_id=1 (0-based): expected chunk ~4 items
    em = DummyExperimentManager(config=config, project_root=tmp_path, job_id=1, job_count=3)
    runner = HDMFSimulationRunner(experiment_manager=em)
    tasks = list(runner.define_tasks_from_config(config))
    # Total combos: 3 G values * 4 alpha values = 12 combos
    # Split into 3 jobs => 4 for each job
    assert len(tasks) == 4
    # Ensure params come from the specified grid domain
    for t in tasks:
        assert t["G"] in {1.0, 2.0, 3.0}
        # Use np.isclose to handle floating point precision
        assert any(np.isclose(t["alpha"], val) for val in [0.1, 0.2, 0.3, 0.4])


def test_run_one_simulation_observables_and_burnout(monkeypatch, tmp_path, base_config):
    # Configure dummy dmf with controlled sizes
    N = 3
    T_rate = 1000
    T_bold = 100
    T_fic = 1000
    dmf = make_dummy_dmf(N=N, T_rate=T_rate, T_bold=T_bold, T_fic=T_fic, dtt=0.1)
    install_mocks(monkeypatch, tmp_path, dmf=dmf, ipps=["P1"], N=N)

    em = DummyExperimentManager(config=base_config, project_root=tmp_path)
    # Build a pipeline: FC for bold and rates; keep raw fic
    pipeline = ObservablesPipeline.from_config({
        "observables": [
            {"name": "fc", "signal": ["bold", "rates"], "params": {"zero_diag": True}},
            {"name": "raw", "signal": ["fic"]},
        ]
    })
    runner = HDMFSimulationRunner(experiment_manager=em, observables=pipeline)

    # Create task with SC matrix (following notebook example)
    task = dict(base_config["simulation"])
    task['sc_matrix'] = np.eye(N)  # Add SC matrix to task as shown in notebook
    task['seed'] = 123
    # Add taoj to avoid loading fit file
    task['taoj'] = 0.1

    out = runner.run_one_simulation(task)

    # Expect observables dict with FC for bold/rates and raw fic
    assert "fc_bold" in out and out["fc_bold"].shape == (N, N)
    assert "fc_rates" in out and out["fc_rates"].shape == (N, N)
    # zero diag for FC
    assert np.allclose(np.diag(out["fc_bold"]), 0.0)
    assert np.allclose(np.diag(out["fc_rates"]), 0.0)

    # fic trimmed by burnout * (TR/dtt) = 10 * (2.0 / 0.1) = 200 steps
    expected_fic_T = T_fic - 200
    assert "raw_fic" in out and out["raw_fic"].shape == (N, expected_fic_T)


def test_run_experiment_happy_path_integration(monkeypatch, tmp_path, base_config):
    # Prepare small dataset with 2 patients and small arrays
    N = 3
    dmf = make_dummy_dmf(N=N, T_rate=400, T_bold=30, T_fic=400, dtt=0.1)
    sc_matrices = install_mocks(monkeypatch, tmp_path, dmf=dmf, ipps=["P1", "P2"], N=N)

    # No grid for simplicity
    config = dict(base_config)
    config.pop("grid", None)
    # Force sequential to simplify
    config["simulation"]["parallel"] = False
    # Add taoj to avoid loading fit file
    config["simulation"]["taoj"] = 0.1

    em = DummyExperimentManager(config=config, project_root=tmp_path)
    pipeline = ObservablesPipeline.from_config({
        "observables": [
            {"name": "fc", "signal": ["bold"]},
            {"name": "raw", "signal": ["fic"]},
        ]
    })
    runner = HDMFSimulationRunner(experiment_manager=em, observables=pipeline)

    results = runner.run_experiment()

    # Save must have been called once
    assert len(em.saved) == 1

    # Observables grids exist and have shape (local_tasks=1, n_patients=2)
    assert "observables" in results and isinstance(results["observables"], dict)
    obs_grids = results["observables"]
    assert "fc_bold" in obs_grids and obs_grids["fc_bold"].shape == (1, 2)  # 1 task, 2 subjects
    assert "raw_fic" in obs_grids and obs_grids["raw_fic"].shape == (1, 2)

    # Failed simulations list present
    assert "failed_simulations" in results
    assert isinstance(results["failed_simulations"], list)


def test_integrator_grid_4x4_distributed_across_4_jobs(monkeypatch, tmp_path, base_config):
    """
    Verify integration with a 4x4 grid split across 4 pseudo jobs (contiguous partitioning):
    - Each job should receive 4 tasks (16 total combos / 4 jobs).
    - Each job integrates its local results into observable grids with shape (4, n_patients).
    """
    # Small, fast dummy DMF
    N = 2
    dmf = make_dummy_dmf(N=N, T_rate=50, T_bold=8, T_fic=50, dtt=0.1)
    install_mocks(monkeypatch, tmp_path, dmf=dmf, ipps=["P1", "P2"], N=N)

    # Build a 4x4 grid (G: 1..4, alpha: 0.1..0.4)
    config = dict(base_config)
    config["grid"] = {
        "G": {"fun": "np.arange", "args": [1.0, 5.0, 1.0]},         # values: 1.0, 2.0, 3.0, 4.0
        "alpha": {"fun": "np.arange", "args": [0.1, 0.5, 0.1]},    # values: 0.1, 0.2, 0.3, 0.4
    }
    # Sequential mode to keep it simple and deterministic
    config["simulation"]["parallel"] = False

    pipeline = ObservablesPipeline.from_config({
        "observables": [
            {"name": "fc", "signal": ["bold"], "params": {"zero_diag": True}},
        ]
    })

    job_results = []

    for job_id in range(4):
        em = DummyExperimentManager(config=config, project_root=tmp_path, job_id=job_id, job_count=4)
        runner = HDMFSimulationRunner(experiment_manager=em, observables=pipeline)

        tasks = list(runner.define_tasks_from_config(config))
        assert len(tasks) == 4

        # Run and validate integration metadata and shapes
        res = runner.run_experiment()
        job_results.append(res)

        assert res["meta"]["local_task_count"] == 4
        assert res["meta"]["global_total_combos"] == 16

        # Two patients from our mock
        n_patients = 2
        assert "observables" in res and isinstance(res["observables"], dict)
        for k, grid in res["observables"].items():
            assert grid.shape == (4, n_patients)
            # spot-check one cell has an ndarray output
            assert grid[0, 0] is None or hasattr(grid[0, 0], "shape")


def test_fcd_observable_shapes_and_keys():
    N, T = 5, 100
    win, overlap = 20, 10

    outputs = {
        "bold": np.random.randn(N, T),
    }

    pipeline = ObservablesPipeline.from_config({
        "observables": [
            {"name": "fcd", "signal": ["bold"], "params": {"window_size": win, "overlap": overlap}},
        ]
    })

    obs = pipeline.compute(outputs)
    assert "fcd_bold" in obs

    fcd = obs["fcd_bold"]
    # Check actual shape - it should be (M, M) where M is number of windows
    expected_wins = len(np.arange(0, T - win - 1, win - overlap))
    M = expected_wins  # FCD is typically (n_windows, n_windows)

    assert fcd.shape == (M, expected_wins) or fcd.shape == (expected_wins, expected_wins)
    assert np.isfinite(fcd).all()