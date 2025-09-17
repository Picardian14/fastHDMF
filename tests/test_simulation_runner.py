import os
from pathlib import Path
import types
import numpy as np
import pytest
import sys



project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
print(project_root)
from src.simulation_runner import HDMFSimulationRunner
from src.observables import ObservablesPipeline

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
    def __init__(self, config, experiment_dir, job_id=None, job_count=None):
        self.current_config = config
        self.experiment_dir = str(experiment_dir)
        self.logger = DummyLogger()
        self.job_id = job_id
        self.job_count = job_count
        self.saved = []

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
        "data": {"test_mode": True},
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
    - metadata loader (returns ipps)
    - SC matrices loader (returns identity-like matrices)
    - dmf module (default_params, run)
    - filter_bold (identity)
    Also returns the created sc_matrices dict.
    """
    if ipps is None:
        ipps = ["P1", "P2"]

    # metadata
    monkeypatch.setattr(
        "modeling.HDMF.src.simulation_runner.load_metadata",
        lambda: FakeMeta(ipps),
        raising=True,
    )

    # SC matrices: accept new kwargs from simulation_runner (sc_root, folders, etc.)
    def _load_all_sc_matrices(sel_ipps, datapath=None, sc_root=None, folders=None, **kwargs):
        out = {}
        for ipp in sel_ipps:
            # simple symmetric connectivity
            C = np.eye(N)
            out[ipp] = C
        return out

    monkeypatch.setattr(
        "modeling.HDMF.src.simulation_runner.load_all_sc_matrices",
        _load_all_sc_matrices,
        raising=True,
    )

    # dmf
    if dmf is None:
        dmf = make_dummy_dmf(N=N)
    monkeypatch.setattr(
        "modeling.HDMF.src.simulation_runner.dmf",
        dmf,
        raising=True,
    )

    # filter_bold -> identity
    monkeypatch.setattr(
        "modeling.HDMF.src.simulation_runner.filter_bold",
        lambda x, flp, fhp, tr: x,
        raising=True,
    )

    return _load_all_sc_matrices(ipps)


def test_define_tasks_no_grid(monkeypatch, tmp_path, base_config):
    # Remove grid for this test
    config = dict(base_config)
    config.pop("grid", None)

    # Mocks (dmf/filter not needed here, but metadata access happens in constructor)
    install_mocks(monkeypatch, tmp_path)

    em = DummyExperimentManager(config=config, experiment_dir=tmp_path)
    runner = HDMFSimulationRunner(project_root=tmp_path, experiment_manager=em)

    tasks = runner.define_tasks_from_config(config)
    assert isinstance(tasks, list)
    assert len(tasks) == 1
    assert runner._axis_names == ["task"]
    assert runner._axis_lengths == [1]
    assert runner._global_total_combos == 1
    assert runner._global_linear_indices == [0]
    assert runner._global_multi_indices == [(0,)]
    assert runner._param_key_list == [(("task", 0.0),)]


def test_define_tasks_with_grid_full(monkeypatch, tmp_path, base_config):
    # Add a small grid: 2x3 = 6 combos
    config = dict(base_config)
    config["grid"] = {
        "G": {"start": 1.0, "end": 3.0, "step": 1.0},      # 1.0, 2.0
        "alpha": {"start": 0.1, "end": 0.4, "step": 0.1},  # 0.1, 0.2, 0.3
    }

    install_mocks(monkeypatch, tmp_path)

    em = DummyExperimentManager(config=config, experiment_dir=tmp_path, job_id=None, job_count=None)
    runner = HDMFSimulationRunner(project_root=tmp_path, experiment_manager=em)

    tasks = runner.define_tasks_from_config(config)
    assert len(tasks) == 6
    # Ensure each task has G and alpha overridden from the grid
    G_vals = sorted({t["G"] for t in tasks})
    alpha_vals = sorted({t["alpha"] for t in tasks})
    assert G_vals == [1.0, 2.0]
    assert alpha_vals == [0.1, 0.2, 0.3]
    assert runner._global_total_combos == 6
    assert len(runner._global_linear_indices) == 6
    assert len(runner._param_key_list) == 6

def test_define_tasks_with_grid_partition(monkeypatch, tmp_path, base_config):
    # Grid with 12 combos: 3x4
    config = dict(base_config)
    config["grid"] = {
        "G": {"start": 1.0, "end": 4.0, "step": 1.0},      # 1,2,3
        "alpha": {"start": 0.1, "end": 0.5, "step": 0.1},  # 0.1,0.2,0.3,0.4
    }
    install_mocks(monkeypatch, tmp_path)

    # Partition across 3 jobs; pick job_id=1 (0-based): expected chunk ~4 items
    em = DummyExperimentManager(config=config, experiment_dir=tmp_path, job_id=1, job_count=3)
    runner = HDMFSimulationRunner(project_root=tmp_path, experiment_manager=em)
    tasks = runner.define_tasks_from_config(config)
    # 12 combos split into 3 contiguous blocks => 4 for each if divisible
    assert len(tasks) == 4
    # Recorded linear indices should correspond to a middle contiguous block
    assert all(0 <= idx < 12 for idx in runner._global_linear_indices)
    # Ensure params come from the specified grid domain
    for t in tasks:
        assert t["G"] in {1.0, 2.0, 3.0}
        assert t["alpha"] in {0.1, 0.2, 0.3, 0.4}


def test_run_one_simulation_observables_and_burnout(monkeypatch, tmp_path, base_config):
    # Configure dummy dmf with controlled sizes
    N = 3
    T_rate = 1000
    T_bold = 100
    T_fic = 1000
    dmf = make_dummy_dmf(N=N, T_rate=T_rate, T_bold=T_bold, T_fic=T_fic, dtt=0.1)
    install_mocks(monkeypatch, tmp_path, dmf=dmf, ipps=["P1"], N=N)

    em = DummyExperimentManager(config=base_config, experiment_dir=tmp_path)
    # Build a pipeline: FC for bold and rates; keep raw fic
    pipeline = ObservablesPipeline.from_config({
        "observables": [
            {"name": "fc", "signal": ["bold", "rates"], "params": {"zero_diag": True}},
            {"name": "raw", "signal": ["fic"]},
        ]
    })
    runner = HDMFSimulationRunner(project_root=tmp_path, experiment_manager=em, observables=pipeline)

    # Single patient SC
    C = np.eye(N)
    task = dict(base_config["simulation"])  # runner copies from config["simulation"]
    config = base_config
    nb_steps = config["simulation"]["nb_steps"]

    out = runner.run_one_simulation(C, task, config, nb_steps, seed=123)

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

    em = DummyExperimentManager(config=config, experiment_dir=tmp_path)
    pipeline = ObservablesPipeline.from_config({
        "observables": [
            {"name": "fc", "signal": ["bold"]},
            {"name": "raw", "signal": ["fic"]},
        ]
    })
    runner = HDMFSimulationRunner(project_root=tmp_path, experiment_manager=em, observables=pipeline)

    results = runner.run_experiment(experiment_id="test")

    # Save must have been called once
    assert len(em.saved) == 1

    # Observables grids exist and have shape (local_tasks=1, n_patients=2)
    assert "observables" in results and isinstance(results["observables"], dict)
    obs_grids = results["observables"]
    assert "fc_bold" in obs_grids and obs_grids["fc_bold"].shape == (1, len(sc_matrices))
    assert "raw_fic" in obs_grids and obs_grids["raw_fic"].shape == (1, len(sc_matrices))

    # Failed simulations list present
    assert "failed_simulations" in results
    assert isinstance(results["failed_simulations"], list)

    # Temp directory should be gone (created under system TMP; verify no tmp_param files remain in experiment_dir)
    leftover = [p for p in os.listdir(tmp_path) if p.startswith("hdmf_exp_test_") or p.startswith("tmp_param_")]
    assert leftover == []


def test_integrator_grid_4x4_distributed_across_4_jobs(monkeypatch, tmp_path, base_config):
    """
    Verify integration with a 4x4 grid split across 4 pseudo jobs (contiguous partitioning):
    - Each job should receive 4 tasks (16 total combos / 4 jobs).
    - The union of global linear indices across jobs covers 0..15 with no overlap.
    - Each job integrates its local results into observable grids with shape (4, n_patients).
    """
    # Small, fast dummy DMF
    N = 2
    dmf = make_dummy_dmf(N=N, T_rate=50, T_bold=8, T_fic=50, dtt=0.1)
    install_mocks(monkeypatch, tmp_path, dmf=dmf, ipps=["P1", "P2"], N=N)

    # Build a 4x4 grid (G: 1..4, alpha: 0.1..0.4)
    config = dict(base_config)
    config["grid"] = {
        "G": {"start": 1.0, "end": 5.0, "step": 1.0},         # values: 1.0, 2.0, 3.0, 4.0
        "alpha": {"start": 0.1, "end": 0.5, "step": 0.1},    # values: 0.1, 0.2, 0.3, 0.4
    }
    # Sequential mode to keep it simple and deterministic
    config["simulation"]["parallel"] = False

    pipeline = ObservablesPipeline.from_config({
        "observables": [
            {"name": "fc", "signal": ["bold"], "params": {"zero_diag": True}},
        ]
    })

    all_indices = []
    all_param_keys = set()
    job_results = []

    for job_id in range(4):
        em = DummyExperimentManager(config=config, experiment_dir=tmp_path, job_id=job_id, job_count=4)
        runner = HDMFSimulationRunner(project_root=tmp_path, experiment_manager=em, observables=pipeline)

        tasks = runner.define_tasks_from_config(config)
        assert len(tasks) == 4

        # Record indices from define step
        idxs = tuple(runner._global_linear_indices)
        assert len(idxs) == 4
        all_indices.extend(idxs)

        # Run and validate integration metadata and shapes
        res = runner.run_experiment(experiment_id=f"job{job_id}")
        job_results.append(res)

        assert res["meta"]["local_task_count"] == 4
        assert res["meta"]["global_total_combos"] == 16
        assert sorted(res["global_linear_indices"]) == sorted(idxs)

        # Two patients from our mock
        n_patients = 2
        assert "observables" in res and isinstance(res["observables"], dict)
        for k, grid in res["observables"].items():
            assert grid.shape == (4, n_patients)
            # spot-check one cell has an ndarray output
            assert grid[0, 0] is None or hasattr(grid[0, 0], "shape")

        # Collect param keys for uniqueness check across jobs
        for pk in res.get("param_keys_local", []):
            all_param_keys.add(tuple(pk))

    # Across all 4 jobs: coverage should be exact and disjoint
    assert len(all_indices) == 16
    assert sorted(set(all_indices)) == list(range(16))
    assert len(all_param_keys) == 16


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
    M = N * (N - 1) // 2
    expected_wins = len(np.arange(0, T - win - 1, win - overlap))

    assert fcd.shape == (M, expected_wins)
    assert np.isfinite(fcd).all()