import numpy as np
from pathlib import Path
import sys

"""
Run from terminal:
PYTHONPATH=src python -m pytest -q src/modeling/HDMF/tests/test_observables_setup.py -k test_pipeline_multi_observables_compute
or
PYTHONPATH=src python -m pytest -q src/modeling/HDMF/tests/test_observables_setup.py -k test_pipeline_from_config_defaults_to_fc_bold
"""

project_root = Path(__file__).parent.parent
print(project_root)
sys.path.insert(0, str(project_root))

from src.observables import ObservablesPipeline

def test_pipeline_from_config_defaults_to_fc_bold():
    pipeline = ObservablesPipeline.from_config(None)
    spec = pipeline.spec()
    assert isinstance(spec, list) and len(spec) == 1
    assert spec[0]["name"] == "fc"
    assert spec[0]["signal"] == ["bold"]


def test_pipeline_multi_observables_compute():
    cfg = {
        "observables": [
            {"name": "fc", "signal": ["bold", "rates"], "params": {"zero_diag": True}},
            {"name": "mean", "signal": ["rates"]},
            {"name": "raw", "signal": ["fic"]},
        ]
    }
    pipeline = ObservablesPipeline.from_config(cfg)

    N, T = 4, 10
    outputs = {
        "bold": np.random.randn(N, T),
        "rates": np.random.randn(N, T),
        "fic": np.random.randn(N, T),
    }

    obs = pipeline.compute(outputs, params=None, config=None)
    # Expect keys for each requested observable
    assert "fc_bold" in obs and obs["fc_bold"].shape == (N, N)
    assert "fc_rates" in obs and obs["fc_rates"].shape == (N, N)
    assert "mean_rates" in obs and obs["mean_rates"].shape == (N,)
    assert "raw_fic" in obs and obs["raw_fic"].shape == (N, T)
