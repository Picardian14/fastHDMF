#!/usr/bin/env python3
"""
Test script to verify custom parameter generation functionality
"""
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation_runner import HDMFSimulationRunner
from src.experiment_manager import ExperimentManager

def test_parameter_generation():
    """Test the _generate_parameter_values method"""
    
    # Create a dummy experiment manager
    em = ExperimentManager(project_root)
    
    # Create simulation runner
    runner = HDMFSimulationRunner(project_root, em)
    
    print("Testing parameter generation...")
    
    # Test 1: Traditional start/end/step
    spec1 = {"start": 0, "end": 10, "step": 2}
    values1 = runner._generate_parameter_values(spec1)
    print(f"Linear spacing: {values1}")
    expected1 = np.array([0, 2, 4, 6, 8])
    assert np.allclose(values1, expected1), f"Expected {expected1}, got {values1}"
    
    # Test 2: Logarithmic spacing
    spec2 = {"fun": "np.logspace", "args": [0, 2, 5]}
    values2 = runner._generate_parameter_values(spec2)
    print(f"Log spacing: {values2}")
    expected2 = np.logspace(0, 2, 5)  # 1, ~3.16, 10, ~31.6, 100
    assert np.allclose(values2, expected2), f"Expected {expected2}, got {values2}"
    
    # Test 3: Linear spacing with exact number of points
    spec3 = {"fun": "np.linspace", "args": [0, 1], "kwargs": {"num": 6, "endpoint": True}}
    values3 = runner._generate_parameter_values(spec3)
    print(f"Linear spacing with num: {values3}")
    expected3 = np.linspace(0, 1, 6, endpoint=True)  # 0, 0.2, 0.4, 0.6, 0.8, 1
    assert np.allclose(values3, expected3), f"Expected {expected3}, got {values3}"
    
    # Test 4: Error handling - missing required parameters
    try:
        spec4 = {"start": 0, "end": 10}  # Missing step
        runner._generate_parameter_values(spec4)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Correctly caught error: {e}")
    
    # Test 5: Error handling - invalid function
    try:
        spec5 = {"fun": "invalid.function", "args": [1, 2, 3]}
        runner._generate_parameter_values(spec5)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Correctly caught error: {e}")
    
    print("\nAll tests passed! ✓")

if __name__ == "__main__":
    test_parameter_generation()