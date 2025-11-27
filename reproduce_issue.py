
import sys
import os
import numpy as np
import json

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from backend.sim import simulate_netlist
from backend.features import extract_features
from backend.ml import load_model, predict_fault, train_svm
from backend.demo_data import generate_training_dataset, generate_per_class_waveforms

def test_simulation():
    print("Testing Simulation...")
    netlist = """V1 in 0 SIN(0 10 50)
R1 in 1 1000
R2 1 2 1000
L1 1 0 2
C1 2 0 6u
.tran 20m 0 0.01m
.END"""
    
    try:
        sim_output = simulate_netlist(netlist)
        print("Simulation successful.")
        print(f"Time points: {len(sim_output['t'])}")
        print(f"V0 range: {min(sim_output['v0'])} to {max(sim_output['v0'])}")
        print(f"V1 range: {min(sim_output['v1'])} to {max(sim_output['v1'])}")
        
        return sim_output, netlist
    except Exception as e:
        print(f"Simulation failed: {e}")
        return None, None

def test_waveforms(netlist):
    print("\nTesting Waveform Generation...")
    try:
        waveforms = generate_per_class_waveforms(netlist)
        print(f"Generated waveforms for classes: {list(waveforms.keys())}")
        for cls, data in waveforms.items():
            print(f"  {cls}: {len(data['t'])} points, V0 range [{min(data['v0']):.3f}, {max(data['v0']):.3f}]")
    except Exception as e:
        print(f"Waveform generation failed: {e}")

def test_distribution(netlist):
    print("\nTesting Distribution Data (Phasor: Real vs Imag)...")
    try:
        X, y, dist = generate_training_dataset(netlist, N=5)
        print("Sample distribution points (Real, Imag):")
        for cls in ['E0', 'E1', 'E2', 'E3', 'E4', 'E5']:
            points = dist[cls]
            avg_x = np.mean([p[0] for p in points])
            avg_y = np.mean([p[1] for p in points])
            print(f"  {cls}: Avg Real={avg_x:.3f}, Avg Imag={avg_y:.3f}")
    except Exception as e:
        print(f"Distribution generation failed: {e}")

if __name__ == "__main__":
    sim_output, netlist = test_simulation()
    if netlist:
        test_waveforms(netlist)
        test_distribution(netlist)
