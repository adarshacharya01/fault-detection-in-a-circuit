"""Training dataset generation."""
import json
import os
import numpy as np
from typing import Dict, List, Tuple
from .sim import simulate_netlist, _simulate_deterministic
from .features import extract_features, compute_scatter
from .utils import inject_fault

FAULT_CLASSES = ['E0', 'E1', 'E2', 'E3', 'E4', 'E5']

def generate_training_dataset(
    base_netlist: str,
    N: int = 60
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[List[float]]]]:
    """Generate training dataset matching IEEE paper specs."""
    print(f"Generating {N} samples per class for {FAULT_CLASSES}...")
    
    X_list = []
    y_list = []
    distribution = {cls: [] for cls in FAULT_CLASSES}
    
    # Simulation parameters for training (fast mode)
    t_end = 0.04
    points = 400
    t_eval = np.linspace(0, t_end, points)
    
    for class_idx, fault_class in enumerate(FAULT_CLASSES):
        for sample_idx in range(N):
            try:
                # Get fault parameters directly (bypass netlist string parsing for speed)
                # We need to extract R1, R2, L1, C1 from inject_fault logic
                # But inject_fault returns a string.
                # Let's parse the string back or modify inject_fault to return dict.
                # For now, let's just parse the string, it's fast enough.
                netlist = inject_fault(base_netlist, fault_class, seed=sample_idx)
                
                # Parse components manually for _simulate_deterministic
                comps = {}
                for line in netlist.split('\n'):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            comps[parts[0]] = float(parts[3])
                        except: pass
                
                R1 = comps.get('R1', 1000.0)
                R2 = comps.get('R2', 1000.0)
                L1 = comps.get('L1', 1e-3)
                C1 = comps.get('C1', 6e-6)
                
                # Simulate using FAST transfer function
                sim_output = _simulate_deterministic(R1, R2, C1, L1, t_eval)
                
                # Extract features (V0 + V1)
                features = extract_features(sim_output)
                
                # Scatter plot (Phasor: Real vs Imag)
                scatter_x, scatter_y = compute_scatter(sim_output['v0'], sim_output['t'])
                
                X_list.append(features)
                y_list.append(class_idx)
                distribution[fault_class].append([scatter_x, scatter_y])
                
            except Exception as e:
                print(f"Error generating sample {sample_idx} for {fault_class}: {e}")
                continue
                
    X = np.array(X_list)
    y = np.array(y_list)
    
    _save_distribution(distribution)
    return X, y, distribution

def _save_distribution(distribution):
    os.makedirs('backend/models', exist_ok=True)
    with open('backend/models/distribution.json', 'w') as f:
        json.dump(distribution, f)

def load_distribution():
    path = 'backend/models/distribution.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def generate_per_class_waveforms(base_netlist: str):
    waveforms = {}
    for cls in FAULT_CLASSES:
        netlist = inject_fault(base_netlist, cls, seed=0)
        waveforms[cls] = simulate_netlist(netlist)
    return waveforms
