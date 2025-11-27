
import numpy as np
import pandas as pd
from backend.sim import simulate_netlist, _simulate_deterministic
from backend.features import extract_features
from backend.ml import train_svm
from backend.utils import inject_fault
import time

def main():
    print("==================================================")
    print("   PHASOR-BASED FAULT CLASSIFICATION (SVM)        ")
    print("==================================================")
    print("Features: [Real(V0), Imag(V0), Real(V1), Imag(V1)]")
    
    # 1. Setup
    base_netlist = """V1 in 0 SIN(0 10 50)
R1 in 1 1000
R2 1 2 1000
L1 1 0 2
C1 2 0 6u
.tran 20m 0 0.01m
.END"""

    FAULT_CLASSES = ['E0', 'E1', 'E2', 'E3', 'E4', 'E5']
    SAMPLES_PER_CLASS = 60
    
    X_list = []
    y_list = []
    
    print(f"\n[1/3] Generating Dataset ({SAMPLES_PER_CLASS} samples/class)...")
    start_time = time.time()
    
    # 2. Generate Data
    t_eval = np.linspace(0, 0.04, 400) # 2 cycles
    
    for class_idx, fault_class in enumerate(FAULT_CLASSES):
        print(f"  Generating {fault_class}...", end='\r')
        for i in range(SAMPLES_PER_CLASS):
            # Inject fault
            netlist = inject_fault(base_netlist, fault_class, seed=i)
            
            # Parse components for fast simulation
            comps = {}
            for line in netlist.split('\n'):
                parts = line.split()
                if len(parts) >= 4:
                    try: comps[parts[0]] = float(parts[3])
                    except: pass
            
            R1 = comps.get('R1', 1000.0)
            R2 = comps.get('R2', 1000.0)
            L1 = comps.get('L1', 2.0)
            C1 = comps.get('C1', 6e-6)
            
            # Simulate
            sim_output = _simulate_deterministic(R1, R2, C1, L1, t_eval)
            
            # Extract Phasor Features
            features = extract_features(sim_output)
            
            X_list.append(features)
            y_list.append(class_idx)
            
    print(f"  ✓ Dataset generated in {time.time() - start_time:.2f}s")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"  Dataset Shape: {X.shape}")
    
    # 3. Train SVM
    print("\n[2/3] Training SVM...")
    model, metrics = train_svm(X, y)
    
    # 4. Results
    print("\n[3/3] Results")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print("\n  Confusion Matrix:")
    print(np.array(metrics['confusion']))
    
    # Verify separation
    print("\n  Feature Means (Real V0, Imag V0):")
    for i, cls in enumerate(FAULT_CLASSES):
        mask = (y == i)
        mean_feat = np.mean(X[mask], axis=0)
        print(f"    {cls}: ({mean_feat[0]:.2f}, {mean_feat[1]:.2f})")

if __name__ == "__main__":
    main()
