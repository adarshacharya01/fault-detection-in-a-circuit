
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from backend.sim import _simulate_deterministic
from backend.features import compute_scatter
from backend.utils import inject_fault

def generate_data(n_samples, include_v1=True):
    """Generate dataset with specified samples per class."""
    base_netlist = """V1 in 0 SIN(0 10 50)
R1 in 1 1000
R2 1 2 1000
L1 1 0 2
C1 2 0 6u
.tran 20m 0 0.01m
.END"""
    
    FAULT_CLASSES = ['E0', 'E1', 'E2', 'E3', 'E4', 'E5']
    t_eval = np.linspace(0, 0.04, 400)
    
    X_list = []
    y_list = []
    
    for class_idx, fault_class in enumerate(FAULT_CLASSES):
        for i in range(n_samples):
            # Inject fault
            netlist = inject_fault(base_netlist, fault_class, seed=i)
            
            # Parse components
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
            sim = _simulate_deterministic(R1, R2, C1, L1, t_eval)
            
            # Extract Features
            re_v0, im_v0 = compute_scatter(sim['v0'], sim['t'])
            features = [re_v0, im_v0]
            
            if include_v1:
                re_v1, im_v1 = compute_scatter(sim['v1'], sim['t'])
                features.extend([re_v1, im_v1])
                
            X_list.append(features)
            y_list.append(class_idx)
            
    return np.array(X_list), np.array(y_list)

def train_and_evaluate(X, y, title):
    print(f"\n{'-'*60}")
    print(f"EXPERIMENT: {title}")
    print(f"{'-'*60}")
    print(f"  Samples: {len(X)} ({len(X)//6}/class)")
    print(f"  Features: {X.shape[1]}")
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train SVM
    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"  Accuracy: {acc:.2%}")
    print("  Confusion Matrix:")
    print(cm)
    return acc

def main():
    print("BENCHMARKING IEEE PAPER EXPERIMENTS")
    
    # Experiment 1: V0 Only (Expect Misclassification)
    # Using L1=1.0H to simulate the "confusion" state mentioned in paper?
    # Actually, with L1=2.0H we fixed the confusion. 
    # To reproduce the paper's "V0 only fails" result, we might need the original parameters 
    # OR maybe V0 is just insufficient even with L1=2.0H? Let's check.
    # The paper says E1 and E4 overlap in V0.
    
    print("\nGenerating data...")
    # We use a large dataset for Exp 1 & 2
    X_v0, y = generate_data(n_samples=50, include_v1=False)
    X_full, _ = generate_data(n_samples=50, include_v1=True)
    
    # Exp 1: V0 Only
    train_and_evaluate(X_v0, y, "V0 Features Only (Table II)")
    
    # Exp 2: V0 + V1
    train_and_evaluate(X_full, y, "V0 + V1 Features (Table III)")
    
    # Exp 3: Small Dataset
    X_small, y_small = generate_data(n_samples=10, include_v1=True)
    train_and_evaluate(X_small, y_small, "Small Dataset (N=10/class) (Table IV)")

if __name__ == "__main__":
    main()
