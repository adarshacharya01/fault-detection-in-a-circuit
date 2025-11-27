
import numpy as np
from backend.sim import simulate_netlist, _simulate_deterministic
from backend.features import extract_features

def test_separation(L1_val):
    print(f"\nTesting L1 = {L1_val} H")
    
    # Base parameters
    R1 = 1000.0
    R2 = 1000.0
    C1 = 6e-6
    
    # Fault multipliers
    faults = {
        'E0': (1.0, 1.0, 1.0), # R1, R2, C1 multipliers
        'E2': (0.5, 1.0, 1.0), # R1 Low
        'E4': (1.0, 0.5, 1.0), # R2 Low
        'E5': (1.0, 1.0, 0.5), # C1 Low
    }
    
    results = {}
    t_eval = np.linspace(0, 0.04, 400)
    
    for cls, (mR1, mR2, mC1) in faults.items():
        try:
            sim = _simulate_deterministic(
                R1 * mR1, 
                R2 * mR2, 
                C1 * mC1, 
                L1_val, 
                t_eval
            )
            v0 = sim['v0']
            pk_pk = float(np.max(v0) - np.min(v0))
            rms = float(np.sqrt(np.mean(np.array(v0)**2)))
            results[cls] = (pk_pk, rms)
        except Exception as e:
            print(f"  {cls}: Failed {e}")
            
    # Print results
    print(f"  {'Class':<5} {'Pk-Pk':<10} {'RMS':<10}")
    for cls, (pk, rms) in results.items():
        print(f"  {cls:<5} {pk:<10.3f} {rms:<10.3f}")
        
    # Check separation
    # We want E2, E4, E5 to be distinct from each other and E0.
    # Currently E2, E4, E5 are all High.
    
    vals = list(results.values())
    # Simple metric: min distance between any pair
    min_dist = float('inf')
    import itertools
    for p1, p2 in itertools.combinations(vals, 2):
        dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        min_dist = min(min_dist, dist)
        
    print(f"  Min Separation: {min_dist:.3f}")
    return min_dist

if __name__ == "__main__":
    # Sweep L1 from 1mH to 2H
    best_L1 = -1
    best_sep = -1
    
    for L1 in [0.001, 0.01, 0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        sep = test_separation(L1)
        if sep > best_sep:
            best_sep = sep
            best_L1 = L1
            
    print(f"\nBest L1: {best_L1} with separation {best_sep:.3f}")
