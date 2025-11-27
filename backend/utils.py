"""Utility functions for fault injection matching IEEE Paper Table I."""
import numpy as np
from typing import Dict

def inject_fault(netlist: str, fault_class: str, seed: int = 0) -> str:
    """
    Inject fault into netlist based on IEEE Paper Table I.
    
    Classes:
    - E0: Normal (Tolerance ±5%)
    - E1: R1 High (150% - 250% more -> 2.5x - 3.5x)
    - E2: R1 Low (50% less -> 0.5x)
    - E3: R2 High (150% - 250% more -> 2.5x - 3.5x)
    - E4: R2 Low (50% less -> 0.5x)
    - E5: C1 Low (50% less -> 0.5x)
    """
    rng = np.random.RandomState(seed)
    
    # Base values
    R1 = 1000.0
    R2 = 1000.0
    L1 = 2.0
    C1 = 6e-6
    
    # Apply tolerance to all components first (±5%)
    R1 *= rng.uniform(0.95, 1.05)
    R2 *= rng.uniform(0.95, 1.05)
    L1 *= rng.uniform(0.95, 1.05)
    C1 *= rng.uniform(0.95, 1.05)
    
    # Apply Faults
    if fault_class == 'E0':
        pass # Just tolerance
        
    elif fault_class == 'E1':
        # R1 150-250% MORE. i.e., +1.5 to +2.5. Total 2.5x to 3.5x.
        # Paper says "R1 is 150% - 250% more than correct value"
        factor = rng.uniform(2.5, 3.5)
        R1 *= factor
        
    elif fault_class == 'E2':
        # R1 50% less
        R1 *= 0.5
        
    elif fault_class == 'E3':
        # R2 150-250% more
        factor = rng.uniform(2.5, 3.5)
        R2 *= factor
        
    elif fault_class == 'E4':
        # R2 50% less
        R2 *= 0.5
        
    elif fault_class == 'E5':
        # C 50% less
        C1 *= 0.5
        
    # Reconstruct netlist
    return f"""V1 in 0 SIN(0 10 50)
R1 in 1 {R1}
R2 1 2 {R2}
L1 1 0 {L1}
C1 2 0 {C1}
.tran 20m 0 0.01m
.END"""

# Note: L1 is connected to Node 1 (between R1 and R2) and Ground.
# R2 connects Node 1 to Node 2.
# C1 connects Node 2 to Ground.
# This matches the circuit derivation in sim.py.
