import numpy as np

def calculate_phasor(R1, R2, C1, L1):
    omega = 314.15
    V_amp = 10.0
    
    Z_R1 = R1
    Z_R2 = R2
    Z_L = 1j * omega * L1
    Z_C = 1 / (1j * omega * C1)
    
    # Z_branch2 = R2 + Z_C
    Z_branch2 = R2 + Z_C
    
    # Z_node1_load = Z_L || Z_branch2
    Z_node1_load = (Z_L * Z_branch2) / (Z_L + Z_branch2)
    
    # Voltage divider for V1
    H_v1 = Z_node1_load / (R1 + Z_node1_load)
    
    # Voltage divider for V0 (from V1)
    H_v0_from_v1 = Z_C / (R2 + Z_C)
    H_v0 = H_v1 * H_v0_from_v1
    
    V0_phasor = H_v0 * V_amp
    return V0_phasor

def tune():
    print("Target: 2.5 - j2.5")
    print("Testing L1 values...")
    
    best_L1 = 0
    min_dist = float('inf')
    
    for L1 in np.linspace(0.001, 5.0, 1000):
        V0 = calculate_phasor(1000, 1000, 6e-6, L1)
        dist = np.abs(V0 - (2.5 - 2.5j))
        if dist < min_dist:
            min_dist = dist
            best_L1 = L1
            
    print(f"Best L1: {best_L1:.4f} H")
    V0 = calculate_phasor(1000, 1000, 6e-6, best_L1)
    print(f"Result V0: {V0.real:.2f} + j{V0.imag:.2f}")
    print(f"Magnitude: {np.abs(V0):.2f}")

if __name__ == "__main__":
    tune()
