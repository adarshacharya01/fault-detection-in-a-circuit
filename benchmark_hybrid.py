import time
import numpy as np
from backend.demo_data import generate_training_dataset
from backend.sim import _simulate_deterministic

def benchmark():
    print("Benchmarking single fast simulation...")
    t_eval = np.linspace(0, 0.04, 400)
    start = time.time()
    for i in range(100):
        _simulate_deterministic(1000, 1000, 6e-6, 1e-3, t_eval)
    end = time.time()
    print(f"100 simulations took: {end - start:.4f}s")
    
    print("\nBenchmarking full dataset generation (N=60)...")
    base_netlist = "V1 in 0 SIN(0 10 50)\nR1 in 1 1k\nR2 1 2 1k\nL1 1 0 1mH\nC1 2 0 6u\n.tran 20m 0 0.01m\n.END"
    start = time.time()
    generate_training_dataset(base_netlist, N=60)
    end = time.time()
    print(f"Dataset generation took: {end - start:.4f}s")

if __name__ == "__main__":
    benchmark()
