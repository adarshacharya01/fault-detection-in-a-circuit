import time
import numpy as np
from backend.sim import simulate_netlist, _simulate_deterministic

def benchmark():
    netlist = "V1 in 0 SIN(0 10 50)\nR1 in 1 1k\nR2 1 2 1k\nL1 1 0 1mH\nC1 2 0 6u\n.tran 20m 0 0.01m\n.END"
    
    print("Benchmarking ODE simulation (simulate_netlist)...")
    start = time.time()
    for i in range(5):
        simulate_netlist(netlist)
    end = time.time()
    print(f"ODE Average (5 runs): {(end - start)/5:.4f}s")
    
    print("\nBenchmarking Transfer Function (_simulate_deterministic)...")
    t_eval = np.linspace(0, 0.02, 200)
    start = time.time()
    for i in range(100):
        _simulate_deterministic(1000, 1000, 6e-6, 1e-3, t_eval)
    end = time.time()
    print(f"Transfer Function Average (100 runs): {(end - start)/100:.4f}s")

if __name__ == "__main__":
    benchmark()
