import time
import numpy as np
from backend.sim import simulate_netlist, _simulate_paper_circuit
from backend.utils import inject_fault

def benchmark():
    netlist = inject_fault("", "E0")
    
    print("Benchmarking simulation...")
    start = time.time()
    for i in range(10):
        simulate_netlist(netlist)
    end = time.time()
    
    avg_time = (end - start) / 10
    print(f"Average time per simulation: {avg_time:.4f}s")
    print(f"Projected time for 120 samples: {avg_time * 120:.2f}s")

if __name__ == "__main__":
    benchmark()
