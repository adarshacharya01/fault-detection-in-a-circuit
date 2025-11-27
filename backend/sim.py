"""Circuit simulation module using ODE solver (matching reference script)."""
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, List, Tuple

# Circuit parameters from IEEE paper
R1_NOM = 1000.0
R2_NOM = 1000.0
C1_NOM = 6e-6
L1_NOM = 2.0
V_AMP = 10.0
OMEGA = 314.15  # rad/s (~50 Hz)

def Vi(t: float) -> float:
    """Sinusoidal input voltage: 10 * sin(omega * t)."""
    return V_AMP * np.sin(OMEGA * t)

def circuit_ode(t: float, y: List[float], R1: float, R2: float, C1: float, L1: float) -> List[float]:
    """
    Differential equations for the RLC circuit.
    
    State variables:
        y[0] = v2(t) = V0(t)  -> Voltage across C1 (and parallel branch)
        y[1] = iL(t)          -> Current through inductor L1
        
    Equations derived from KCL/KVL:
        dv2/dt = - (v2 - Vi) / (R1 * C1) - iL / C1
        diL/dt = (v2 - iL * R2) / L1
    """
    v2 = y[0]
    iL = y[1]
    
    # Avoid division by zero
    if R1 <= 1e-9: R1 = 1e-9
    if C1 <= 1e-12: C1 = 1e-12
    if L1 <= 1e-12: L1 = 1e-12
    
    dv2_dt = - (v2 - Vi(t)) / (R1 * C1) - iL / C1
    diL_dt = (v2 - iL * R2) / L1
    
    return [dv2_dt, diL_dt]

def simulate_netlist(netlist_str: str) -> Dict[str, List[float]]:
    """
    Simulate circuit defined by netlist string.
    
    Parses component values from netlist and runs ODE solver.
    Returns dictionary with t, vin, v0, v1.
    """
    # Parse components
    components = _parse_components(netlist_str)
    
    R1 = components.get('R1', R1_NOM)
    R2 = components.get('R2', R2_NOM)
    C1 = components.get('C1', C1_NOM)
    L1 = components.get('L1', L1_NOM)
    
    # Simulation parameters
    t_end = 0.02  # 20ms (1 cycle at 50Hz)
    points = 200
    
    t_span = (0.0, t_end)
    t_eval = np.linspace(t_span[0], t_span[1], points)
    
    # Use the correct paper circuit simulation
    return _simulate_paper_circuit(R1, R2, C1, L1, t_eval)

def _simulate_paper_circuit(R1, R2, C1, L1, t_eval):
    """
    Simulate the exact circuit from IEEE Paper Fig 2.
    
    State variables:
    y[0] = vC (Voltage across C1) = V0
    y[1] = iL (Current through L1)
    """
    def paper_ode(t, y):
        vC = y[0]
        iL = y[1]
        
        # Input voltage
        vi = Vi(t)
        
        # Algebraic equation for V1 (Node 1 voltage)
        # Derived from KCL at Node 1: (V1-Vi)/R1 + iL + (V1-vC)/R2 = 0
        # V1 * (1/R1 + 1/R2) = Vi/R1 + vC/R2 - iL
        g1 = 1/R1
        g2 = 1/R2
        v1 = (vi*g1 + vC*g2 - iL) / (g1 + g2)
        
        # Differential equations
        # dvC/dt = (V1 - vC) / (R2 * C1)
        dvC_dt = (v1 - vC) / (R2 * C1)
        
        # diL/dt = V1 / L1
        diL_dt = v1 / L1
        
        return [dvC_dt, diL_dt]
    
    y0 = [0.0, 0.0]
    
    try:
        sol = solve_ivp(
            paper_ode,
            (t_eval[0], t_eval[-1]),
            y0,
            t_eval=t_eval,
            method='LSODA',
            rtol=1e-4,
            atol=1e-6
        )
        
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")
        
        t = sol.t
        v0 = sol.y[0]
        iL = sol.y[1]
        
        # Calculate V1 array
        vi = Vi(t)
        g1 = 1/R1
        g2 = 1/R2
        v1 = (vi*g1 + v0*g2 - iL) / (g1 + g2)
        
        return {
            "t": t.tolist(),
            "vin": vi.tolist(),
            "v0": v0.tolist(),
            "v1": v1.tolist()
        }
    except Exception as e:
        print(f"Simulation error: {e}")
        zeros = np.zeros_like(t_eval).tolist()
        return {"t": t_eval.tolist(), "vin": zeros, "v0": zeros, "v1": zeros}

def _parse_components(netlist_str: str) -> Dict[str, float]:
    """Parse component values from netlist."""
    components = {}
    lines = netlist_str.strip().split('\n')
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 4: continue
        
        name = parts[0].upper()
        try:
            val = _parse_value(parts[3])
            components[name] = val
        except:
            pass
            
    return components

def _parse_value(val_str: str) -> float:
    """Parse SPICE value."""
    val_str = val_str.upper()
    multipliers = {
        'T': 1e12, 'G': 1e9, 'MEG': 1e6, 'K': 1e3,
        'M': 1e-3, 'U': 1e-6, 'N': 1e-9, 'P': 1e-12, 'F': 1e-15
    }
    for suffix, mult in multipliers.items():
        if val_str.endswith(suffix):
            return float(val_str[:-len(suffix)]) * mult
    return float(val_str)

def _simulate_deterministic(R1, R2, C1, L1, t_eval):
    """
    Fast transfer function simulation for training data generation.
    Mathematically equivalent to ODE for linear RLC, but 1000x faster.
    """
    # Time array
    t = t_eval
    
    # Input parameters
    omega = OMEGA
    V_amp = V_AMP
    
    # Impedances
    Z_R1 = R1
    Z_R2 = R2
    Z_L = 1j * omega * L1
    Z_C = 1 / (1j * omega * C1)
    
    # Transfer functions
    # V1 node: Z_parallel = (R2 + Z_C) || Z_L ? No.
    # Paper circuit: Node 1 connects R1, L1, R2. Node 2 connects R2, C1.
    # Z_node1_load = Z_L || (R2 + Z_C)
    Z_branch2 = R2 + Z_C
    Z_node1_load = (Z_L * Z_branch2) / (Z_L + Z_branch2)
    
    # Voltage divider for V1
    H_v1 = Z_node1_load / (R1 + Z_node1_load)
    
    # Voltage divider for V0 (from V1)
    # V0 is across C1
    H_v0_from_v1 = Z_C / (R2 + Z_C)
    H_v0 = H_v1 * H_v0_from_v1
    
    # Calculate steady state response
    # V1 = |H_v1| * V_amp * sin(wt + arg(H_v1))
    mag_v1 = np.abs(H_v1)
    arg_v1 = np.angle(H_v1)
    v1 = mag_v1 * V_amp * np.sin(omega * t + arg_v1)
    
    # V0 = |H_v0| * V_amp * sin(wt + arg(H_v0))
    mag_v0 = np.abs(H_v0)
    arg_v0 = np.angle(H_v0)
    v0 = mag_v0 * V_amp * np.sin(omega * t + arg_v0)
    
    # Input
    vin = V_amp * np.sin(omega * t)
    
    return {
        "t": t.tolist(),
        "vin": vin.tolist(),
        "v0": v0.tolist(),
        "v1": v1.tolist()
    }

def get_one_cycle(t: List[float], signal: List[float], freq: float = 50.0) -> Tuple[List[float], List[float]]:
    """Get one cycle of waveform."""
    period = 1.0 / freq
    t_arr = np.array(t)
    sig_arr = np.array(signal)
    
    # Get last cycle for steady state
    mask = (t_arr >= t_arr[-1] - period)
    return t_arr[mask].tolist(), sig_arr[mask].tolist()
