"""Feature extraction matching IEEE paper methodology."""
import numpy as np
from typing import Dict, List, Tuple

def peak_to_peak(v: List[float]) -> float:
    v_array = np.array(v)
    return float(np.max(v_array) - np.min(v_array))

def rms(v: List[float]) -> float:
    v_array = np.array(v)
    return float(np.sqrt(np.mean(v_array**2)))

def mean_value(v: List[float]) -> float:
    return float(np.mean(v))

def std_dev(v: List[float]) -> float:
    return float(np.std(v))

def energy(v: List[float]) -> float:
    v_array = np.array(v)
    return float(np.sum(v_array**2))

def dominant_frequency(v: List[float], dt: float) -> float:
    v_array = np.array(v)
    fft_vals = np.fft.fft(v_array)
    fft_mag = np.abs(fft_vals[:len(fft_vals)//2])
    freqs = np.fft.fftfreq(len(v_array), dt)
    freqs_positive = freqs[:len(freqs)//2]
    
    if len(fft_mag) > 0:
        dom_idx = np.argmax(fft_mag)
        return float(abs(freqs_positive[dom_idx]))
    return 0.0

def extract_features(sim_output: Dict[str, List[float]]) -> np.ndarray:
    """
    Extract Phasor features from V0 and V1.
    Features: [Real(V0), Imag(V0), Real(V1), Imag(V1)]
    """
    t = sim_output['t']
    v0 = sim_output['v0']
    v1 = sim_output['v1']
    
    # Calculate Phasors
    # compute_scatter now returns (Real, Imag)
    re_v0, im_v0 = compute_scatter(v0, t)
    re_v1, im_v1 = compute_scatter(v1, t)
    
    return np.array([re_v0, im_v0, re_v1, im_v1])

def compute_scatter(v0: List[float], t: List[float]) -> Tuple[float, float]:
    """
    Compute Phasor components (Real, Imag) of V0.
    Reference is sin(omega*t) (Input voltage phase).
    
    v0(t) = Real * sin(wt) + Imag * cos(wt)
    """
    v_arr = np.array(v0)
    t_arr = np.array(t)
    omega = 314.15 # 50Hz
    
    # Reference signals
    ref_real = np.sin(omega * t_arr) # In-phase with input
    ref_imag = np.cos(omega * t_arr) # Quadrature (90 deg lead)
    
    # Correlation (Projection)
    # Factor of 2 because avg(sin^2) = 0.5
    real_part = 2.0 * np.mean(v_arr * ref_real)
    imag_part = 2.0 * np.mean(v_arr * ref_imag)
    
    return (float(real_part), float(imag_part))
