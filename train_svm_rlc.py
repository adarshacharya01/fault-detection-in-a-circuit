#!/usr/bin/env python3
"""
train_svm_rlc.py

Single-file pipeline:
- Simulate an RLC circuit (synthetic deterministic model)
- Generate training dataset for E0..E5 (N samples/class)
- Extract features (V0, V1)
- Train SVM classifier
- Evaluate metrics + confusion matrix
- Produce distribution scatter (V0 peak-to-peak vs V0 RMS)
- Save representative waveforms and outputs to disk

Outputs (in ./output/):
- model.joblib
- distribution.csv
- distribution.png (scatter)
- waveforms_{class}.png (representative traces)
- results.json (metrics, confusion, classification of supplied healthy netlist)
"""

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from pathlib import Path
import time

# deterministic
RNG = np.random.RandomState(12345)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Simulation parameters (time domain)
FS = 10000.0   # samples per second (high res)
DURATION = 0.02  # 20 ms => one full 50 Hz cycle (period = 20 ms)
N_SAMPLES = int(FS * DURATION)
T = np.linspace(0.0, DURATION, N_SAMPLES, endpoint=False)  # time array in seconds

# nominal circuit values (based on your diagram)
NOMINAL = {
    "R1": 1000.0,    # ohm
    "R2": 1000.0,    # ohm
    "L1": 1e-3,      # H (1 mH)
    "C1": 6e-6,      # F (6 uF)
}

# input
FREQ = 50.0  # Hz
OMEGA = 2 * np.pi * FREQ
VIN_AMP = 10.0  # 10 V amplitude

# classes mapping
CLASSES = ["E0", "E1", "E2", "E3", "E4", "E5"]  # E0 healthy

# how many training samples per class
N_PER_CLASS = 60

# utility: deterministic synthetic "circuit" response
def transfer_gain_v0(R1, R2, L, C, omega=OMEGA):
    """
    Approximate magnitude gain from Vin to Vc (voltage across C1).
    We'll model V0 as a second-order band-limited response:
    H(w) = K / sqrt((1 - (w/w0)^2)^2 + (w/(Q*w0))^2)
    where w0 = 1/sqrt(L*C)
    Choose an effective damping / Q depending on R2 (loads)
    """
    if L <= 0 or C <= 0:
        return 0.0
    w0 = 1.0 / math.sqrt(L * C)
    # rough estimate of equivalent series resistance affecting Q:
    # use R_equiv = R2 (lower R2 -> heavier damping)
    R_eq = max(1.0, R2)
    # define Q proportional to sqrt(L/C)/R_eq
    Q = max(0.01, 0.5 * math.sqrt(L / C) / (R_eq / 1000.0))
    # static gain factor (depends on divider effect of R1)
    divider = 1.0 / (1.0 + R1 / (R2 + 1e-9))
    denom = math.sqrt((1 - (omega / w0) ** 2) ** 2 + (omega / (Q * w0)) ** 2)
    K = divider
    gain = K / (denom + 1e-12)
    return float(gain)

def transfer_gain_v1(R1, R2, L, C, omega=OMEGA):
    """
    Approximate magnitude gain from Vin to Vr2 (voltage across R2).
    Assume V1 is proportional to current through branch times R2.
    We'll approximate current amplitude ~ Vin / (R1 + R2 + reactance_equiv)
    For simplicity, compute a frequency-dependent reactance term from L and C.
    """
    # compute reactances
    Xl = omega * L
    Xc = 1.0 / (omega * C) if C > 0 else 1e12
    # effective reactive magnitude (just combine)
    Xeff = abs(Xl - Xc)
    Zeff = math.sqrt((R2) ** 2 + (Xeff) ** 2)
    denom = R1 + Zeff
    if denom <= 0:
        return 0.0
    # current amplitude approx = Vin / denom, Vr2 amplitude = I * R2
    gain = (R2 / denom)
    # apply minor shaping via resonant denominator too
    # reuse same shape as v0 but smaller gain factor
    gain *= 0.8
    return float(gain)

def simulate_one(R1, R2, L, C, add_transient=True):
    """
    Simulate Vin, V0 (across C1), V1 (across R2) as deterministic time-domain waveforms.
    Uses simple frequency-domain gain applied to sinusoidal input, plus small transient.
    """
    # compute gains
    g0 = transfer_gain_v0(R1, R2, L, C, omega=OMEGA)
    g1 = transfer_gain_v1(R1, R2, L, C, omega=OMEGA)
    vin = VIN_AMP * np.sin(2 * np.pi * FREQ * T)  # perfect sinusoid
    # compute phasing: small phase shift depending on component values
    # approximate phase shift as arctan of (omega/(w0*Q)) scaled
    w0 = 1.0 / math.sqrt(max(1e-12, L * C))
    # small stable phase terms
    phase0 = math.atan2(OMEGA/(w0+1e-9), max(1e-9, R2/1000.0)) * 0.1
    phase1 = -phase0 * 0.6
    v0 = g0 * VIN_AMP * np.sin(2 * np.pi * FREQ * T + phase0)
    v1 = g1 * VIN_AMP * np.sin(2 * np.pi * FREQ * T + phase1)
    # add a deterministic small transient (exponential decay) to mimic start-up
    if add_transient:
        decay = np.exp(-np.linspace(0, 5.0, len(T)))
        v0 = v0 * (1 - 0.05*decay)  # slight approach to steady-state
        v1 = v1 * (1 - 0.05*decay)
    # add tiny deterministic noise from RNG (seeded) but make reproducible by using a fixed pattern
    # We avoid random noise that varies between runs; instead use a reproducible pseudo-random perturbation
    indices = (np.arange(len(T)) * 37) % 101
    tiny_noise = (np.sin(indices) * 0.001)  # micro-volt like perturbations
    v0 = v0 + tiny_noise
    v1 = v1 + tiny_noise * 0.5
    return {"t": T.copy(), "vin": vin.copy(), "v0": v0.copy(), "v1": v1.copy()}

# Fault injection functions: given nominal netlist (R1,R2,L,C) return modified values
def apply_fault(nom, fault_label, variation_pct=0.0):
    R1 = float(nom["R1"]) * (1.0 + variation_pct)
    R2 = float(nom["R2"]) * (1.0 + variation_pct)
    L1 = float(nom["L1"]) * (1.0 + variation_pct)
    C1 = float(nom["C1"]) * (1.0 + variation_pct)
    # fault mapping:
    # E0: healthy (no additional change)
    # E1: R1 open -> very large R1
    # E2: R2 open -> very large R2
    # E3: L1 altered (increase by factor)
    # E4: C1 altered (decrease by factor)
    # E5: short -> R2 dramatically small (simulate short)
    if fault_label == "E1":
        R1 *= 1e3  # very large -> open
    elif fault_label == "E2":
        R2 *= 1e3
    elif fault_label == "E3":
        L1 *= 10.0  # large change
    elif fault_label == "E4":
        C1 *= 0.1  # small capacitance
    elif fault_label == "E5":
        # short: R2 almost zero
        R2 *= 0.001
    # ensure min values positive
    R1 = max(1e-3, R1); R2 = max(1e-3, R2); L1 = max(1e-12, L1); C1 = max(1e-12, C1)
    return R1, R2, L1, C1

# Feature extraction helpers
def peak_to_peak(x):
    return float(np.max(x) - np.min(x))

def rms(x):
    return float(np.sqrt(np.mean(np.array(x)**2)))

def energy(x):
    a = np.array(x)
    return float(np.sum(a*a))

def dominant_freq(x, fs=FS):
    # small FFT peak detection
    y = np.array(x)
    # remove mean
    y = y - np.mean(y)
    N = len(y)
    # take rfft
    yf = np.abs(np.fft.rfft(y))
    xf = np.fft.rfftfreq(N, 1.0/fs)
    idx = int(np.argmax(yf[1:]) + 1) if len(yf) > 1 else 0
    return float(xf[idx]) if idx < len(xf) else 0.0

def extract_features_from_waveforms(sim_out):
    """
    Compute feature vector for a sample.
    We'll compute for V0 primarily, and include a few V1/Vin stats as extra features.
    """
    v0 = np.array(sim_out["v0"])
    v1 = np.array(sim_out["v1"])
    vin = np.array(sim_out["vin"])
    feats = []
    # V0 features
    feats.append(peak_to_peak(v0))
    feats.append(rms(v0))
    feats.append(float(np.mean(v0)))
    feats.append(float(np.std(v0)))
    feats.append(energy(v0))
    feats.append(dominant_freq(v0))
    # V1 features (short)
    feats.append(peak_to_pp(v1) if False else peak_to_peak(v1))
    feats.append(rms(v1))
    # Vin quick stats (to be robust)
    feats.append(np.max(vin))
    feats.append(np.min(vin))
    return np.array(feats, dtype=float)

def compute_scatter_point(sim_out):
    v0 = np.array(sim_out["v0"])
    return [peak_to_peak(v0), rms(v0)]

# Training dataset generation
def generate_dataset(n_per_class=N_PER_CLASS, seed=12345):
    """
    For each class E0..E5 generate n_per_class samples.
    For E0 vary parameters slightly (±5%).
    For faults, apply the fault then add ±3% variation.
    """
    rng = np.random.RandomState(seed)
    X = []
    y = []
    dist = {c: [] for c in CLASSES}
    # Also store representative waveforms (median sample) per class for plotting
    repr_waveforms = {}
    all_waveforms_per_class = {c: [] for c in CLASSES}

    for cls in CLASSES:
        for i in range(n_per_class):
            if cls == "E0":
                var = rng.uniform(-0.05, 0.05)
            else:
                var = rng.uniform(-0.03, 0.03)
            R1, R2, L1, C1 = apply_fault(NOMINAL, cls, variation_pct=var)
            sim = simulate_one(R1, R2, L1, C1, add_transient=True)
            fv = extract_features_from_waveforms(sim)
            X.append(fv)
            y.append(cls)
            sp = compute_scatter_point(sim)
            dist[cls].append(sp)
            all_waveforms_per_class[cls].append(sim)
        # choose representative waveform as the one with median V0_pp
        vpps = [compute_scatter_point(s)[0] for s in all_waveforms_per_class[cls]]
        med_idx = int(np.argsort(vpps)[len(vpps)//2])
        repr_waveforms[cls] = all_waveforms_per_class[cls][med_idx]
    X = np.vstack(X)
    y = np.array(y)
    return X, y, dist, repr_waveforms

# Training and evaluation
def train_and_evaluate(X, y):
    """
    Train SVM with 80/20 split, return model and metrics dict.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES).tolist()
    metrics = {
        "accuracy": float(acc),
        "report": report,
        "confusion": cm
    }
    return clf, metrics

def classify_sample(clf, sim_out):
    fv = extract_features_from_waveforms(sim_out).reshape(1, -1)
    label = clf.predict(fv)[0]
    probs = clf.predict_proba(fv)[0]
    # map class to probability
    label_prob = float(np.max(probs))
    # also create per-class probability dict
    prob_dict = {cls: float(probs[i]) for i, cls in enumerate(clf.classes_)}
    return label, label_prob, prob_dict

# Save utilities
def save_distribution_csv(dist_dict, path):
    rows = []
    for cls, pts in dist_dict.items():
        for (x, y) in pts:
            rows.append({"class": cls, "x": float(x), "y": float(y)})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)

def save_results_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def plot_distribution(dist_dict, outpath):
    plt.figure(figsize=(6,6))
    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]
    for i, cls in enumerate(CLASSES):
        pts = np.array(dist_dict.get(cls, []))
        if pts.shape[0] > 0:
            plt.scatter(pts[:,0], pts[:,1], s=25, alpha=0.8, label=cls, color=colors[i%len(colors)])
    plt.xlabel("V0 peak-to-peak (V)")
    plt.ylabel("V0 RMS (V)")
    plt.title("Training distribution (V0 Vpp vs RMS)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_waveforms_for_class(repr_waveforms, outdir):
    for cls, sim in repr_waveforms.items():
        t = sim["t"] * 1000.0  # ms
        plt.figure(figsize=(8,3))
        plt.plot(t, sim["vin"], label="Vin")
        plt.plot(t, sim["v0"], label="V0 (C1)")
        plt.plot(t, sim["v1"], label="V1 (R2)")
        plt.xlim(0, 20)  # one cycle in ms
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (V)")
        plt.title(f"Representative waveforms — {cls}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = outdir / f"waveforms_{cls}.png"
        plt.savefig(fname, dpi=150)
        plt.close()

def main():
    start = time.time()
    print("Generating training dataset...")
    X, y, dist, repr_waveforms = generate_dataset(n_per_class=N_PER_CLASS, seed=12345)
    print("Dataset generated:", X.shape, "samples.")

    print("Saving distribution CSV...")
    save_distribution_csv(dist, OUTPUT_DIR / "distribution.csv")

    print("Training SVM...")
    clf, metrics = train_and_evaluate(X, y)
    print("Training finished. Accuracy (cv test split):", metrics["accuracy"])

    # save model
    joblib.dump({"clf": clf, "classes": list(clf.classes_)}, OUTPUT_DIR / "model.joblib")
    print("Saved model to", OUTPUT_DIR / "model.joblib")

    # save distribution plot
    plot_distribution(dist, OUTPUT_DIR / "distribution.png")
    print("Saved distribution plot to", OUTPUT_DIR / "distribution.png")

    # save representative waveforms plot
    plot_waveforms_for_class(repr_waveforms, OUTPUT_DIR)
    print("Saved representative waveform plots to", OUTPUT_DIR)

    # Evaluate full training set predictions (for extra diagnostics)
    y_pred_all = clf.predict(X)
    overall_acc = accuracy_score(y, y_pred_all)
    print("Overall training-set accuracy (for info):", overall_acc)

    # print classification report (test-split) in readable form
    rpt = metrics["report"]
    print("\nPer-class metrics (test split):")
    for cls in CLASSES:
        info = rpt.get(cls, {})
        p = info.get("precision", 0.0)
        r = info.get("recall", 0.0)
        f1 = info.get("f1-score", 0.0)
        print(f"{cls}: precision={p:.3f}, recall={r:.3f}, f1={f1:.3f}")

    # Save JSON results for frontend-style consumption
    print("Preparing JSON outputs...")
    def tolist_downsample(arr, maxlen=800):
        a = np.array(arr)
        if len(a) <= maxlen:
            return a.tolist()
        idx = np.linspace(0, len(a)-1, maxlen).astype(int)
        return a[idx].tolist()

    waveforms_out = {
        "t": tolist_downsample(repr_waveforms["E0"]["t"].tolist()),
        "vin": tolist_downsample(repr_waveforms["E0"]["vin"].tolist()),
        "v0_healthy": tolist_downsample(repr_waveforms["E0"]["v0"].tolist()),
        "v1_healthy": tolist_downsample(repr_waveforms["E0"]["v1"].tolist()),
    }
    for cls in CLASSES:
        sim = repr_waveforms[cls]
        waveforms_out[f"v0_{cls}"] = tolist_downsample(sim["v0"].tolist())
        waveforms_out[f"v1_{cls}"] = tolist_downsample(sim["v1"].tolist())

    per_class_simple = {}
    for cls in CLASSES:
        info = rpt.get(cls, {})
        per_class_simple[cls] = {
            "precision": float(info.get("precision", 0.0)),
            "recall": float(info.get("recall", 0.0)),
            "f1": float(info.get("f1-score", 0.0))
        }
    results_json = {
        "metrics": {
            "accuracy": float(metrics["accuracy"]),
            "per_class": {k: per_class_simple[k]["precision"] for k in CLASSES},
            "confusion": metrics["confusion"]
        },
        "distribution": dist,
        "waveforms": waveforms_out
    }

    print("Simulating nominal (healthy) netlist and classifying...")
    sim_nom = simulate_one(NOMINAL["R1"], NOMINAL["R2"], NOMINAL["L1"], NOMINAL["C1"], add_transient=True)
    label, label_prob, prob_dict = classify_sample(clf, sim_nom)
    results_json["classification"] = {"label": label, "confidence": float(label_prob), "probabilities": prob_dict}
    results_json["waveforms"]["t_measured"] = tolist_downsample(sim_nom["t"].tolist())
    results_json["waveforms"]["vin_measured"] = tolist_downsample(sim_nom["vin"].tolist())
    results_json["waveforms"]["v0_measured"] = tolist_downsample(sim_nom["v0"].tolist())
    results_json["waveforms"]["v1_measured"] = tolist_downsample(sim_nom["v1"].tolist())

    save_results_json(results_json, OUTPUT_DIR / "results.json")
    print("Saved results JSON to", OUTPUT_DIR / "results.json")

    end = time.time()
    print(f"Completed in {end-start:.2f} seconds. Outputs in {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
