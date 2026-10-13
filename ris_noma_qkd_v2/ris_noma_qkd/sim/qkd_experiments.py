"""
qkd_experiments.py
==================
Two QKD experiments that replace the (previously constant) SKR claim:

 (D) SKR vs transmittance T -- validates the QKD module on its own.
     This is the standard CV-QKD figure; it does not depend on any
     RF-to-optical mapping choice.
 (E) SKR vs RIS size, using a FIXED calibration gain G_REF (T = |g|^2 / G_REF).
     G_REF is a placeholder for a real link budget, so the curve is shown for
     several G_REF values (sensitivity), not one hand-picked value.

Run:  python sim/qkd_experiments.py
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from channel.channel_model import RISChannelModel
from qkd.qkd_model import secret_key_rate
from main import _gains_for_config, NUM_USERS, NUM_BS_ANTENNAS, RESULTS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)


def experiment_skr_vs_transmittance():
    T_vals = np.logspace(-3, 0, 60)
    skr, iab, chi = [], [], []
    for T in T_vals:
        s, i, c, _ = secret_key_rate(T, reference_gain=1.0)   # gain == T when ref = 1
        skr.append(s); iab.append(i); chi.append(c)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.semilogx(T_vals, iab, label=r"$I(A{:}B)$", color="#1C7293", lw=2)
    ax.semilogx(T_vals, chi, label=r"$\chi(B{:}E)$ (Eve, Holevo)", color="#7C8798", lw=2)
    ax.semilogx(T_vals, skr, label="Secret key rate", color="#D1495B", lw=2.5)
    ax.set_xlabel("Channel transmittance T"); ax.set_ylabel("bits / symbol")
    ax.set_title("CV-QKD module: key rate vs transmittance\n(xi=0.01, V_mod=4, beta=0.95, eta=0.6)")
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "skr_vs_transmittance.png")
    plt.savefig(path, dpi=150); plt.close()
    print("Saved:", path)


def experiment_skr_vs_ris(num_trials=40, g_refs=(1e-7, 3e-7, 1e-6)):
    Ns = [4, 8, 16, 32, 64, 128, 256]
    bf = np.ones(NUM_BS_ANTENNAS) / np.sqrt(NUM_BS_ANTENNAS)
    colors = {"none": "#7C8798", "optimized": "#D1495B"}
    labels = {"none": "No RIS", "optimized": "RIS (optimized)"}
    fig, axes = plt.subplots(1, len(g_refs), figsize=(5 * len(g_refs), 4.4), sharey=True)
    for ax, g_ref in zip(np.atleast_1d(axes), g_refs):
        for mode in ["none", "optimized"]:
            means = []
            for N in Ns:
                vals = []
                for t in range(num_trials):
                    model = RISChannelModel(NUM_BS_ANTENNAS, N, NUM_USERS, seed=10_000 + t)
                    g = _gains_for_config(model, bf, mode, 10_000 + t).max()
                    vals.append(secret_key_rate(g, reference_gain=g_ref)[0])
                means.append(np.mean(vals))
            ax.plot(Ns, means, "o-", color=colors[mode], label=labels[mode], lw=2)
            print(f"G_REF={g_ref:g} {labels[mode]:15s}", np.round(means, 3))
        ax.set_xscale("log", base=2); ax.grid(alpha=0.25)
        ax.set_title(f"G_REF = {g_ref:g}"); ax.set_xlabel("RIS elements N")
    np.atleast_1d(axes)[0].set_ylabel("Mean SKR (bits/symbol)")
    np.atleast_1d(axes)[0].legend()
    fig.suptitle("SKR vs RIS size under three calibration gains (T = |g|^2 / G_REF)")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "skr_vs_ris_elements.png")
    plt.savefig(path, dpi=150); plt.close()
    print("Saved:", path)


if __name__ == "__main__":
    experiment_skr_vs_transmittance()
    experiment_skr_vs_ris()
