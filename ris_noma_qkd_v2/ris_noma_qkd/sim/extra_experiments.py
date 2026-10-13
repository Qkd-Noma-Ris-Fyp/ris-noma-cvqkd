"""
extra_experiments.py
=====================
Runs the extra metrics (BER, outage probability, energy efficiency, SIC
decoding success, quantum transmission probability) across the SAME
RIS-element sweep used in sim/main.py, so these new plots sit next to
your original ablation results with a consistent x-axis.

These are ADD-ON metrics (see metrics/extra_metrics.py for exactly which
assumptions each one makes) -- they don't change any of the core
sum-rate / SKR numbers you already validated in sim/main.py.

Run:
    python sim/extra_experiments.py
Outputs (results/):
    ber_vs_ris_elements.png
    outage_probability_vs_ris_elements.png
    energy_efficiency_vs_ris_elements.png
    sic_success_vs_ris_elements.png
    extra_metrics_summary.csv
"""

import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channel.channel_model import RISChannelModel
from ris.ris_optimizer import coordinate_ascent_weighted_sum
from noma.noma_allocation import fractional_power_allocation, sic_sinr_values
from qkd.qkd_model import secret_key_rate
from metrics.extra_metrics import (
    ber_bpsk, sic_decode_success, outage_probability,
    energy_efficiency, quantum_transmission_probability, key_consumption_rate,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_USERS = 3
NUM_BS_ANTENNAS = 4
NOISE_POWER = 1e-9
TX_POWER = 1.0
RATE_OUTAGE_THRESHOLD = 1.0       # bits/s/Hz -- "acceptable minimum" for outage definition
SIC_SINR_THRESHOLD_DB = 0.0       # 0 dB -- minimum SINR to decode & cancel a signal
SIC_SINR_THRESHOLD_LINEAR = 10 ** (SIC_SINR_THRESHOLD_DB / 10.0)


def _one_trial(num_ris_elements, seed):
    model = RISChannelModel(num_bs_antennas=NUM_BS_ANTENNAS, num_ris_elements=num_ris_elements,
                             num_users=NUM_USERS, seed=seed)
    beamformer = np.ones(NUM_BS_ANTENNAS) / np.sqrt(NUM_BS_ANTENNAS)
    theta, _ = coordinate_ascent_weighted_sum(
        model.H1, model.H2, model.Hd, weights=None, beamformer=beamformer,
        n_outer=12, grid_size=120, seed=seed,
    )
    gains = np.array([np.abs(model.effective_channel(k, theta, beamformer)) ** 2 for k in range(NUM_USERS)])

    powers, order = fractional_power_allocation(gains, TX_POWER, alpha=0.6)
    sinrs = sic_sinr_values(gains, powers, order, NOISE_POWER)
    rates = np.log2(1 + sinrs)

    bers = ber_bpsk(sinrs)
    sic_success = sic_decode_success(sinrs, SIC_SINR_THRESHOLD_LINEAR)
    ee = energy_efficiency(rates.sum(), TX_POWER)
    # Report the AVERAGE transmission probability across all K users,
    # each normalized against this trial's strongest user (gains.max()).
    # Reporting only the strongest user's own T would trivially always
    # read 1.0 (T = gains.max()/gains.max()) and hide how much weaker
    # the other users' links are -- the mean across users is what
    # actually varies and is worth plotting.
    T_per_user = [quantum_transmission_probability(g, reference_gain=gains.max()) for g in gains]
    T = float(np.mean(T_per_user))
    skr, *_ = secret_key_rate(gains.max(), reference_gain=gains.max())

    return {
        "sum_rate": rates.sum(),
        "mean_ber": bers.mean(),
        "sic_success_rate": sic_success.mean(),
        "energy_efficiency": ee,
        "quantum_transmission_probability": T,
        "skr": skr,
    }


def run_sweep(num_trials=40):
    element_counts = [4, 8, 16, 32, 64, 128, 256]
    rows = []
    summary = {n: {"sum_rate": [], "mean_ber": [], "sic_success_rate": [],
                    "energy_efficiency": [], "quantum_transmission_probability": [], "skr": []}
               for n in element_counts}

    print("=== Extra metrics sweep vs. RIS elements ===")
    for n in element_counts:
        for t in range(num_trials):
            result = _one_trial(n, seed=30_000 + t)
            for key in summary[n]:
                summary[n][key].append(result[key])
            rows.append({"ris_elements": n, "trial": t, **result})
        means = {k: np.mean(v) for k, v in summary[n].items()}
        print(f"  N={n:4d}  sum_rate={means['sum_rate']:.2f}  BER={means['mean_ber']:.2e}  "
              f"SIC_success={means['sic_success_rate']:.2f}  EE={means['energy_efficiency']:.2f}  "
              f"T={means['quantum_transmission_probability']:.3f}  SKR={means['skr']:.4f}")

    # outage probability needs the full distribution of sum_rate per N, not just the mean
    outage_by_n = {
        n: outage_probability(np.array(summary[n]["sum_rate"]), RATE_OUTAGE_THRESHOLD)
        for n in element_counts
    }

    return element_counts, summary, outage_by_n, rows


def plot_results(element_counts, summary, outage_by_n):
    mean_ber = [np.mean(summary[n]["mean_ber"]) for n in element_counts]
    mean_sic = [np.mean(summary[n]["sic_success_rate"]) for n in element_counts]
    mean_ee = [np.mean(summary[n]["energy_efficiency"]) for n in element_counts]
    outage = [outage_by_n[n] for n in element_counts]

    # --- BER plot ---
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.semilogy(element_counts, mean_ber, "o-", color="#D1495B", linewidth=2.2, markersize=5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of RIS elements (N)")
    ax.set_ylabel("Mean BER (uncoded BPSK, log scale)")
    ax.set_title("Bit Error Rate vs. RIS Size\n(illustrative BPSK assumption -- see metrics/extra_metrics.py)")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ber_vs_ris_elements.png"), dpi=150)
    plt.close()

    # --- Outage probability plot ---
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(element_counts, outage, "o-", color="#1C7293", linewidth=2.2, markersize=5)
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Number of RIS elements (N)")
    ax.set_ylabel(f"Outage probability (rate < {RATE_OUTAGE_THRESHOLD} bits/s/Hz)")
    ax.set_title("Outage Probability vs. RIS Size\n(40 Monte Carlo trials per point)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "outage_probability_vs_ris_elements.png"), dpi=150)
    plt.close()

    # --- Energy efficiency plot ---
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(element_counts, mean_ee, "o-", color="#2C5F2D", linewidth=2.2, markersize=5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of RIS elements (N)")
    ax.set_ylabel("Energy efficiency (bits/s/Hz per unit tx power)")
    ax.set_title("Energy Efficiency vs. RIS Size")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "energy_efficiency_vs_ris_elements.png"), dpi=150)
    plt.close()

    # --- SIC decode success plot ---
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(element_counts, mean_sic, "o-", color="#C98A2B", linewidth=2.2, markersize=5)
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Number of RIS elements (N)")
    ax.set_ylabel(f"Mean SIC decode success rate (threshold={SIC_SINR_THRESHOLD_DB} dB)")
    ax.set_title("SIC Decoding Success Rate vs. RIS Size")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sic_success_vs_ris_elements.png"), dpi=150)
    plt.close()

    print(f"\nSaved 4 plots to {RESULTS_DIR}")


def save_csv(rows):
    path = os.path.join(RESULTS_DIR, "extra_metrics_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


if __name__ == "__main__":
    element_counts, summary, outage_by_n, rows = run_sweep(num_trials=40)
    plot_results(element_counts, summary, outage_by_n)
    save_csv(rows)

    # one illustrative key-consumption-rate example, printed (not swept --
    # it's an application-layer "what if" calculation, see docstring)
    example_skr = np.mean(summary[256]["skr"])
    ratio = key_consumption_rate(example_skr, application_data_rate_bits_per_s=0.05, symbol_rate_hz=1.0)
    print(f"\nExample key consumption ratio @ N=256 (illustrative OTP assumption, "
          f"app needs 0.05 bits/s): {ratio:.3f}  "
          f"({'keys build up faster than used' if ratio < 1 else 'application would be key-starved'})")
