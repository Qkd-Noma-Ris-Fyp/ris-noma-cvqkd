"""
main.py
=======
Validation-grade simulation driver for the RIS + NOMA + QKD system.

Three experiments, each built to answer the question a review panel asks
first -- "how do you know this works better than the obvious alternative?":

    (A) Ablation study: No-RIS vs RIS(heuristic) vs RIS(optimized) x
        OMA vs NOMA, sweeping RIS size -- isolates how much of the gain
        comes from RIS and how much from NOMA.
    (B) Same ablation, swept over transmit power instead of RIS size.
    (C) Convergence proof for the coordinate-ascent RIS optimizer --
        shows the optimization actually converges, monotonically, which
        is what makes "optimized" a defensible claim rather than a label.

Every curve is a mean over `num_trials` independent channel realizations,
plotted with a shaded +/-1 std-dev band -- not a single lucky run.

Run:
    python sim/main.py
Outputs (results/):
    ablation_vs_ris_elements.png
    ablation_vs_tx_power.png
    ris_optimizer_convergence.png
    summary.csv
"""

import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channel.channel_model import RISChannelModel
from ris.ris_optimizer import phase_alignment_heuristic, coordinate_ascent_weighted_sum
from noma.noma_allocation import sum_rate, oma_sum_rate

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_USERS = 3
NUM_BS_ANTENNAS = 4
NOISE_POWER = 1e-9


def _gains_for_config(model, beamformer, ris_mode, seed):
    """Return (K,) effective |g_k|^2 gains for one of four RIS configurations."""
    K = model.K
    if ris_mode == "none":
        # TRUE no-RIS baseline: direct link only, RIS entirely absent from
        # the link budget. (theta=ones would still reflect through all N
        # elements with zero phase shift, which is a *different* system --
        # a passive, non-optimized RIS -- and would spuriously gain array
        # gain as N grows, undermining the "no RIS at all" comparison.)
        return np.array([np.abs(model.Hd[k] @ beamformer) ** 2 for k in range(K)])
    elif ris_mode == "random":
        rng = np.random.default_rng(seed + 777)
        theta = np.exp(1j * rng.uniform(-np.pi, np.pi, model.Nr))
    elif ris_mode == "heuristic":
        # align to the weakest direct-link user as a simple, common choice
        direct_strengths = [np.abs(model.Hd[k]).sum() for k in range(K)]
        weakest = int(np.argmin(direct_strengths))
        theta = phase_alignment_heuristic(model.H1, model.H2[weakest], model.Hd[weakest], beamformer)
    elif ris_mode == "optimized":
        theta, _ = coordinate_ascent_weighted_sum(
            model.H1, model.H2, model.Hd, weights=None, beamformer=beamformer,
            n_outer=12, grid_size=120, seed=seed,
        )
    else:
        raise ValueError(ris_mode)

    return np.array([np.abs(model.effective_channel(k, theta, beamformer)) ** 2 for k in range(K)])


def _trial(num_ris_elements, tx_power, ris_mode, access_mode, seed):
    model = RISChannelModel(
        num_bs_antennas=NUM_BS_ANTENNAS, num_ris_elements=max(num_ris_elements, 1),
        num_users=NUM_USERS, seed=seed,
    )
    beamformer = np.ones(NUM_BS_ANTENNAS) / np.sqrt(NUM_BS_ANTENNAS)
    gains = _gains_for_config(model, beamformer, ris_mode, seed)

    if access_mode == "noma":
        rate, *_ = sum_rate(gains, tx_power, NOISE_POWER, alpha=0.6)
    else:
        rate, _ = oma_sum_rate(gains, tx_power, NOISE_POWER)

    # NOTE: SKR is NOT computed here. An earlier version normalised the gain by
    # its own maximum, forcing T=1 and a constant SKR. SKR experiments live in
    # sim/qkd_experiments.py and use a FIXED calibration gain instead.
    return rate


def _sweep(x_values, x_setter, num_trials, configs):
    """configs: list of (label, ris_mode, access_mode). Returns dict label -> (mean, std) arrays."""
    out = {label: {"mean": [], "std": []} for label, _, _ in configs}
    for x in x_values:
        n_ris, p_tx = x_setter(x)
        for label, ris_mode, access_mode in configs:
            trial_rates = [
                _trial(n_ris, p_tx, ris_mode, access_mode, seed=10_000 + t)
                for t in range(num_trials)
            ]
            out[label]["mean"].append(np.mean(trial_rates))
            out[label]["std"].append(np.std(trial_rates))
        print(f"  x={x}  " + "  ".join(f"{l}={out[l]['mean'][-1]:.2f}" for l, _, _ in configs))
    for label in out:
        out[label]["mean"] = np.array(out[label]["mean"])
        out[label]["std"] = np.array(out[label]["std"])
    return out


CONFIGS = [
    ("No-RIS + OMA",            "none",      "oma"),
    ("No-RIS + NOMA",           "none",      "noma"),
    ("RIS (heuristic) + NOMA",  "heuristic", "noma"),
    ("RIS (optimized) + NOMA",  "optimized", "noma"),
]
COLORS = {
    "No-RIS + OMA": "#B9C0D4",
    "No-RIS + NOMA": "#7C8798",
    "RIS (heuristic) + NOMA": "#1C7293",
    "RIS (optimized) + NOMA": "#D1495B",
}


def experiment_ablation_vs_ris_elements(num_trials=40):
    print("=== Ablation vs. RIS elements (NOISE_POWER fixed, tx_power=1.0) ===")
    element_counts = [4, 8, 16, 32, 64, 128, 256]
    results = _sweep(
        element_counts,
        x_setter=lambda n: (n, 1.0),
        num_trials=num_trials,
        configs=CONFIGS,
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for label, _, _ in CONFIGS:
        m, s = results[label]["mean"], results[label]["std"]
        ax.plot(element_counts, m, "o-", color=COLORS[label], label=label, linewidth=2.2, markersize=5)
        ax.fill_between(element_counts, m - s, m + s, color=COLORS[label], alpha=0.15)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of RIS elements (N)")
    ax.set_ylabel("Sum-rate (bits/s/Hz)")
    ax.set_title(f"Ablation: RIS & NOMA Contributions vs. RIS Size\n(mean ± 1 std over {num_trials} channel realizations)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "ablation_vs_ris_elements.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    return element_counts, results


def experiment_ablation_vs_power(num_trials=40):
    print("\n=== Ablation vs. transmit power (RIS elements fixed at N=32) ===")
    tx_powers_db = np.arange(-10, 21, 5)
    results = _sweep(
        tx_powers_db,
        x_setter=lambda p: (32, 10 ** (p / 10.0)),
        num_trials=num_trials,
        configs=CONFIGS,
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for label, _, _ in CONFIGS:
        m, s = results[label]["mean"], results[label]["std"]
        ax.plot(tx_powers_db, m, "o-", color=COLORS[label], label=label, linewidth=2.2, markersize=5)
        ax.fill_between(tx_powers_db, m - s, m + s, color=COLORS[label], alpha=0.15)
    ax.set_xlabel("Relative transmit power (dB)")
    ax.set_ylabel("Sum-rate (bits/s/Hz)")
    ax.set_title(f"Ablation: RIS & NOMA Contributions vs. Transmit Power (N=32)\n(mean ± 1 std over {num_trials} channel realizations)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "ablation_vs_tx_power.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    return tx_powers_db, results


def experiment_convergence(num_runs=15, num_ris_elements=32):
    print(f"\n=== Coordinate-ascent convergence proof (N={num_ris_elements}, {num_runs} random channel seeds) ===")
    beamformer = np.ones(NUM_BS_ANTENNAS) / np.sqrt(NUM_BS_ANTENNAS)
    all_histories = []
    for t in range(num_runs):
        model = RISChannelModel(num_bs_antennas=NUM_BS_ANTENNAS, num_ris_elements=num_ris_elements,
                                 num_users=NUM_USERS, seed=20_000 + t)
        _, history = coordinate_ascent_weighted_sum(
            model.H1, model.H2, model.Hd, weights=None, beamformer=beamformer,
            n_outer=12, grid_size=120, seed=20_000 + t,
        )
        # normalize each run to its own starting objective so runs are comparable on one axis
        all_histories.append(history / history[0])
    all_histories = np.array(all_histories)  # (num_runs, n_outer+1)
    mean_h = all_histories.mean(axis=0)
    std_h = all_histories.std(axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    iters = np.arange(len(mean_h))
    for h in all_histories:
        ax.plot(iters, h, color="#1C7293", alpha=0.15, linewidth=1)
    ax.plot(iters, mean_h, "o-", color="#D1495B", linewidth=2.5, markersize=5, label=f"mean over {num_runs} runs")
    ax.fill_between(iters, mean_h - std_h, mean_h + std_h, color="#D1495B", alpha=0.15)
    ax.set_xlabel("Outer sweep (full pass over all N elements)")
    ax.set_ylabel("Objective / initial objective")
    ax.set_title("RIS Coordinate-Ascent Optimizer: Convergence")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "ris_optimizer_convergence.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    print(f"Average objective improvement after {len(mean_h)-1} sweeps: {mean_h[-1]:.1f}x initial (monotonic per-run by construction)")
    return mean_h, std_h


def save_summary_csv(exp_a, exp_b):
    path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "x_value", "config", "mean_sum_rate", "std_sum_rate"])
        x_vals_a, res_a = exp_a
        for label, _, _ in CONFIGS:
            for x, m, s in zip(x_vals_a, res_a[label]["mean"], res_a[label]["std"]):
                writer.writerow(["ris_elements", x, label, m, s])
        x_vals_b, res_b = exp_b
        for label, _, _ in CONFIGS:
            for x, m, s in zip(x_vals_b, res_b[label]["mean"], res_b[label]["std"]):
                writer.writerow(["tx_power_db", x, label, m, s])
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    exp_a = experiment_ablation_vs_ris_elements(num_trials=40)
    exp_b = experiment_ablation_vs_power(num_trials=40)
    experiment_convergence(num_runs=15, num_ris_elements=32)
    save_summary_csv(exp_a, exp_b)

    # headline numbers for the presentation / report
    n_idx = -1  # N=256
    ris_opt = exp_a[1]["RIS (optimized) + NOMA"]["mean"][n_idx]
    no_ris_oma = exp_a[1]["No-RIS + OMA"]["mean"][n_idx]
    no_ris_noma = exp_a[1]["No-RIS + NOMA"]["mean"][n_idx]
    print(f"\nHeadline @ N=256: RIS(optimized)+NOMA = {ris_opt:.2f} bits/s/Hz "
          f"vs No-RIS+OMA = {no_ris_oma:.2f}  (x{ris_opt/no_ris_oma:.1f})  "
          f"vs No-RIS+NOMA = {no_ris_noma:.2f}  (x{ris_opt/no_ris_noma:.1f})")
