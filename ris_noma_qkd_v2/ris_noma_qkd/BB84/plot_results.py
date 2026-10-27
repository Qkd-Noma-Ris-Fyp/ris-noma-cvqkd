import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bb84_qkd import BB84QKDProtocol

THRESH = 0.11
def eff(N): return min(0.95, 0.6 + np.sqrt(N) * 0.1 * 0.05)
def run(N=64, err=0.015, eve=False, seed=0, bits=5000):
    return BB84QKDProtocol(bits, eve, seed).run_bb84(eff(N), err)

fig, ax = plt.subplots(2, 2, figsize=(12, 9))

# 1. QBER with / without Eve (mean of 30 runs)
q0 = [run(seed=s)["qber"] * 100 for s in range(30)]
q1 = [run(eve=True, seed=s)["qber"] * 100 for s in range(30)]
a = ax[0, 0]
a.bar(["No Eve", "With Eve"], [np.mean(q0), np.mean(q1)],
      yerr=[np.std(q0), np.std(q1)], color=["tab:green", "tab:red"], capsize=6)
a.axhline(THRESH * 100, ls="--", c="k", label="11% threshold")
a.set_ylabel("QBER (%)"); a.set_title("QBER: secure vs eavesdropped"); a.legend()

# 2. QBER vs channel error rate
errs = np.linspace(0, 0.15, 16)
for eve, c, l in [(False, "tab:green", "No Eve"), (True, "tab:red", "With Eve")]:
    m = [np.mean([run(err=e, eve=eve, seed=s, bits=10000)["qber"] for s in range(10)]) * 100 for e in errs]
    ax[0, 1].plot(errs * 100, m, "o-", c=c, label=l)
ax[0, 1].axhline(THRESH * 100, ls="--", c="k", label="11% threshold")
ax[0, 1].set_xlabel("Channel error rate (%)"); ax[0, 1].set_ylabel("QBER (%)")
ax[0, 1].set_title("QBER vs channel noise"); ax[0, 1].legend()

# 3. Secret key length vs RIS elements
Ns = [4, 16, 64, 256, 1024, 4096]
keys = [np.mean([run(N=N, seed=s)["secret_key_length"] for s in range(20)]) for N in Ns]
effs = [eff(N) for N in Ns]
a = ax[1, 0]
a.plot(Ns, keys, "o-", c="tab:blue"); a.set_xscale("log")
a.set_xlabel("RIS elements"); a.set_ylabel("Secret key length (bits)", color="tab:blue")
a2 = a.twinx(); a2.plot(Ns, effs, "s--", c="tab:orange")
a2.set_ylabel("Transmission efficiency", color="tab:orange")
a.set_title("Secret key vs RIS size (5000 pulses)")

# 4. Bit funnel
r = run(seed=42)
labels = ["Raw", "Sifted", "After QBER test", "Secret key"]
vals = [r["raw_bits"], r["sifted_length"], r["key_after_qber_test"], r["secret_key_length"]]
b = ax[1, 1].bar(labels, vals, color="tab:purple")
for rect, v in zip(b, vals):
    ax[1, 1].text(rect.get_x() + rect.get_width() / 2, v + 40, str(v), ha="center")
ax[1, 1].set_ylabel("Bits"); ax[1, 1].set_title("Key pipeline (seed 42, no Eve)")

plt.tight_layout()
plt.savefig("bb84_results.png", dpi=150)
