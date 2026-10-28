"""BB84 QKD simulation (sifting, QBER test, intercept-resend Eve, secret fraction)."""
import numpy as np


def binary_entropy(x):
    x = float(x)
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


class BB84QKDProtocol:
    def __init__(self, num_bits=5000, eavesdropping=False, seed=None,
                 qber_threshold=0.11, sample_fraction=0.2):
        self.n = num_bits
        self.eve = eavesdropping
        self.thresh = qber_threshold
        self.frac = sample_fraction
        self.rng = np.random.default_rng(seed)

    def run_bb84(self, transmission_efficiency=0.85, error_rate=0.01):
        rng, n = self.rng, self.n
        bit = rng.integers(0, 2, n)
        alice_basis = rng.integers(0, 2, n)
        bob_basis = rng.integers(0, 2, n)
        received = bit.copy()
        if self.eve:  # intercept-resend: wrong Eve basis -> resent bit is random
            wrong = rng.integers(0, 2, n) != alice_basis
            received = np.where(wrong, rng.integers(0, 2, n), received)
        keep = (rng.random(n) <= transmission_efficiency) & (alice_basis == bob_basis)
        a_s, b_s = bit[keep], received[keep]
        b_s = np.where(rng.random(len(a_s)) < error_rate, 1 - b_s, b_s)
        sifted = len(a_s)
        m = int(self.frac * sifted)
        qber = float(np.mean(a_s[:m] != b_s[:m])) if m else 0.0
        after_test = sifted - m
        secure = qber < self.thresh
        r = max(0.0, 1 - 2 * binary_entropy(qber)) if secure else 0.0
        return {
            "raw_bits": n,
            "sifted_length": sifted,
            "key_after_qber_test": after_test,
            "qber": qber,
            "secret_fraction": r,
            "secret_key_length": int(r * after_test),
            "secure": secure,
        }