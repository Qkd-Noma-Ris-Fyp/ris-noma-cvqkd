"""
qkd_model.py
============
Gaussian-modulated coherent-state CV-QKD, HETERODYNE (no-switching) detection,
reverse reconciliation, asymptotic key length, collective attacks.

    SKR = beta * I(A:B) - chi(B:E)          [bits / channel use]

Model (entanglement-based picture, shot-noise units, SNU):
    V_A   : Alice's modulation variance           V = V_A + 1
    T     : channel transmittance (0, 1]
    xi    : excess noise referred to the channel input
    chi_line = 1/T - 1 + xi       (noise added by the channel, input-referred)

Detector is assumed IDEAL here (unit efficiency, no electronic noise).
Realistic trusted-detector noise is a planned extension.

References for the formulas:
    Laudenbach et al., "Continuous-Variable Quantum Key Distribution with
    Gaussian Modulation - The Theory of Practical Implementations",
    Adv. Quantum Technol. 1, 1800011 (2018).
    Garcia-Patron & Cerf, PRL 97, 190503 (2006)  (collective-attack bound).

Derivation of chi(B:E) used below:
    chi(B:E) = S(E) - S(E|b)
    S(E)   = S(AB) = G((l1-1)/2) + G((l2-1)/2)          (symplectic eigenvalues
                                                          of the Alice-Bob state)
    S(E|b) = S(A|b) = G((V_A|b - 1)/2)    since A,B,E is pure and Bob's
                                          heterodyne is a rank-one measurement
    G(x)   = (x+1) log2(x+1) - x log2(x),  G(0) = 0
"""

import numpy as np


def channel_transmittance(effective_channel_gain_linear, reference_gain):
    """Map RF effective gain |g|^2 to a QKD transmittance T in (0, 1].

    PLACEHOLDER COUPLING: T = |g|^2 / reference_gain, with reference_gain a
    FIXED calibration constant (must not depend on the sample being evaluated).
    A physical link budget would replace this mapping.
    """
    T = effective_channel_gain_linear / reference_gain
    return float(np.clip(T, 1e-9, 1.0))


def _G(x):
    """Entropy building block G(x) = (x+1)log2(x+1) - x log2(x), G(0)=0."""
    x = max(float(x), 0.0)
    if x < 1e-12:
        return 0.0
    return (x + 1) * np.log2(x + 1) - x * np.log2(x)


def mutual_information_AB(T, excess_noise, modulation_variance):
    """Heterodyne mutual information between Alice and Bob.

    I(A:B) = log2[(V + chi_tot) / (1 + chi_tot)],  chi_tot = chi_line + 1/T
           = log2(1 + T * V_A / (2 + T * xi))            (ideal detector)
    """
    snr = T * modulation_variance / (2.0 + T * excess_noise)
    return float(np.log2(1.0 + snr))


def holevo_bound(T, excess_noise, modulation_variance):
    """Holevo bound chi(B:E) under a Gaussian collective attack."""
    V = modulation_variance + 1.0
    chi_line = 1.0 / T - 1.0 + excess_noise

    # Symplectic eigenvalues of the Alice-Bob covariance matrix
    A = V ** 2 * (1 - 2 * T) + 2 * T + T ** 2 * (V + chi_line) ** 2
    B = T ** 2 * (V * chi_line + 1) ** 2
    disc = np.sqrt(max(A ** 2 - 4 * B, 0.0))
    lam1 = np.sqrt(max(0.5 * (A + disc), 1.0))
    lam2 = np.sqrt(max(0.5 * (A - disc), 1.0))
    S_E = _G((lam1 - 1) / 2) + _G((lam2 - 1) / 2)

    # Alice's conditional variance after Bob's heterodyne measurement
    V_A_given_b = V - T * (V ** 2 - 1) / (T * (V + chi_line) + 1.0)
    S_E_given_b = _G((V_A_given_b - 1) / 2)

    return float(max(S_E - S_E_given_b, 0.0))


def secret_key_rate(effective_channel_gain_linear, excess_noise=0.01,
                    modulation_variance=4.0, beta=0.95, reference_gain=1.0):
    """End-to-end SKR for one user's channel.

    Returns (skr, I_AB, chi_BE, T). `reference_gain` must be a FIXED constant.
    """
    T = channel_transmittance(effective_channel_gain_linear, reference_gain)
    I_AB = mutual_information_AB(T, excess_noise, modulation_variance)
    chi_BE = holevo_bound(T, excess_noise, modulation_variance)
    return max(beta * I_AB - chi_BE, 0.0), I_AB, chi_BE, T


if __name__ == "__main__":
    for T in [1.0, 0.8, 0.5, 0.3, 0.1, 0.01]:
        skr, i_ab, chi, _ = secret_key_rate(T, reference_gain=1.0)
        print(f"T={T:5.2f}  I(A:B)={i_ab:.3f}  chi(B:E)={chi:.3f}  SKR={skr:.3f}")
