"""
channel_model.py
=================
Generates the wireless channels for the RIS-assisted, multi-user NOMA
system described in the Review-1 proposal:

    Base Station (BS) --H1--> RIS --H2--> Users
                      \\----------- h_d (weak direct link) ------------/

- H1  : BS -> RIS channel      (Nr x Nt), usually near-LoS  -> Rician
- H2  : RIS -> user_k channel  (Nr,)      often more scattered -> Rician/Rayleigh
- h_d : BS -> user_k direct link (Nt,)    typically blocked/weak -> Rayleigh

All channels are complex baseband gains. Large-scale path loss is
combined with small-scale Rician/Rayleigh fading.
"""

import numpy as np


def _rician_fading(shape, k_factor_db):
    """Draw complex Rician-faded coefficients.

    Parameters
    ----------
    shape : tuple
        Output array shape.
    k_factor_db : float
        Rician K-factor in dB. Use -np.inf (or a very negative number)
        for pure Rayleigh fading (no LoS component).
    """
    k = 10 ** (k_factor_db / 10.0)
    los = np.sqrt(k / (k + 1))
    nlos_scale = np.sqrt(1 / (2 * (k + 1)))
    los_component = np.ones(shape, dtype=complex)  # normalized LoS phase reference
    nlos_component = nlos_scale * (np.random.randn(*shape) + 1j * np.random.randn(*shape))
    return los * los_component + nlos_component


def path_loss_linear(distance_m, ref_distance_m=1.0, path_loss_exp=2.2, ref_loss_db=30.0):
    """Simple log-distance path loss model, returns LINEAR (not dB) gain."""
    pl_db = ref_loss_db + 10 * path_loss_exp * np.log10(distance_m / ref_distance_m)
    return 10 ** (-pl_db / 10.0)


class RISChannelModel:
    """Bundles BS-RIS, RIS-user, and BS-user (direct) channels for K users."""

    def __init__(self, num_bs_antennas, num_ris_elements, num_users,
                 d_bs_ris=30.0, d_ris_user=(5.0, 15.0), d_bs_user=(40.0, 60.0),
                 k_bs_ris_db=10.0, k_ris_user_db=5.0, k_direct_db=-np.inf,
                 seed=None):
        self.Nt = num_bs_antennas
        self.Nr = num_ris_elements
        self.K = num_users
        if seed is not None:
            np.random.seed(seed)

        # --- BS -> RIS channel (shared by all users) ---
        pl_bs_ris = path_loss_linear(d_bs_ris)
        self.H1 = np.sqrt(pl_bs_ris) * _rician_fading((self.Nr, self.Nt), k_bs_ris_db)

        # --- RIS -> user_k channels, and direct BS -> user_k links ---
        self.H2 = []   # list of (Nr,) arrays
        self.Hd = []   # list of (Nt,) arrays
        rng_ris_user = np.random.default_rng(seed)
        for _ in range(self.K):
            d_ru = rng_ris_user.uniform(*d_ris_user)
            d_bu = rng_ris_user.uniform(*d_bs_user)
            pl_ru = path_loss_linear(d_ru)
            pl_bu = path_loss_linear(d_bu, path_loss_exp=3.5)  # direct link usually more obstructed
            self.H2.append(np.sqrt(pl_ru) * _rician_fading((self.Nr,), k_ris_user_db))
            self.Hd.append(np.sqrt(pl_bu) * _rician_fading((self.Nt,), k_direct_db))

    def effective_channel(self, user_idx, ris_phase_vector, beamformer):
        """Combine direct + RIS-reflected path into one scalar effective channel.

        Parameters
        ----------
        user_idx : int
        ris_phase_vector : (Nr,) complex array with unit-modulus entries e^{j*theta_n}
        beamformer : (Nt,) complex transmit beamforming vector (unit norm typically)

        Returns
        -------
        complex scalar effective channel gain g_k
        """
        Theta = np.diag(ris_phase_vector)
        h2 = self.H2[user_idx]
        hd = self.Hd[user_idx]
        cascaded = h2.conj().T @ Theta @ self.H1  # (Nt,)
        g_k = cascaded @ beamformer + hd @ beamformer
        return g_k


if __name__ == "__main__":
    model = RISChannelModel(num_bs_antennas=4, num_ris_elements=32, num_users=3, seed=1)
    theta = np.exp(1j * np.zeros(model.Nr))       # RIS OFF (phases = 0) as a sanity check
    bf = np.ones(model.Nt) / np.sqrt(model.Nt)    # naive equal-gain beamformer
    for k in range(model.K):
        g = model.effective_channel(k, theta, bf)
        print(f"User {k}: |g_k|^2 = {abs(g) ** 2:.3e}")
