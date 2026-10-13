"""Sanity checks. Run from the repo root:  python tests/test_sanity.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channel.channel_model import RISChannelModel
from ris.ris_optimizer import coordinate_ascent_weighted_sum
from noma.noma_allocation import sum_rate
from qkd.qkd_model import secret_key_rate, mutual_information_AB


def test_optimizer_never_decreases():
    m = RISChannelModel(4, 32, 3, seed=1)
    _, h = coordinate_ascent_weighted_sum(m.H1, m.H2, m.Hd, n_outer=8, grid_size=60, seed=1)
    assert np.all(np.diff(h) >= -1e-18), "objective must be non-decreasing"
    assert h[-1] > h[0]


def test_optimized_theta_is_unit_modulus():
    m = RISChannelModel(4, 16, 3, seed=2)
    theta, _ = coordinate_ascent_weighted_sum(m.H1, m.H2, m.Hd, n_outer=3, grid_size=60, seed=2)
    assert np.allclose(np.abs(theta), 1.0)


def test_noma_strongest_user_gets_least_power_most_rate():
    gains = np.array([1e-3, 4e-4, 1e-4])
    total, rates, powers, order = sum_rate(gains, 1.0, 1e-6)
    assert powers[0] < powers[1] < powers[2]
    assert rates[0] == rates.max()
    assert np.isclose(powers.sum(), 1.0)


def test_qkd_skr_monotone_in_transmittance():
    T = np.logspace(-3, 0, 200)
    s = [secret_key_rate(t)[0] for t in T]
    assert all(np.diff(s) >= -1e-9)


def test_qkd_mutual_information_ideal_case():
    assert np.isclose(mutual_information_AB(1.0, 0.0, 4.0), np.log2(3.0))


def test_qkd_more_noise_means_less_key():
    s = [secret_key_rate(0.8, excess_noise=x)[0] for x in [0.0, 0.02, 0.05, 0.1]]
    assert all(np.diff(s) < 0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
