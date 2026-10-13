"""
noma_allocation.py
===================
Power-domain NOMA for K users sharing one time-frequency resource block,
served over the RIS-assisted effective channels computed upstream.

Implements:
- Fixed fractional power allocation (weaker channel -> more power),
  the standard baseline before moving to optimized allocation.
- Successive Interference Cancellation (SIC) achievable-rate calculation,
  assuming perfect SIC ordering by descending effective channel gain.
"""

import numpy as np


def fractional_power_allocation(channel_gains, total_power, alpha=0.6):
    """Assign more power to weaker users (standard NOMA fairness heuristic).

    Parameters
    ----------
    channel_gains : (K,) array of |g_k|^2 (linear, not dB)
    total_power   : scalar, total transmit power budget
    alpha         : fairness exponent in (0, 1]; higher alpha -> more
                    power skew toward weak users. alpha=0 -> equal power.

    Returns
    -------
    powers : (K,) array, sums to total_power
    order  : indices sorted by descending channel gain (SIC decoding order)
    """
    channel_gains = np.asarray(channel_gains, dtype=float)
    order = np.argsort(-channel_gains)  # strongest first (decodes first in SIC)

    # weaker channel -> larger weight -> larger power share
    weights = (1.0 / channel_gains) ** alpha
    weights = weights / weights.sum()
    powers = weights * total_power
    return powers, order


def sic_achievable_rates(channel_gains, powers, order, noise_power):
    """Compute per-user Shannon rates under NOMA + perfect SIC.

    Convention: the strongest user is decoded FIRST and must cancel the
    signals of all users decoded after it -> strongest user still sees
    interference from every weaker (not-yet-decoded) user? No -- standard
    NOMA SIC: the receiver with the STRONGEST channel decodes and strips
    the signals of users with WEAKER channels (allocated more power)
    first, so the strong user ultimately sees no interference, while the
    weakest user decodes its own signal treating all stronger users'
    (lower-power) signals as noise.

    Parameters
    ----------
    channel_gains : (K,) |g_k|^2
    powers        : (K,) power allocated per user (from fractional_power_allocation)
    order         : indices sorted strongest -> weakest (from fractional_power_allocation)
    noise_power   : scalar noise power (N0 * B)

    Returns
    -------
    rates : (K,) achievable rate in bits/s/Hz, indexed by ORIGINAL user index
    """
    K = len(channel_gains)
    rates = np.zeros(K)
    # order[0] = strongest user (least power) ... order[K-1] = weakest user (most power)
    # Every receiver decodes signals in DESCENDING power order (i.e. from
    # weakest-channel user down to its own) and cancels those. A user can
    # only cancel users ranked AFTER it (weaker channel, higher power);
    # it cannot cancel users ranked BEFORE it (stronger channel, lower
    # power) -- those remain as uncancellable interference. This is why
    # the strongest user ends up interference-free, and the weakest user
    # suffers interference from everyone else.
    for rank, user in enumerate(order):
        g = channel_gains[user]
        p_own = powers[user]
        stronger_users = order[:rank]  # already "ahead" in channel quality, lower power, uncancellable
        interference_power = powers[stronger_users].sum() if len(stronger_users) else 0.0
        sinr = (p_own * g) / (interference_power * g + noise_power)
        rates[user] = np.log2(1 + sinr)
    return rates


def sic_sinr_values(channel_gains, powers, order, noise_power):
    """Compute per-user SINR values under the same model as sic_achievable_rates."""
    channel_gains = np.asarray(channel_gains, dtype=float)
    powers = np.asarray(powers, dtype=float)
    sinrs = np.zeros(len(channel_gains))

    for rank, user in enumerate(order):
        interference_users = order[:rank]
        interference_power = (
            powers[interference_users].sum()
            if len(interference_users) else 0.0
        )

        sinrs[user] = (
            powers[user] * channel_gains[user]
        ) / (
            interference_power * channel_gains[user] + noise_power
        )

    return sinrs


def sum_rate(channel_gains, total_power, noise_power, alpha=0.6):
    """Convenience wrapper: allocate power, run SIC, return total + per-user rates."""
    powers, order = fractional_power_allocation(channel_gains, total_power, alpha)
    rates = sic_achievable_rates(channel_gains, powers, order, noise_power)
    return rates.sum(), rates, powers, order


def oma_sum_rate(channel_gains, total_power, noise_power):
    """Orthogonal Multiple Access (OMA / TDMA) baseline: each user gets an
    equal share 1/K of the time-frequency resource with full power when
    scheduled, no superposition, no SIC, no inter-user interference at all.

    This is the standard baseline every NOMA paper benchmarks against --
    the gap between `sum_rate()` and this function IS the NOMA gain.
    """
    channel_gains = np.asarray(channel_gains, dtype=float)
    K = len(channel_gains)
    per_user_rate = (1.0 / K) * np.log2(1 + total_power * channel_gains / noise_power)
    return per_user_rate.sum(), per_user_rate


if __name__ == "__main__":
    # Toy 3-user example
    gains = np.array([1.0e-3, 4.0e-4, 1.0e-4])   # strong, medium, weak
    P_total = 1.0
    N0 = 1.0e-6

    total, rates, powers, order = sum_rate(gains, P_total, N0, alpha=0.6)
    print("Decode order (strong->weak):", order)
    print("Powers:", powers)
    print("Per-user rates (bits/s/Hz):", rates)
    print("Sum rate:", total)
