"""
extra_metrics.py
================
Additional performance metrics layered on top of the core RIS/NOMA/QKD
pipeline (channel_model -> ris_optimizer -> noma_allocation -> qkd_model).

Each function here is a SEPARATE, clearly-labeled add-on -- it does not
change any of the core physics already implemented. Read each docstring
before quoting a number from these to a reviewer: several of these are
simplifying assumptions (explicitly flagged below), not universal truths.

Implements:
    - ber_bpsk(sinr)                  Bit Error Rate (uncoded BPSK over AWGN)
    - sic_decode_success(...)         SIC decoding success (SINR above threshold)
    - outage_probability(...)         P(achievable rate < target threshold)
    - energy_efficiency(...)          bits/s/Hz per unit transmit power
    - quantum_transmission_probability(...)   alias/wrapper of channel transmittance
    - key_consumption_rate(...)       application-layer key-usage ratio (OTP assumption)
"""

import numpy as np
from scipy.special import erfc


def ber_bpsk(sinr_linear):
    """Bit Error Rate for uncoded BPSK over an AWGN channel.

    ASSUMPTION: this assumes BPSK modulation specifically. The actual
    modulation scheme used by the NOMA/QKD data link isn't fixed anywhere
    else in this project -- treat this as "what the BER WOULD be if we
    used plain BPSK at this SINR," a common illustrative choice in papers
    that don't fix a modulation scheme, not a measurement of your actual
    system's bit errors. If your guide specifies QPSK or M-QAM instead,
    say so and swap in that modulation's BER formula.

    Formula: BER = Q(sqrt(2*SINR)) = 0.5 * erfc(sqrt(SINR))
    (standard result, e.g. Proakis, "Digital Communications")

    Parameters
    ----------
    sinr_linear : scalar or array, linear (not dB) SINR

    Returns
    -------
    ber : same shape as input, each value in [0, 0.5]
    """
    sinr_linear = np.asarray(sinr_linear, dtype=float)
    return 0.5 * erfc(np.sqrt(np.clip(sinr_linear, 0, None)))


def sic_decode_success(sinr_linear, sinr_threshold_linear):
    """Whether SIC decoding succeeds for each user, given a minimum
    SINR a receiver needs to reliably decode and cancel a signal.

    ASSUMPTION: real SIC failure depends on receiver hardware, coding,
    and more -- this is a simplified "hard threshold" model: decoding
    succeeds if SINR is above the threshold, fails otherwise. This is
    the standard simplification used for system-level NOMA studies
    before moving to link-level simulation.

    Parameters
    ----------
    sinr_linear : (K,) array, linear SINR per user (from sic_sinr_values)
    sinr_threshold_linear : scalar, minimum SINR needed to decode
        (e.g. 10**(0/10) = 1.0 for a 0 dB threshold)

    Returns
    -------
    success : (K,) boolean array
    """
    sinr_linear = np.asarray(sinr_linear, dtype=float)
    return sinr_linear >= sinr_threshold_linear


def outage_probability(rate_samples, rate_threshold):
    """Fraction of Monte Carlo trials where the achievable rate fell
    below a target threshold -- the standard definition of outage
    probability in wireless systems literature.

    Parameters
    ----------
    rate_samples : (num_trials,) array of sum-rate (or per-user rate)
        values collected across independent channel realizations
    rate_threshold : scalar, the minimum acceptable rate (bits/s/Hz)

    Returns
    -------
    p_outage : float in [0, 1]
    """
    rate_samples = np.asarray(rate_samples, dtype=float)
    return float(np.mean(rate_samples < rate_threshold))


def energy_efficiency(rate_bits_per_s_hz, tx_power, circuit_power=0.0):
    """Energy efficiency: bits delivered per unit of energy spent.

    EE = rate / (tx_power + circuit_power)

    `circuit_power` optionally adds a fixed hardware power cost (RIS
    control circuitry, baseband processing, etc.) on top of the radiated
    transmit power -- set to 0 if you only want pure radiated-power
    efficiency. Units: bits/s/Hz per watt (relative, since this project
    uses normalized power units, not absolute watts -- see channel_model.py).

    Parameters
    ----------
    rate_bits_per_s_hz : scalar or array, achievable rate
    tx_power : scalar, transmit power used to achieve that rate
    circuit_power : scalar, optional fixed overhead power

    Returns
    -------
    ee : same shape as rate_bits_per_s_hz
    """
    total_power = tx_power + circuit_power
    return rate_bits_per_s_hz / total_power if total_power > 0 else np.inf


def quantum_transmission_probability(effective_channel_gain_linear, reference_gain=1.0):
    """Quantum channel transmission probability T.

    NOTE: in CV-QKD, the "transmission probability" and the "channel
    transmittance" are the SAME quantity -- this is a thin, clearly-named
    wrapper around qkd_model.channel_transmittance() so you can report it
    under either name without duplicating the math. If asked "what's your
    quantum transmission probability," point to this -- it's not a new
    computation, it's the existing transmittance under its other name.
    """
    from qkd.qkd_model import channel_transmittance
    return channel_transmittance(effective_channel_gain_linear, reference_gain)


def key_consumption_rate(secret_key_rate_bits_per_symbol, application_data_rate_bits_per_s,
                          symbol_rate_hz=1.0, key_bits_per_data_bit=1.0):
    """How fast generated secret key bits are consumed by the application
    that uses them, relative to how fast they're produced.

    ASSUMPTION: this models the simplest case -- a One-Time-Pad-style
    scheme that consumes `key_bits_per_data_bit` key bits for every
    application data bit sent (1.0 for true OTP; set lower, e.g. for an
    AES-rekeying scheme that only periodically consumes a fresh key, if
    your guide wants that model instead). This is an APPLICATION-LAYER
    metric, not a physical-layer one -- it's not computed anywhere else
    in this project, so say so explicitly if asked why it wasn't in the
    original results.

    Parameters
    ----------
    secret_key_rate_bits_per_symbol : SKR from qkd_model.secret_key_rate()
    application_data_rate_bits_per_s : how fast the app needs to send data
    symbol_rate_hz : QKD symbol rate (symbols/sec), to convert SKR to bits/s
    key_bits_per_data_bit : key bits consumed per app data bit (1.0 = OTP)

    Returns
    -------
    ratio : float. ratio < 1 means keys are produced FASTER than consumed
        (healthy, a key buffer builds up). ratio > 1 means the application
        would run out of key material (key-starved).
    """
    key_production_rate_bits_per_s = secret_key_rate_bits_per_symbol * symbol_rate_hz
    key_demand_rate_bits_per_s = application_data_rate_bits_per_s * key_bits_per_data_bit
    if key_production_rate_bits_per_s <= 0:
        return np.inf
    return key_demand_rate_bits_per_s / key_production_rate_bits_per_s


if __name__ == "__main__":
    # Quick standalone sanity checks
    sinrs = np.array([20.0, 5.0, 0.5])  # linear SINR for 3 users, strong->weak
    print("BER (BPSK):", ber_bpsk(sinrs))
    print("SIC success @ 0dB threshold:", sic_decode_success(sinrs, sinr_threshold_linear=1.0))

    rates = np.array([8.2, 7.9, 1.1, 8.5, 0.9, 7.7])  # pretend Monte Carlo samples
    print("Outage probability @ threshold=2.0:", outage_probability(rates, rate_threshold=2.0))

    print("Energy efficiency (rate=10.5, power=1.0):", energy_efficiency(10.5, tx_power=1.0))

    print("Key consumption ratio (SKR=0.08, app needs 0.05 bits/s, symbol_rate=1Hz):",
          key_consumption_rate(0.08, application_data_rate_bits_per_s=0.05, symbol_rate_hz=1.0))
