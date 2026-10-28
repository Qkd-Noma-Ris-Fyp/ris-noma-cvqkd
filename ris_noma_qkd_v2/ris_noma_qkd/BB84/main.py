import numpy as np
from bb84_qkd import BB84QKDProtocol


def evaluate_hybrid_qkd_system(ris_elements=64, error_rate=0.015, seed=42):
    eta = min(0.95, 0.6 + 0.005 * np.sqrt(ris_elements))
    print(f"N = {ris_elements}, eta = {eta:.3f}")
    for eve in (False, True):
        res = BB84QKDProtocol(5000, eve, seed).run_bb84(eta, error_rate)
        print(f"\n--- BB84, Eve {'present' if eve else 'absent'} ---")
        print(f"Raw {res['raw_bits']} -> sifted {res['sifted_length']} -> "
              f"after test {res['key_after_qber_test']} -> secret {res['secret_key_length']}")
        print(f"QBER {res['qber']*100:.2f}%  r = {res['secret_fraction']:.2f}")
        print("SECURE" if res["secure"] else "ABORT (QBER >= 11%)")


if __name__ == "__main__":
    evaluate_hybrid_qkd_system()