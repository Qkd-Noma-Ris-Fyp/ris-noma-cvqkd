from bb84_qkd import BB84QKDProtocol

def evaluate_hybrid_qkd_system():
    print("Initializing RIS-NOMA & BB84 Integrated Simulation...")

    # Example: Suppose your RIS configuration outputs a specific channel efficiency/SNR
    # Let's derive channel efficiency from an effective RIS gain model
    ris_elements = 64
    channel_gain_boost = np.sqrt(ris_elements) * 0.1  # Simplified model proxy
    transmission_efficiency = min(0.95, 0.6 + channel_gain_boost * 0.05)

    print(f"Calculated RIS Transmission Efficiency: {transmission_efficiency:.4f}")

    # Run BB84 Protocol simulation under current RIS conditions
    bb84 = BB84QKDProtocol(num_bits=5000, eavesdropping=False)
    results = bb84.run_bb84(transmission_efficiency=transmission_efficiency, error_rate=0.015)

    print("\n--- BB84 QKD Simulation Results ---")
    print(f"Total Raw Bits Generated : {results['raw_bits']}")
    print(f"Final Sifted Key Length  : {results['sifted_key_length']}")
    print(f"Estimated QBER           : {results['qber'] * 100:.2f}%")
    
    if results['qber'] < 0.11: # Standard security threshold for BB84
        print("Status: Secure Key Successfully Established!")
    else:
        print("Status: High QBER Detected. Connection Aborted.")

if __name__ == "__main__":
    evaluate_hybrid_qkd_system()