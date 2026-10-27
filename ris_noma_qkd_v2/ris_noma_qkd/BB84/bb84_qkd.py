import numpy as np
import random

class BB84QKDProtocol:
    def __init__(self, num_bits: int = 10000, eavesdropping: bool = False):
        """
        Initializes the BB84 QKD simulation parameters.
        :param num_bits: Number of raw bits to transmit
        :param eavesdropping: Boolean flag to simulate Eve's presence (intercept-resend attack)
        """
        self.num_bits = num_bits
        self.eavesdropping = eavesdropping
        
        # Storage for states and measurements
        self.alice_bits = []
        self.alice_bases = []
        self.bob_bases = []
        self.bob_bits = []
        self.shared_key = []
        self.qber = 0.0

    def generate_alice_states(self):
        """Step 1: Alice generates random bits and random bases (0: Z-basis +, 1: X-basis x)."""
        self.alice_bits = [random.randint(0, 1) for _ in range(self.num_bits)]
        self.alice_bases = [random.randint(0, 1) for _ in range(self.num_bits)]

    def quantum_channel_transmission(self, transmission_efficiency: float = 0.8, error_rate: float = 0.02):
        """
        Step 2: Simulate quantum transmission through the channel (factoring in RIS gain / path loss).
        """
        received_bits = []
        self.bob_bases = [random.randint(0, 1) for _ in range(self.num_bits)]

        for i in range(self.num_bits):
            bit = self.alice_bits[i]
            a_basis = self.alice_bases[i]
            b_basis = self.bob_bases[i]

            # Simulate eavesdropping (Eve intercepts and measures in a random basis)
            if self.eavesdropping:
                eve_basis = random.randint(0, 1)
                if eve_basis != a_basis:
                    bit = random.randint(0, 1) # Perturbation due to wrong basis choice by Eve

            # Channel loss / photon loss simulation
            if random.random() > transmission_efficiency:
                received_bits.append(None) # Photon lost
                continue

            # Transmission through matching/mismatching bases
            if a_basis == b_basis:
                # Introduce slight hardware/channel noise errors
                if random.random() < error_rate:
                    bit = 1 - bit
                received_bits.append(bit)
            else:
                # Random outcome if bases do not match
                received_bits.append(random.randint(0, 1))

        self.bob_bits = received_bits

    def sifting_process(self):
        """Step 3: Basis reconciliation (Sifting phase)."""
        sifted_alice = []
        sifted_bob = []

        for i in range(self.num_bits):
            if self.bob_bits[i] is not None and self.alice_bases[i] == self.bob_bases[i]:
                sifted_alice.append(self.alice_bits[i])
                sifted_bob.append(self.bob_bits[i])

        return sifted_alice, sifted_bob

    def estimate_qber(self, sifted_alice, sifted_bob, sample_fraction: float = 0.2):
        """Step 4: Estimate Quantum Bit Error Rate (QBER) using a subset of sifted keys."""
        sample_size = int(len(sifted_alice) * sample_fraction)
        if sample_size == 0:
            self.qber = 0.0
            return sifted_alice, sifted_bob

        sample_indices = random.sample(range(len(sifted_alice)), sample_size)
        
        errors = 0
        test_alice = []
        test_bob = []
        
        for idx in sorted(sample_indices, reverse=True):
            if sifted_alice[idx] != sifted_bob[idx]:
                errors += 1
            test_alice.append(sifted_alice.pop(idx))
            test_bob.append(sifted_bob.pop(idx))

        self.qber = errors / sample_size if sample_size > 0 else 0
        return sifted_alice, sifted_bob

    def run_bb84(self, transmission_efficiency=0.85, error_rate=0.01):
        """Executes the full BB84 simulation pipeline."""
        self.generate_alice_states()
        self.quantum_channel_transmission(transmission_efficiency, error_rate)
        sift_a, sift_b = self.sifting_process()
        
        if len(sift_a) > 0:
            final_a, final_b = self.estimate_qber(sift_a, sift_b)
            # Assuming error correction/privacy amplification yields final shared key
            self.shared_key = final_a 
        else:
            self.shared_key = []

        return {
            "raw_bits": self.num_bits,
            "sifted_key_length": len(self.shared_key),
            "qber": self.qber,
            "shared_key": self.shared_key
        }