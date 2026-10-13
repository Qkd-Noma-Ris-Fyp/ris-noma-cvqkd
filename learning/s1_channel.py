import numpy as np
rng = np.random.default_rng(0)

def path_loss(d, n=2.2, ref_db=30.0):
    """Linear power gain at distance d metres."""
    return 10 ** (-(ref_db + 10 * n * np.log10(d)) / 10)

def rician(shape, k_db):
    """Complex Rician coefficients with average power 1."""
    k = 10 ** (k_db / 10)
    los = np.sqrt(k / (k + 1))
    scatter = np.sqrt(1 / (2 * (k + 1))) * (rng.standard_normal(shape)
                                             + 1j * rng.standard_normal(shape))
    return los + scatter

print("path loss at 10 m      :", path_loss(10))          # 6.3e-06
print("path loss at 50 m, n=3.5:", path_loss(50, 3.5))    # 1.1e-09
x = rician(100000, 5)
print("average |h|^2          :", np.mean(abs(x) ** 2))   # ~1.0