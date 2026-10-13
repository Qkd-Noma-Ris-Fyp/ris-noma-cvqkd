import numpy as np
rng = np.random.default_rng(0)

def path_loss(d, n=2.2):
    return 10 ** (-(30 + 10 * n * np.log10(d)) / 10)

def rician(shape, k_db):
    k = 10 ** (k_db / 10)
    return np.sqrt(k / (k + 1)) + np.sqrt(1 / (2 * (k + 1))) * (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

def rayleigh(size=None):
    return (rng.standard_normal(size) + 1j * rng.standard_normal(size)) / np.sqrt(2)

# ---- Part 1: random vs aligned phases, one user ----
def trial(N):
    h1 = np.sqrt(path_loss(30)) * rician(N, 10)               # BS -> RIS
    h2 = np.sqrt(path_loss(10)) * rician(N, 5)                # RIS -> user
    hd = np.sqrt(path_loss(50, 3.5)) * rayleigh()             # direct link
    a = np.conj(h2) * h1                                      # reflected path before RIS phase
    rnd = abs(a @ np.exp(1j * rng.uniform(-np.pi, np.pi, N)) + hd) ** 2
    ali = abs(a @ np.exp(1j * (np.angle(hd) - np.angle(a))) + hd) ** 2
    return rnd, ali, abs(hd) ** 2

for N in [4, 16, 64, 256]:
    rnd, ali, d = np.mean([trial(N) for _ in range(300)], axis=0)
    print(f"N={N:3d}  random/direct=x{rnd/d:.1f}  aligned/direct=x{ali/d:.0f}")

# ---- Part 2: three users, one RIS ----
N, K = 32, 3
h1 = np.sqrt(path_loss(30)) * rician(N, 10)
h2 = [np.sqrt(path_loss(rng.uniform(5, 15))) * rician(N, 5) for _ in range(K)]
hd = np.array([np.sqrt(path_loss(rng.uniform(40, 60), 3.5)) * rayleigh() for _ in range(K)])
a = np.array([np.conj(h2[k]) * h1 for k in range(K)])         # shape (K, N)

def gains(theta):
    return np.abs(a @ theta + hd) ** 2                        # |g_k|^2 for every user

theta0 = np.exp(1j * (np.angle(hd[0]) - np.angle(a[0])))      # aligned to user 0 only
print("\naligned to user 0 only:", gains(theta0), " sum:", gains(theta0).sum())

# ---- Part 3: coordinate ascent (maximize sum of gains) ----
grid = np.exp(1j * np.linspace(-np.pi, np.pi, 120, endpoint=False))
theta = np.ones(N, dtype=complex)                             # start: all phases zero
g = a @ theta + hd
history = [np.sum(abs(g) ** 2)]

for sweep in range(10):
    for n in rng.permutation(N):
        B = g - a[:, n] * theta[n]                            # everyone else's contribution
        cand = B[:, None] + a[:, n][:, None] * grid[None, :]  # (K, 120)
        theta[n] = grid[np.argmax((abs(cand) ** 2).sum(axis=0))]
        g = B + a[:, n] * theta[n]
    history.append(np.sum(abs(g) ** 2))

print("improvement vs start:", np.round(np.array(history) / history[0], 2))
print("never decreases     :", bool(np.all(np.diff(history) >= -1e-18)))
print("final user gains    :", gains(theta), " sum:", gains(theta).sum())