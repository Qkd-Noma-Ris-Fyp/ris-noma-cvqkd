"""RIS-NOMA-QKD model. Sections map to slide 8: channel, RIS, NOMA, CV-QKD, BB84."""
import numpy as np

SIG = 1e-9          # noise power (slide 9)
NS = [4, 8, 16, 32, 64, 128, 256]

# ---------------- 1. CHANNEL ----------------
def _amp(d, expo):                       # path loss: 30 dB at 1 m
    return 10 ** (-(30 + 10 * expo * np.log10(d)) / 20)

def _rician(rng, shape, k_db):           # unit-power Rician fading
    K = 10 ** (k_db / 10)
    los = np.sqrt(K / (K + 1)) * np.exp(1j * 2 * np.pi * rng.random(shape))
    nlos = np.sqrt(1 / (2 * (K + 1))) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    return los + nlos

def gen_channel(N, K, seed):
    """Returns A (K x N cascaded BS-RIS-user) and d (K direct, Rayleigh)."""
    rng = np.random.default_rng(seed)
    H1 = _rician(rng, N, 10) * _amp(50, 2.2)                       # BS -> RIS
    dru = rng.uniform(7, 16, K)                                    # RIS -> user distances
    h2 = np.stack([_rician(rng, N, 5) * _amp(x, 2.2) for x in dru])
    A = h2 * H1
    d = _rician(rng, K, -100) * np.array([_amp(x, 3.5) for x in rng.uniform(42, 60, K)])
    return A, d

def gains(A, d, theta=None):
    g = d if theta is None else d + A @ theta
    return np.abs(g) ** 2

# ---------------- 2. RIS OPTIMISER ----------------
def heuristic(A, d):                     # align every element to the weakest-direct user
    k = np.argmin(np.abs(d))
    return np.exp(1j * (np.angle(d[k]) - np.angle(A[k])))

def optimise(A, d, sweeps=12):
    """Coordinate ascent: phi_n* = -arg( sum_k w_k a_kn conj(B_k) )  (closed form, monotone)."""
    K, N = A.shape
    w = 1 / np.sum(np.abs(A) ** 2, axis=1)
    th = heuristic(A, d); g = d + A @ th
    obj = lambda: float(np.sum(w * np.abs(g) ** 2))
    hist = [obj()]
    for _ in range(sweeps):
        for n in range(N):
            B = g - A[:, n] * th[n]
            th[n] = np.exp(-1j * np.angle(np.sum(w * A[:, n] * np.conj(B))))
            g = B + A[:, n] * th[n]
        hist.append(obj())
    return th, np.abs(g) ** 2, np.array(hist) / hist[0]

# ---------------- 3. NOMA ----------------
def noma(g, P, alpha=0.6):
    """Weakest user gets most power; user k is interfered by STRONGER users (SIC)."""
    idx = np.argsort(g); gs = g[idx]
    p = gs ** (-alpha); p = P * p / p.sum()
    rows = []
    for j in range(len(gs)):
        I = p[j + 1:].sum()
        sinr = p[j] * gs[j] / (gs[j] * I + SIG)
        rows.append(dict(user=int(idx[j]) + 1, gain_db=10 * np.log10(gs[j]), power=p[j] / P,
                         sinr_db=10 * np.log10(sinr), rate=np.log2(1 + sinr)))
    return rows, sum(r["rate"] for r in rows)

def oma(g, P):
    return float(np.mean(np.log2(1 + P * g / SIG)))

# ---------------- 4. CV-QKD ----------------
def cvqkd_skr(g, G_REF, VA=4.0, xi=0.01, beta=0.95):
    T = np.minimum(np.asarray(g) / G_REF, 1.0)                         # PLACEHOLDER coupling
    I = np.log2(1 + T * VA / (2 + T * xi))
    chi = 0.5 * np.log2(1 + (1 - T) * VA / 2 + 20 * xi * T)           # simplified Holevo proxy
    return np.maximum(0, beta * I - chi)

# ---------------- 5. BB84 ----------------
def hb(x):
    return 0.0 if x <= 0 or x >= 1 else float(-x * np.log2(x) - (1 - x) * np.log2(1 - x))

def eta_of_N(N):                                                       # RIS proxy (slide 10)
    return min(0.95, 0.6 + 0.005 * np.sqrt(N))

def bb84(n, eta, err, eve=False, seed=0, thresh=0.11):
    rng = np.random.default_rng(seed)
    bit, ab, bb = rng.integers(0, 2, n), rng.integers(0, 2, n), rng.integers(0, 2, n)
    b = bit.copy()
    if eve:                                                            # intercept-resend
        wrong = rng.integers(0, 2, n) != ab
        b = np.where(wrong, rng.integers(0, 2, n), b)
    keep = (rng.random(n) <= eta) & (ab == bb)                         # loss + sifting
    a_s, b_s = bit[keep], b[keep]
    b_s = np.where(rng.random(len(a_s)) < err, 1 - b_s, b_s)
    ns = len(a_s); m = int(0.2 * ns)                                   # 20% sampled for QBER
    q = float(np.mean(a_s[:m] != b_s[:m])) if m else 0.0
    after = ns - m
    r = 0.0 if q >= thresh else max(0.0, 1 - 2 * hb(q))
    return dict(raw=n, sifted=ns, after_test=after, secret=int(r * after), qber=q, r=r, secure=q < thresh)
