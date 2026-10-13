import numpy as np

def G(x):
    x = max(float(x), 0.0)
    return 0.0 if x < 1e-12 else (x + 1) * np.log2(x + 1) - x * np.log2(x)

def mutual_info(T, xi, VA):
    return np.log2(1 + T * VA / (2 + T * xi))

def holevo(T, xi, VA):
    V = VA + 1
    chi_line = 1 / T - 1 + xi
    A = V**2 * (1 - 2*T) + 2*T + T**2 * (V + chi_line)**2
    B = T**2 * (V * chi_line + 1)**2
    disc = np.sqrt(max(A**2 - 4*B, 0))
    l1 = np.sqrt(max(0.5 * (A + disc), 1))
    l2 = np.sqrt(max(0.5 * (A - disc), 1))
    S_E = G((l1 - 1) / 2) + G((l2 - 1) / 2)
    VAb = V - T * (V**2 - 1) / (T * (V + chi_line) + 1)
    return max(S_E - G((VAb - 1) / 2), 0)

def skr(T, xi=0.01, VA=4.0, beta=0.95):
    return max(beta * mutual_info(T, xi, VA) - holevo(T, xi, VA), 0)

print(" T    I(A:B)  chi(B:E)   SKR")
for T in [1.0, 0.8, 0.5, 0.3, 0.1, 0.01]:
    print(f"{T:4.2f}  {mutual_info(T,0.01,4):6.3f}  {holevo(T,0.01,4):7.3f}  {skr(T):6.3f}")