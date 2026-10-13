import numpy as np
g  = np.array([1e-3, 4e-4, 1e-4])
P, N0, alpha = 1.0, 1e-6, 0.6

w = (1 / g) ** alpha
p = P * w / w.sum()
print("powers:", p.round(3))                     # [0.149 0.258 0.593]

order = np.argsort(-g)                         
rates = np.zeros(3)
for rank, u in enumerate(order):
    interference = p[order[:rank]].sum()   
    sinr = p[u] * g[u] / (interference * g[u] + N0)
    rates[u] = np.log2(1 + sinr)
    print(f"user {u}: SINR={sinr:8.2f}  rate={rates[u]:.2f}")
print("NOMA sum:", rates.sum().round(2))         

oma = (1 / 3) * np.log2(1 + P * g / N0)
print("OMA  sum:", oma.sum().round(2))          