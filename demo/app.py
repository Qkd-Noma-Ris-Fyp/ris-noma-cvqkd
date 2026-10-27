import numpy as np, streamlit as st, matplotlib.pyplot as plt
import model as M

st.set_page_config(page_title="RIS-NOMA-QKD Demo", page_icon="📡", layout="wide")
PAL = ["#60a5fa", "#fbbf24", "#34d399", "#f87171", "#a78bfa"]
plt.rcParams.update({"figure.facecolor": "none", "axes.facecolor": "none", "savefig.transparent": True,
    "text.color": "#94a3b8", "axes.labelcolor": "#94a3b8", "xtick.color": "#94a3b8", "ytick.color": "#94a3b8",
    "axes.edgecolor": "#64748b", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#64748b", "grid.alpha": .25, "axes.prop_cycle": plt.cycler(color=PAL), "font.size": 9})

st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1320px}
.hero{background:linear-gradient(120deg,#1e3a8a,#6d28d9);padding:22px 28px;border-radius:16px;color:#fff;margin-bottom:12px}
.hero h1{margin:0;font-size:26px;color:#fff}.hero p{margin:6px 0 0;opacity:.88;font-size:14px}
.flow{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
.flow span{background:rgba(128,128,128,.14);border:1px solid rgba(128,128,128,.3);padding:5px 12px;border-radius:99px;font-size:12px}
div[data-testid="stMetric"]{background:rgba(128,128,128,.10);border:1px solid rgba(128,128,128,.28);border-radius:14px;padding:14px 18px}
div[data-testid="stMetricValue"]{font-size:28px}
</style>
<div class="hero"><h1>📡 RIS-Assisted Quantum-Secure Wireless Communication for 6G</h1>
<p>Joint RIS phase optimisation · NOMA power allocation · CV-QKD &amp; BB84 key analysis — live simulator</p>
<p style="font-size:12px;opacity:.75">BECE497J Project I · Review 3 · School of Electronics Engineering, VIT Vellore</p></div>
<div class="flow"><span>① Channel</span><span>→</span><span>② RIS optimiser</span><span>→</span><span>③ NOMA + SIC</span><span>→</span><span>④ CV-QKD</span><span>+</span><span>⑤ BB84</span></div>
""", unsafe_allow_html=True)

sb = st.sidebar
sb.header("⚙️ Controls")
sb.subheader("Wireless link")
N = sb.slider("RIS elements N", 4, 256, 128, 4)
K = sb.slider("NOMA users K", 2, 4, 3)
P_dbm = sb.slider("Tx power (dBm)", 10, 40, 30)
alpha = sb.slider("Power exponent α", 0.2, 1.0, 0.6, 0.1)
sb.subheader("Quantum link")
err = sb.slider("BB84 channel error (%)", 0.0, 15.0, 1.5, 0.5) / 100
pulses = sb.slider("BB84 pulses", 1000, 20000, 5000, 1000)
G_REF = float(sb.selectbox("CV-QKD coupling G_REF", [1e-8, 3e-8, 1e-7]))
sb.subheader("Scenario")
ris_on = sb.toggle("RIS enabled (optimised phases)", True)
eve = sb.toggle("Eavesdropper (intercept-resend)", False)
seed = int(sb.number_input("Seed", 1, 999, 42))
P = 10 ** ((P_dbm - 30) / 10)

A, d = M.gen_channel(N, K, seed)
theta, g_opt, hist = M.optimise(A, d)
g0 = M.gains(A, d); g = g_opt if ris_on else g0
rows, R_noma = M.noma(g, P, alpha); R_oma0 = M.oma(g0, P)
T = float(np.mean(np.minimum(g / G_REF, 1))); skr = float(np.mean(M.cvqkd_skr(g, G_REF)))
et = M.eta_of_N(N if ris_on else 0); bb = M.bb84(pulses, et, err, eve, seed)

c = st.columns(4)
c[0].metric("RIS + NOMA sum-rate", f"{R_noma:.2f}", "bit/s/Hz", delta_color="off")
c[1].metric("Gain vs No-RIS OMA", f"{R_noma / R_oma0:.1f}×", f"baseline {R_oma0:.2f} bit/s/Hz", delta_color="off")
c[2].metric("CV-QKD key rate", f"{skr:.2f}", f"bit/symbol · T = {T:.2f}", delta_color="off")
c[3].metric("BB84 QBER", f"{bb['qber']*100:.1f}%", "SECURE" if bb["secure"] else "ABORTED",
            delta_color="normal" if bb["secure"] else "inverse")
if bb["secure"]: st.success(f"🔐 Channel secure: {bb['secret']} secret bits established (secret fraction r = {bb['r']:.2f}).")
else: st.error("🚨 QBER above the 11% threshold: eavesdropper or excess noise detected, key aborted.")

@st.cache_data
def sweep(K, P, alpha, seed):
    out = [[] for _ in range(4)]; sk = []
    for n in M.NS:
        acc = np.zeros(4); gl = []
        for m in range(6):
            A_, d_ = M.gen_channel(n, K, seed * 100 + m * 7 + n); _, go, _ = M.optimise(A_, d_); g0_ = M.gains(A_, d_)
            acc += [M.oma(g0_, P), M.noma(g0_, P, alpha)[1], M.noma(M.gains(A_, d_, M.heuristic(A_, d_)), P, alpha)[1], M.noma(go, P, alpha)[1]]
            gl += list(go)
        for o, v in zip(out, acc / 6): o.append(v)
        sk.append([float(np.mean(M.cvqkd_skr(gl, G))) for G in (1e-8, 3e-8, 1e-7)])
    return out, np.array(sk)

@st.cache_data
def validate():
    A_, d_ = M.gen_channel(32, 3, 1); _, g_, h = M.optimise(A_, d_)
    q = lambda e, ev, n=5000: np.mean([M.bb84(n, .64, e, ev, s)["qber"] for s in range(30)])
    _, sk_ = sweep(3, 1.0, .6, 42)
    ris = np.mean([M.noma(M.optimise(*M.gen_channel(64, 3, s))[1], 1.0)[1] for s in range(6)])
    base = np.mean([M.oma(M.gains(*M.gen_channel(64, 3, s)), 1.0) for s in range(6)])
    chk = [("Optimiser is monotone", "objective never decreases", f"{h[-1]:.2f}× start", bool(np.all(np.diff(h) >= -1e-9))),
      ("BB84 QBER, no Eve", "≈ channel error 1.5%", f"{q(.015, False)*100:.1f}%", abs(q(.015, False) - .015) < .005),
      ("BB84 QBER, Eve present", "≈ 25%", f"{q(0, True)*100:.1f}%", abs(q(0, True) - .25) < .03),
      ("BB84 Eve + 15% noise", "0.25 + 0.5e = 32.5%", f"{q(.15, True)*100:.1f}%", abs(q(.15, True) - .325) < .03),
      ("Abort above 11% QBER", "key = 0 bits", f"{M.bb84(5000, .64, .20, False, 3)['secret']} bits", M.bb84(5000, .64, .20, False, 3)["secret"] == 0),
      ("Key rate grows with N", "rises with N (±0.05 MC noise)", f"{sk_[0,0]:.2f} → {sk_[-1,0]:.2f}", bool(np.all(np.diff(sk_[:, 0]) >= -0.05) and sk_[-1, 0] > sk_[0, 0])),
      ("RIS beats no-RIS OMA", "RIS+NOMA > baseline at N=64", f"{ris/base:.1f}×", ris > base)]
    return chk

out, sk = sweep(K, P, alpha, seed)
t1, t2, t3, t4, t5 = st.tabs(["📶 RIS + NOMA", "🧪 CV-QKD", "🔐 BB84", "✅ Validation", "📖 How it works"])

with t1:
    a, b = st.columns(2)
    f, ax = plt.subplots(figsize=(5.4, 3.5))
    for o, lab in zip(out, ["No-RIS OMA", "No-RIS NOMA", "RIS heuristic", "RIS optimised"]): ax.plot(M.NS, o, "o-", label=lab, lw=2, ms=4)
    ax.axvline(N, c="#94a3b8", ls=":"); ax.set_xscale("log", base=2); ax.set_xticks(M.NS); ax.set_xticklabels(M.NS)
    ax.set_xlabel("RIS elements N"); ax.set_ylabel("sum-rate (bit/s/Hz)"); ax.legend(frameon=False, fontsize=8)
    a.markdown("**Sum-rate vs RIS size**"); a.pyplot(f)
    f, ax = plt.subplots(figsize=(5.4, 3.5)); ax.plot(hist, "o-", lw=2, ms=4, c=PAL[2]); ax.set_xlabel("sweep"); ax.set_ylabel("objective / start")
    b.markdown(f"**Optimiser convergence** (N = {N}, ends at {hist[-1]:.2f}× start)"); b.pyplot(f)
    a, b = st.columns([3, 2])
    a.markdown("**NOMA users** (weakest gets most power)")
    a.dataframe([{"User": r["user"], "Gain (dB)": round(r["gain_db"], 1), "Power share": f"{r['power']*100:.1f}%",
                  "SINR (dB)": round(r["sinr_db"], 1), "Rate (bit/s/Hz)": round(r["rate"], 2)} for r in rows], hide_index=True, width="stretch")
    cols = int(np.ceil(np.sqrt(N * 1.6))); rw = int(np.ceil(N / cols)); ph = np.full(cols * rw, np.nan); ph[:N] = np.angle(theta) % (2 * np.pi)
    f, ax = plt.subplots(figsize=(4.2, 2.6)); ax.imshow(ph.reshape(rw, cols), cmap="hsv", vmin=0, vmax=2 * np.pi); ax.axis("off")
    b.markdown("**RIS phase map** (colour = φₙ)"); b.pyplot(f)

with t2:
    f, ax = plt.subplots(figsize=(7, 3.6))
    for i, G in enumerate((1e-8, 3e-8, 1e-7)): ax.plot(M.NS, sk[:, i], "o-", lw=2, ms=4, label=f"G_REF = {G:.0e}")
    ax.set_xscale("log", base=2); ax.set_xticks(M.NS); ax.set_xticklabels(M.NS); ax.set_xlabel("RIS elements N"); ax.set_ylabel("key rate (bit/symbol)")
    ax.legend(frameon=False); st.pyplot(f)
    st.info("Key rate rises steeply only as transmittance T → 1. T = min(|g|²/G_REF, 1) is a **placeholder coupling**: it shows sensitivity, not a real link. G_REF values here are scaled to this simulator\u2019s gains.")

with t3:
    a, b = st.columns(2); E = np.arange(0, 16)
    f, ax = plt.subplots(figsize=(5.4, 3.5))
    for ev, col, l in [(False, PAL[2], "No Eve"), (True, PAL[3], "With Eve")]:
        ax.plot(E, [np.mean([M.bb84(5000, et, e / 100, ev, s)["qber"] for s in range(5)]) * 100 for e in E], "o-", c=col, lw=2, ms=4, label=l)
    ax.axhline(11, ls="--", c="#e2e8f0", lw=1); ax.text(.3, 11.8, "11% abort threshold", fontsize=8)
    ax.set_xlabel("channel error (%)"); ax.set_ylabel("QBER (%)"); ax.legend(frameon=False); a.markdown("**QBER vs channel noise**"); a.pyplot(f)
    f, ax = plt.subplots(figsize=(5.4, 3.5)); v = [bb["raw"], bb["sifted"], bb["after_test"], bb["secret"]]
    ax.bar(["Raw", "Sifted", "After QBER\ntest", "Secret\nkey"], v, color=[PAL[0], PAL[4], PAL[1], PAL[2]])
    for i, x in enumerate(v): ax.text(i, x, str(x), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("bits"); b.markdown(f"**Key pipeline** (η = {et:.2f}, r = {bb['r']:.2f})"); b.pyplot(f)

with t4:
    st.markdown("Checks run live on the model, mirroring slide 11.")
    chk = validate()
    st.dataframe([{"Check": x[0], "Expected": x[1], "Observed": x[2], "Result": "✅ Pass" if x[3] else "❌ Fail"} for x in chk], hide_index=True, width="stretch")
    st.metric("Checks passed", f"{sum(x[3] for x in chk)} / {len(chk)}")

with t5:
    st.markdown("### One chain from radio channel to secret key")
    with st.expander("① Channel", True): st.write("BS → RIS → user cascaded Rician channel plus a weak Rayleigh direct link, with log-distance path loss (30 dB at 1 m; exponent 2.2 via RIS, 3.5 direct).")
    with st.expander("② RIS optimiser", True): st.latex(r"\varphi_n^*=-\arg\Big(\sum_k w_k\,a_{k,n}B_k^*\Big)"); st.write("Each element has a closed-form best phase given the others, so coordinate ascent is monotone (reaches a local optimum).")
    with st.expander("③ NOMA + SIC", True): st.latex(r"p_k\propto|g_k|^{-2\alpha},\quad \mathrm{SINR}_k=\frac{p_kg_k}{g_k\sum_{j\in S_k}p_j+\sigma^2}"); st.write("Weaker users get more power; stronger users cancel their signals first (SIC). OMA is the baseline.")
    with st.expander("④ CV-QKD", True): st.latex(r"\mathrm{SKR}=\beta I(A{:}B)-\chi(B{:}E)"); st.write("Better RIS gain → higher transmittance T → higher key rate. T and χ use placeholder / simplified forms.")
    with st.expander("⑤ BB84", True): st.latex(r"r=\max\big(0,\,1-2h(\mathrm{QBER})\big)"); st.write("Photon loss (η grows with N), sifting, 20% QBER sample. Eve's intercept-resend pushes QBER to ≈25% → abort above 11%.")
