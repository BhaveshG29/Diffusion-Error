"""Regenerates every number and figure in README.md.  python3 experiments.py"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gaussian as g
import mixture as mx

OUT = "results"
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#b5b4ae",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "lines.linewidth": 2,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
})
res = {}

# --- Fig 1: the constants vs eigenvalue ---------------------------------------
lam = np.logspace(-3, 2, 400)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for sch, col in (("sde", BLUE), ("ode", ORANGE)):
    a, b = g.first_order_coeffs(lam, 1.0, np.inf, sch)
    ax[0].loglog(lam, b**2 / (4 * lam**2), color=col, label=f"{sch.upper()}: variance part")
    ax[0].loglog(lam, a**2 / (2 * lam), color=col, ls="--", label=f"{sch.upper()}: mean part ($\\mu=1$)")
ax[0].loglog(lam[lam < 0.1], 0.25 / lam[lam < 0.1] ** 2, color=GRAY, ls=":", lw=1.2)
ax[0].loglog(lam[lam < 0.1], (1 / 16) / lam[lam < 0.1] ** 2, color=GRAY, ls=":", lw=1.2)
ax[0].annotate(r"$1/(4\lambda^2)$", (3e-3, 0.25 / 9e-6), xytext=(6e-3, 0.25 / 9e-6 * 8), color=GRAY, fontsize=9)
ax[0].annotate(r"$1/(16\lambda^2)$", (3e-3, 0.0625 / 9e-6), xytext=(1.2e-3, 0.0625 / 9e-6 / 40), color=GRAY, fontsize=9)
ax[0].set_xlabel(r"eigenvalue $\lambda$ of $\Sigma$")
ax[0].set_ylabel(r"$C(\lambda)$ in KL $= h^2 C + O(h^3)$")
ax[0].set_title("Leading KL constant per eigen-direction", loc="left")
ax[0].legend(fontsize=8, frameon=False)
Cs = np.array([g.kl_constant(l, 0.0, np.inf, "sde") for l in lam])
Co = np.array([g.kl_constant(l, 0.0, np.inf, "ode") for l in lam])
ax[1].semilogx(lam, Cs / Co, color=BLUE)
ax[1].axhline(4, color=GRAY, ls=":", lw=1.2)
ax[1].set_ylim(0, 6)
ax[1].text(1.15, 5.4, r"$C_{\rm ODE}=0$ at $\lambda=1$", fontsize=8, color=GRAY)
ax[1].text(2.2, 0.25, r"$C_{\rm SDE}=0$ at $\lambda^*\approx1.597$", fontsize=8, color=GRAY)
ax[1].set_xlabel(r"$\lambda$")
ax[1].set_ylabel(r"$C_{\rm SDE}/C_{\rm ODE}$  (variance part, $\mu=0$)")
ax[1].set_title("EM-SDE vs Euler-ODE (no mean)", loc="left")
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_constants.png", dpi=160)

# --- Table: Richardson check of the Gaussian constants ------------------------
rows = []
for l, m in [(0.01, 1.0), (0.1, 1.0), (1.0, 1.0), (10.0, 1.0)]:
    for sch in g.SCHEMES:
        C = g.kl_constant(l, m, 10.0, sch)
        h = min(1e-3, l / 40)
        h = 10.0 / round(10.0 / h)
        r1 = g.kl_output(l, m, 10.0, h, sch, "exact") / h**2
        r2 = g.kl_output(l, m, 10.0, h / 2, sch, "exact") / (h / 2) ** 2
        rows.append(dict(lam=l, mu=m, scheme=sch, C_pred=C, richardson=2 * r2 - r1,
                         rel_err=abs(2 * r2 - r1 - C) / C, h=h))
res["gaussian_table"] = rows

# --- Fig 2: init error vs T -----------------------------------------------------
Ts = np.linspace(0.5, 8, 60)
lam1, mu1 = np.array([0.3]), np.array([1.0])
kl_s = [g.kl_gauss_diag(*g.init_error_law(lam1, mu1, T, "sde"), mu1, lam1) for T in Ts]
kl_o = [g.kl_gauss_diag(*g.init_error_law(lam1, mu1, T, "ode"), mu1, lam1) for T in Ts]
kl_b = [g.kl_prior_to_pT(lam1, mu1, T) for T in Ts]
fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.semilogy(Ts, kl_b, color=GRAY, lw=4, alpha=0.5, label=r"data-processing bound KL$(\gamma\,\|\,p_T)$")
ax.semilogy(Ts, kl_o, color=ORANGE, label="PF-ODE (exact flow)")
ax.semilogy(Ts, kl_s, color=BLUE, label="reverse SDE (exact flow)")
ax.semilogy(Ts, 0.5 * lam1[0] * np.exp(-4 * Ts), color=BLUE, ls=":", lw=1.2,
            label=r"$\frac{\lambda\mu^2}{2}e^{-4T}$")
ax.set_xlabel("horizon T")
ax.set_ylabel("KL(output || data), h → 0")
ax.set_title(r"Prior-mismatch error ($\lambda=0.3,\ \mu=1$)", loc="left")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_init_error.png", dpi=160)

# --- Fig 3: mixture convergence ---------------------------------------------------
MIX = mx.Mixture1D([0.3, 0.7], [-2.0, 1.5], [0.15, 0.4])
T = 8.0
klode = {float(k): v for k, v in json.load(open(f"{OUT}/klode_mix.json")).items()}
klsde = {float(k): v for k, v in json.load(open(f"{OUT}/klsde_mix.json")).items()}
Cpred = json.load(open(f"{OUT}/C_mix_dx2.5e-4.json"))["C"]
res["mixture"] = dict(klode_over_h2=klode, klsde_over_h2=klsde, C_ode_pred=Cpred)
hs = np.array(sorted(klode))
hs2 = np.array(sorted(klsde))
r2 = np.array([klsde[h] for h in hs2])
A = np.vander(hs2, 5, increasing=True)
Csde = np.linalg.lstsq(A, r2, rcond=None)[0][0]
res["mixture"]["C_sde_extrapolated_deg4"] = Csde
fig, ax = plt.subplots(2, 1, figsize=(5.6, 5.2), sharex=True)
ax[0].semilogx(hs, [klode[h] for h in hs], "o-", color=ORANGE, ms=6, label="exact KL/$h^2$ (characteristics)")
ax[0].axhline(Cpred, color=GRAY, ls="--", lw=1.2, label=f"P9 prediction C = {Cpred:.4f}")
ax[0].set_ylabel(r"KL/$h^2$")
ax[0].set_title("Euler PF-ODE, bimodal target", loc="left")
ax[0].legend(fontsize=8, frameon=False)
ax[1].semilogx(hs2, r2, "s-", color=BLUE, ms=6, label="exact KL/$h^2$ (Nyström density)")
ax[1].axhline(Csde, color=GRAY, ls="--", lw=1.2, label=f"degree-4 extrapolation C = {Csde:.4f}")
ax[1].set_ylabel(r"KL/$h^2$")
ax[1].set_xlabel("step h")
ax[1].set_title("Euler–Maruyama SDE, same target (non-monotone!)", loc="left")
ax[1].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_mixture_convergence.png", dpi=160)

# --- Fig 4: where the ODE error lives -------------------------------------------
C, P = mx.ode_first_order_constant(MIX, T, n_ref=1000, dx=2.5e-4, return_parts=True)
y = P["Y"]
dens_y = 0.5 * P["integrand"] / P["dPhi"]    # contribution to C per unit y
share = np.sum((0.5 * P["integrand"] * 2.5e-4)[(y > -1.3) & (y < 0.0)]) / C
res["mixture"]["share_of_C_in_gap"] = share
fig, ax = plt.subplots(2, 1, figsize=(5.6, 4.6), sharex=True)
ax[0].plot(y, MIX.pdf(0.0, y), color=GRAY, lw=1.5)
ax[0].set_ylabel(r"data $p_0(y)$")
ax[0].set_title(f"{100*share:.0f}% of the ODE constant comes from the gap y in (-1.3, 0)", loc="left")
ax[1].plot(y, dens_y, color=ORANGE)
ax[1].axvspan(-1.3, 0.0, color=YELLOW, alpha=0.15, lw=0)
ax[1].set_ylabel("dC / dy")
ax[1].set_xlim(-4, 4)
ax[1].set_xlabel("y (output space)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_where_error_lives.png", dpi=160)

json.dump(res, open(f"{OUT}/summary.json", "w"), indent=2, default=float)
print(json.dumps(res, indent=2, default=float))
