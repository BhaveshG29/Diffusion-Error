"""Numerically exact sampler error for non-Gaussian 1D targets (Gaussian mixtures).

* SDE / Euler–Maruyama : the chain's density is propagated exactly up to spectrally
  accurate quadrature (Nyström discretisation of the Chapman–Kolmogorov equation).
* ODE / Euler          : the chain is a deterministic map; push the initial density
  along characteristics, carrying the Jacobian, and integrate KL in *initial* coords.
* First-order theory for the ODE (pre-registered P9): variational equation for the
  global error e(x), then KL ~ (h^2/2) int p_T(x) (e'(x)/Phi'(x) + e(x) s_0(Phi(x)))^2 dx.
"""
from __future__ import annotations

import numpy as np


def logsumexp(a, axis=-1, keepdims=False):
    """Complex-analytic logsumexp (shift by max of the real part), so the whole
    score pipeline supports complex-step differentiation."""
    a = np.asarray(a)
    amax = np.max(a.real, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True)) + amax
    return out if keepdims else np.squeeze(out, axis=axis)

from gaussian import sigma2, n_steps


class Mixture1D:
    def __init__(self, w, m, v):
        self.w = np.asarray(w, float) / np.sum(w)
        self.m = np.asarray(m, float)
        self.v = np.asarray(v, float)

    # -- marginal p_t ---------------------------------------------------------
    def _params(self, t):
        return np.exp(-t) * self.m, sigma2(t, self.v)

    def _logcomp(self, t, y):
        M, V = self._params(t)
        y = np.asarray(y)[..., None]
        return np.log(self.w) - 0.5 * np.log(2 * np.pi * V) - 0.5 * (y - M) ** 2 / V, y, M, V

    def logpdf(self, t, y):
        lc, *_ = self._logcomp(t, y)
        return logsumexp(lc, axis=-1)

    def pdf(self, t, y):
        return np.exp(self.logpdf(t, y))

    def score_derivs(self, t, y):
        """Return S, S', S'' (derivatives in y of log p_t) via posterior moments."""
        lc, y, M, V = self._logcomp(t, y)
        r = np.exp(lc - logsumexp(lc, axis=-1, keepdims=True))
        u = -(y - M) / V
        up = -1.0 / V + 0 * u
        E = lambda f: np.sum(r * f, axis=-1)
        S = E(u)
        cov = lambda f: E(f * u) - E(f) * S
        Sp = E(up) + E(u * u) - S**2
        Spp = cov(up) + 2 * E(u * up) + cov(u * u) - 2 * S * Sp
        return S, Sp, Spp

    def dt_score(self, t, y):
        """d/dt of the score, from the Fokker–Planck equation dp/dt = d(yp)/dy + d2p/dy2."""
        S, Sp, Spp = self.score_derivs(t, y)
        y = np.asarray(y)
        return S + y * Sp + Spp + 2 * S * Sp


# ----------------------------------------------------------------------------
# SDE: Nyström propagation of the EM chain density
# ----------------------------------------------------------------------------
def em_density(mix: Mixture1D, T, h, L=10.0, pts_per_std=8, nsig=9.0, init="exact"):
    """Return grid x and the density of Y_N on x (EM chain, uniform step h)."""
    s = np.sqrt(2 * h)
    dx = s / pts_per_std
    M = int(np.ceil(2 * L / dx)) + 1
    x = np.linspace(-L, L, M)
    dx = x[1] - x[0]
    rho = mix.pdf(T, x) if init == "exact" else np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    W = int(np.ceil(2 * nsig * s / dx)) + 2
    offs = np.arange(W)
    for k in range(n_steps(T, h)):
        t = T - k * h
        S, _, _ = mix.score_derivs(t, x)
        z = x + h * (x + 2 * S)
        j0 = np.floor((z - nsig * s - x[0]) / dx).astype(int)
        J = j0[:, None] + offs[None, :]
        valid = (J >= 0) & (J < M)
        Jc = np.clip(J, 0, M - 1)
        kern = np.exp(-0.5 * ((x[Jc] - z[:, None]) / s) ** 2) / (np.sqrt(2 * np.pi) * s)
        wts = (rho * dx)[:, None] * kern * valid
        rho = np.bincount(Jc.ravel(), weights=wts.ravel(), minlength=M)
    return x, rho


def kl_grid(x, rho, logp):
    dx = x[1] - x[0]
    pos = rho > 0
    # generalised KL (robust to O(tiny) mass loss at the boundary)
    return float(np.sum(dx * rho[pos] * (np.log(rho[pos]) - logp[pos])) - np.sum(dx * rho) + 1.0)


def kl_sde(mix, T, h, **kw):
    x, rho = em_density(mix, T, h, **kw)
    return kl_grid(x, rho, mix.logpdf(0.0, x))


# ----------------------------------------------------------------------------
# ODE: characteristics + Jacobian
# ----------------------------------------------------------------------------
def _init_grid(mix, T, L, dx):
    x = np.arange(-L, L + dx / 2, dx)
    return x, mix.logpdf(T, x)


def kl_ode(mix: Mixture1D, T, h, L=None, dx=2e-3):
    """KL(Law(Y_N) || p_0) for Euler on the PF-ODE, started from the exact p_T."""
    if L is None:
        L = 9.0 * np.sqrt(max(sigma2(T, mix.v).max(), 1.0)) + np.exp(-T) * np.abs(mix.m).max()
    x, logq = _init_grid(mix, T, L, dx)
    Y, logJ = x.copy(), np.zeros_like(x)
    for k in range(n_steps(T, h)):
        t = T - k * h
        S, Sp, _ = mix.score_derivs(t, Y)
        mult = 1 + h * (1 + Sp)
        if np.any(mult <= 0):
            raise FloatingPointError("Euler map not monotone; h too large")
        logJ += np.log(mult)
        Y = Y + h * (Y + S)
    q = np.exp(logq)
    return float(np.sum(dx * q * (logq - logJ - mix.logpdf(0.0, Y))))


def _flow_rhs(mix, T, s, state):
    Y, logJ, e = state
    t = T - s
    S, Sp, _ = mix.score_derivs(t, Y)
    f = Y + S
    a = 1 + Sp
    Yddot = -mix.dt_score(t, Y) + a * f        # d^2 Y / ds^2 along the exact flow
    return np.array([f, a, a * e - 0.5 * Yddot])


def ode_first_order_constant(mix: Mixture1D, T, n_ref=1000, L=None, dx=5e-4, return_parts=False):
    """Pre-registered P9: KL = h^2 * C + O(h^3) for Euler on the PF-ODE (exact init).

    de/dx is obtained by complex-step differentiation through the whole RK4 solve
    (x -> x + i*1e-30), which is exact to machine precision. A 4th-order finite
    difference was used first and was badly wrong near the inter-mode "watershed",
    where the flow map has Phi' ~ 1e2 (see AUDIT.md #7).
    """
    if L is None:
        L = 9.0 * np.sqrt(max(sigma2(T, mix.v).max(), 1.0)) + np.exp(-T) * np.abs(mix.m).max()
    x, logq = _init_grid(mix, T, L, dx)
    delta = 1e-30
    xc = x + 1j * delta
    state = np.array([xc, np.zeros_like(xc), np.zeros_like(xc)])
    H = T / n_ref
    for k in range(n_ref):                      # classical RK4 on (Y, log J, e)
        s = k * H
        k1 = _flow_rhs(mix, T, s, state)
        k2 = _flow_rhs(mix, T, s + H / 2, state + H / 2 * k1)
        k3 = _flow_rhs(mix, T, s + H / 2, state + H / 2 * k2)
        k4 = _flow_rhs(mix, T, s + H, state + H * k3)
        state = state + H / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    Y, logJ, e = state.real
    de = state[2].imag / delta
    dPhi = state[0].imag / delta               # independent of logJ: consistency check
    S0, _, _ = mix.score_derivs(0.0, Y)
    integrand = np.exp(logq) * (de / dPhi + e * S0) ** 2
    C = 0.5 * np.sum(integrand) * dx
    if return_parts:
        return C, dict(x=x, Y=Y, logJ=logJ, e=e, de=de, dPhi=dPhi, logq=logq, integrand=integrand)
    return C
