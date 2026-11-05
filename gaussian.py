"""Exact laws and asymptotic error constants for diffusion samplers on Gaussian data.

Forward process:  dX = -X dt + sqrt(2) dW  (VP / OU),  X_0 ~ N(mu, Sigma).
Samplers (reverse time s = T - t, uniform step h, score at left endpoint):
  "sde": Euler–Maruyama on dY = [Y + 2 grad log p_{T-s}(Y)] ds + sqrt(2) dB
  "ode": Euler on the probability-flow ODE dY = [Y + grad log p_{T-s}(Y)] ds
Because Sigma_t = e^{-2t} Sigma + (1 - e^{-2t}) I shares Sigma's eigenvectors,
everything decouples into scalar problems indexed by eigenvalue lam.
"""
from __future__ import annotations

import numpy as np

SCHEMES = ("sde", "ode")


# ----------------------------------------------------------------------------
# Marginals
# ----------------------------------------------------------------------------
def sigma2(t, lam):
    """Marginal variance of the OU process at time t for data variance lam."""
    t = np.asarray(t, dtype=float)
    return lam * np.exp(-2.0 * t) - np.expm1(-2.0 * t)


def n_steps(T: float, h: float) -> int:
    N = int(round(T / h))
    if N < 1 or abs(N * h - T) > 1e-9 * max(1.0, T):
        raise ValueError(f"T/h must be a positive integer (T={T}, h={h})")
    return N


# ----------------------------------------------------------------------------
# Exact law of the discrete chain (diagonal / eigenbasis form)
# ----------------------------------------------------------------------------
def discrete_law(lam, mu, T: float, h: float, scheme: str, init: str = "prior"):
    """Return (mean, var) of Y_N per eigen-direction.

    init = "prior": Y_0 ~ N(0, 1);  init = "exact": Y_0 ~ p_T.
    """
    if scheme not in SCHEMES:
        raise ValueError(scheme)
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    mu = np.broadcast_to(np.asarray(mu, dtype=float), lam.shape).copy()
    N = n_steps(T, h)
    if init == "prior":
        m = np.zeros_like(lam)
        v = np.ones_like(lam)
    elif init == "exact":
        m = np.exp(-T) * mu
        v = sigma2(T, lam)
    else:
        raise ValueError(init)
    for k in range(N):
        t = T - k * h
        s2 = sigma2(t, lam)
        a = np.exp(-t)
        if scheme == "sde":
            b = 1.0 - 2.0 / s2
            c = 2.0 * a * mu / s2
            m = m + h * (b * m + c)
            v = (1.0 + h * b) ** 2 * v + 2.0 * h
        else:
            b = 1.0 - 1.0 / s2
            c = a * mu / s2
            m = m + h * (b * m + c)
            v = (1.0 + h * b) ** 2 * v
    return m, v


def discrete_law_full(Sigma, mu, T: float, h: float, scheme: str, init: str = "prior"):
    """Same chain, but with full matrices and NO eigendecomposition (independent check)."""
    Sigma = np.asarray(Sigma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    d = len(mu)
    I = np.eye(d)
    N = n_steps(T, h)
    if init == "prior":
        m, V = np.zeros(d), I.copy()
    else:
        m = np.exp(-T) * mu
        V = np.exp(-2 * T) * Sigma + (1 - np.exp(-2 * T)) * I
    k_score = 2.0 if scheme == "sde" else 1.0
    for k in range(N):
        t = T - k * h
        St = np.exp(-2 * t) * Sigma - np.expm1(-2 * t) * I
        P = np.linalg.inv(St)
        B = I - k_score * P                      # drift matrix
        c = k_score * P @ (np.exp(-t) * mu)
        A = I + h * B
        m = A @ m + h * c
        V = A @ V @ A.T + (2.0 * h * I if scheme == "sde" else 0.0)
    return m, V


# ----------------------------------------------------------------------------
# KL divergences
# ----------------------------------------------------------------------------
def _x_minus_log1p(x):
    """x - log(1+x), accurate for small |x|."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-3
    xs = x[small]
    out[small] = xs**2 / 2 - xs**3 / 3 + xs**4 / 4 - xs**5 / 5 + xs**6 / 6
    out[~small] = x[~small] - np.log1p(x[~small])
    return out


def kl_gauss_diag(m1, v1, m2, v2):
    """KL(N(m1, v1) || N(m2, v2)) summed over independent coordinates."""
    m1, v1, m2, v2 = (np.asarray(z, dtype=float) for z in (m1, v1, m2, v2))
    x = (v1 - v2) / v2
    return float(np.sum(0.5 * (_x_minus_log1p(x) + (m1 - m2) ** 2 / v2)))


def kl_gauss_full(m1, V1, m2, V2):
    d = len(m1)
    L = np.linalg.cholesky(V2)
    Li = np.linalg.inv(L)
    M = Li @ V1 @ Li.T
    dm = Li @ (m1 - m2)
    sign, logdet = np.linalg.slogdet(M)
    return 0.5 * (np.trace(M) - d - logdet + dm @ dm)


def kl_output(lam, mu, T, h, scheme, init="prior"):
    m, v = discrete_law(lam, mu, T, h, scheme, init)
    return kl_gauss_diag(m, v, mu, lam)


# ----------------------------------------------------------------------------
# Closed-form first-order error coefficients  (dm, dv) = h * (alpha, beta) + O(h^2)
# ----------------------------------------------------------------------------
def _F_sde(w):
    """F as a function of w = 1 + q (pass w directly: forming 1 + q from q = lam - 1
    loses all relative precision when lam -> 0)."""
    return -np.log(w) + 2.0 / w - 2.0 / w**2


def _ratio_log(lam, eT):
    """[ln(lam) - ln(1+qT)] / (lam - 1), with qT = (lam-1) e^{-2T}, eT = e^{-2T}; smooth at lam = 1."""
    eps = lam - 1.0
    qT = eps * eT
    u = np.full_like(eps, eT)
    out = np.empty_like(eps)
    small = np.abs(eps) < 1e-4
    # series:  [ln(1+e) - ln(1+e u)]/e = (1-u) - e(1-u^2)/2 + e^2(1-u^3)/3
    e, uu = eps[small], u[small]
    out[small] = (1 - uu) - e * (1 - uu**2) / 2 + e**2 * (1 - uu**3) / 3
    out[~small] = (np.log1p(eps[~small]) - np.log1p(qT[~small])) / eps[~small]
    return out


def first_order_coeffs(lam, mu, T=np.inf, scheme="sde"):
    """Return (alpha, beta): m_N - mu ~ h*alpha, v_N - lam ~ h*beta (exact init / T->inf)."""
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    mu = np.broadcast_to(np.asarray(mu, dtype=float), lam.shape)
    eT = 0.0 if np.isinf(T) else np.exp(-2.0 * T)
    eps = lam - 1.0
    qT = eps * eT
    if scheme == "sde":
        alpha = -(mu * lam / 4.0) * _ratio_log(lam, eT)
        beta = np.empty_like(lam)
        small = np.abs(eps) < 1e-4
        # series of [F(e) - F(e u)]/e with F(e) = e - 7/2 e^2 + 17/3 e^3 + ...
        e = eps[small]
        beta[small] = lam[small] ** 2 / 2 * (
            (1 - eT) - 3.5 * e * (1 - eT**2) + (17.0 / 3.0) * e**2 * (1 - eT**3)
        )
        ls = ~small
        beta[ls] = lam[ls] ** 2 / (2 * eps[ls]) * (_F_sde(lam[ls]) - _F_sde(1.0 + qT[ls]))
        return alpha, beta
    elif scheme == "ode":
        # alpha = -(mu sqrt(lam)/2) * int_{x0}^{1} dx / sqrt(1 + eps x^2),  x0 = e^{-T}
        x0 = np.sqrt(eT)
        alpha = -(mu * np.sqrt(lam) / 2.0) * (_A(eps, 1.0) - _A(eps, x0))
        # beta = 2 sqrt(lam) * dr,  dr = -(h sqrt(lam)/4) [ln(lam/(1+qT)) + 1/(1+qT) - 1/lam]
        bracket = (np.log(lam) - np.log1p(qT)) + 1.0 / (1.0 + qT) - 1.0 / lam
        beta = -(lam / 2.0) * bracket
        return alpha, beta
    raise ValueError(scheme)


def _A(eps, x):
    """int_0^x dy / sqrt(1 + eps y^2), smooth in eps."""
    eps = np.asarray(eps, dtype=float)
    x = np.broadcast_to(np.asarray(x, dtype=float), eps.shape)
    out = np.empty_like(eps)
    pos, neg = eps > 1e-8, eps < -1e-8
    mid = ~(pos | neg)
    r = np.sqrt(eps[pos])
    out[pos] = np.arcsinh(r * x[pos]) / r
    r = np.sqrt(-eps[neg])
    out[neg] = np.arcsin(r * x[neg]) / r
    xm, em = x[mid], eps[mid]
    out[mid] = xm - em * xm**3 / 6 + 3 * em**2 * xm**5 / 40
    return out


def kl_constant(lam, mu, T=np.inf, scheme="sde"):
    """C such that KL(Law(Y_N) || data) = h^2 * C + O(h^3) (exact init)."""
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    alpha, beta = first_order_coeffs(lam, mu, T, scheme)
    return float(np.sum(alpha**2 / (2 * lam) + beta**2 / (4 * lam**2)))


# ----------------------------------------------------------------------------
# Initialisation (prior-mismatch) error of the EXACT continuous dynamics
# ----------------------------------------------------------------------------
def init_error_law(lam, mu, T, scheme):
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    mu = np.broadcast_to(np.asarray(mu, dtype=float), lam.shape)
    sT = sigma2(T, lam)
    if scheme == "sde":
        dm = -(lam * np.exp(-2 * T) / sT) * mu
        dv = lam**2 * np.exp(-2 * T) * (1 - sT) / sT**2
    else:
        dm = -(np.sqrt(lam) * np.exp(-T) / np.sqrt(sT)) * mu
        # r_end = sqrt(lam)/sigma_T * r_0 with r_0 = 1  -> v_end = lam / sigma_T^2
        dv = lam / sT - lam
    return mu + dm, lam + dv


def kl_prior_to_pT(lam, mu, T):
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    return kl_gauss_diag(0.0 * lam, np.ones_like(lam), np.exp(-T) * np.asarray(mu) + 0 * lam, sigma2(T, lam))
