import numpy as np
import pytest
from scipy import integrate

import gaussian as g

LAMS = np.array([0.05, 0.3, 0.9999, 1.0, 1.00007, 3.0, 20.0])
MUS = np.array([1.0, -0.7, 0.3, 0.5, -1.2, 2.0, 1.0])


# --- marginals --------------------------------------------------------------
def test_sigma2_limits():
    assert np.allclose(g.sigma2(0.0, LAMS), LAMS)
    assert np.allclose(g.sigma2(50.0, LAMS), 1.0)


def test_identity_P1_b2_plus_bdot_equals_one():
    """P1: for b = 1 - 2/sigma^2, b^2 + db/ds = 1 for every lam and t."""
    t = np.linspace(0.01, 5, 200)[:, None]
    lam = LAMS[None, :]
    b = lambda tt: 1 - 2 / g.sigma2(tt, lam)
    d = 1e-6
    bdot_s = -(b(t + d) - b(t - d)) / (2 * d)       # ds = -dt
    assert np.allclose(b(t) ** 2 + bdot_s, 1.0, atol=1e-7)


# --- exact recursion vs. independent implementations -----------------------
def test_recursion_matches_monte_carlo():
    """Simulate the actual EM / Euler chain with particles; compare mean & variance."""
    rng = np.random.default_rng(0)
    lam, mu, T, h = np.array([0.4, 2.5]), np.array([1.0, -0.5]), 3.0, 0.05
    n = 400_000
    for scheme in g.SCHEMES:
        Y = rng.standard_normal((n, 2))
        for k in range(g.n_steps(T, h)):
            t = T - k * h
            s2 = g.sigma2(t, lam)
            score = -(Y - np.exp(-t) * mu) / s2
            if scheme == "sde":
                Y = Y + h * (Y + 2 * score) + np.sqrt(2 * h) * rng.standard_normal(Y.shape)
            else:
                Y = Y + h * (Y + score)
        m, v = g.discrete_law(lam, mu, T, h, scheme)
        se_m = np.sqrt(v / n)
        se_v = v * np.sqrt(2 / n)
        assert np.all(np.abs(Y.mean(0) - m) < 5 * se_m + 1e-12), scheme
        assert np.all(np.abs(Y.var(0) - v) < 5 * se_v + 1e-12), scheme


@pytest.mark.parametrize("scheme", g.SCHEMES)
def test_full_matrix_matches_eigen_decomposition(scheme):
    """Random SPD Sigma, rotated mean: full-matrix chain == decoupled chain."""
    rng = np.random.default_rng(1)
    d = 5
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    lam = np.array([0.1, 0.5, 1.0, 2.0, 7.0])
    Sigma = Q @ np.diag(lam) @ Q.T
    mu = rng.standard_normal(d)
    T, h = 4.0, 0.02
    mF, VF = g.discrete_law_full(Sigma, mu, T, h, scheme)
    mD, vD = g.discrete_law(lam, Q.T @ mu, T, h, scheme)
    assert np.allclose(Q.T @ mF, mD, atol=1e-12)
    assert np.allclose(Q.T @ VF @ Q, np.diag(vD), atol=1e-12)
    klF = g.kl_gauss_full(mF, VF, mu, Sigma)
    klD = g.kl_gauss_diag(mD, vD, Q.T @ mu, lam)
    assert np.isclose(klF, klD, rtol=1e-9)


def test_lambda_one_fixed_point_by_hand():
    """lam=1: v_{k+1} = (1-h)^2 v + 2h has fixed point 2/(2-h); init v=1 is exact p_T."""
    for h in [0.1, 0.01]:
        _, v = g.discrete_law(1.0, 0.0, 40.0, h, "sde")
        assert np.isclose(v[0], 2 / (2 - h), rtol=1e-12)


def test_kl_small_x_branch_continuity():
    for x in [9.99e-4, 1.001e-3]:
        v = 1 + x
        ref = 0.5 * (x - np.log1p(x))
        assert np.isclose(g.kl_gauss_diag(0, v, 0, 1.0), ref, rtol=1e-10)


# --- closed forms vs. their defining integrals (scipy.quad) ----------------
@pytest.mark.parametrize("T", [1.5, 6.0, np.inf])
def test_closed_forms_match_defining_integrals(T):
    Tq = 60.0 if np.isinf(T) else T
    for lam, mu in zip(LAMS, MUS):
        s2 = lambda t: g.sigma2(t, lam)
        # SDE variance: h * int_0^T Phi * g dt,  Phi = lam^2 e^{-2t}/s2^2,
        # g = -(b^2 + b') s2 - 2b  with b = 1 - 2/s2 (b' computed numerically, NOT via P1)
        def bfun(t):
            return 1 - 2 / (lam * np.exp(-2 * t) + 1 - np.exp(-2 * t))

        def gfun(t):
            # complex-step derivative: exact to machine precision, valid at t = 0
            d = 1e-30
            bdot = -np.imag(bfun(t + 1j * d)) / d
            return -(bfun(t) ** 2 + bdot) * s2(t) - 2 * bfun(t)

        beta_sde = integrate.quad(lambda t: lam**2 * np.exp(-2 * t) / s2(t) ** 2 * gfun(t),
                                  0.0, Tq, limit=400, epsabs=1e-12)[0]
        alpha_sde = integrate.quad(lambda t: -0.5 * mu * lam * np.exp(-2 * t) / s2(t),
                                   0, Tq, limit=400, epsabs=1e-13)[0]
        # ODE: r = sigma_t, Psi = sqrt(lam)/sigma_t, r'' along s
        def rdd(t):
            q = (lam - 1) * np.exp(-2 * t)
            return q * (2 + q) / (1 + q) ** 1.5

        dr = integrate.quad(lambda t: -0.5 * np.sqrt(lam) / np.sqrt(s2(t)) * rdd(t),
                            0, Tq, limit=400, epsabs=1e-13)[0]
        beta_ode = 2 * np.sqrt(lam) * dr
        alpha_ode = integrate.quad(lambda t: -0.5 * mu * np.sqrt(lam) * np.exp(-t) / np.sqrt(s2(t)),
                                   0, Tq, limit=400, epsabs=1e-13)[0]
        a_s, b_s = g.first_order_coeffs(lam, mu, T, "sde")
        a_o, b_o = g.first_order_coeffs(lam, mu, T, "ode")
        assert np.isclose(a_s[0], alpha_sde, rtol=1e-6, atol=1e-9), (lam, "alpha_sde")
        assert np.isclose(b_s[0], beta_sde, rtol=2e-5, atol=1e-7), (lam, "beta_sde")
        assert np.isclose(a_o[0], alpha_ode, rtol=1e-6, atol=1e-9), (lam, "alpha_ode")
        assert np.isclose(b_o[0], beta_ode, rtol=1e-6, atol=1e-9), (lam, "beta_ode")


def test_special_values():
    a, b = g.first_order_coeffs(np.array([1.0, 1e-12]), np.array([1.0, 1.0]), np.inf, "sde")
    assert np.isclose(b[0], 0.5) and np.isclose(a[0], -0.25)
    assert np.isclose(b[1], 1.0, atol=1e-9)        # c_v(0+) = 1
    a, b = g.first_order_coeffs(np.array([1.0]), np.array([1.0]), np.inf, "ode")
    assert np.isclose(b[0], 0.0) and np.isclose(a[0], -0.5)


# --- the main claim: KL = C h^2 + O(h^3), checked by Richardson -------------
@pytest.mark.parametrize("scheme", g.SCHEMES)
@pytest.mark.parametrize("T", [3.0, 10.0])
def test_kl_constant_richardson(scheme, T):
    lam = np.array([0.2, 0.7, 1.0, 4.0, 15.0])
    mu = np.array([0.8, -1.0, 0.4, 1.5, -2.0])
    C = g.kl_constant(lam, mu, T, scheme)
    hs = [0.004, 0.002, 0.001]
    r = [g.kl_output(lam, mu, T, h, scheme, init="exact") / h**2 for h in hs]
    # KL/h^2 = C + c1 h + ...  -> Richardson
    rich = [2 * r[i + 1] - r[i] for i in range(2)]
    assert abs(rich[1] - C) / C < 2e-4, (rich, C)
    # and the error of the raw ratio must shrink linearly
    e = [abs(x - C) for x in r]
    assert 1.7 < e[0] / e[1] < 2.3 and 1.7 < e[1] / e[2] < 2.3


# --- initialisation error ----------------------------------------------------
@pytest.mark.parametrize("scheme", g.SCHEMES)
def test_init_error_continuous_limit(scheme):
    lam, mu, T = np.array([0.3, 1.0, 5.0]), np.array([1.0, 2.0, -1.0]), 2.0
    m_pred, v_pred = g.init_error_law(lam, mu, T, scheme)
    # discrete prior-init law minus discrete exact-init law isolates the init error;
    # it converges to the continuous prediction as h -> 0
    errs = []
    for h in [0.002, 0.001]:
        mp, vp = g.discrete_law(lam, mu, T, h, scheme, "prior")
        me, ve = g.discrete_law(lam, mu, T, h, scheme, "exact")
        errs.append(np.max(np.abs(np.r_[mp - me - (m_pred - mu), vp - ve - (v_pred - lam)])))
    assert errs[1] < 1e-3 and errs[0] / errs[1] > 1.7


def test_ode_init_error_equals_kl_prior_pT():
    """P7: continuous PF flow is an affine bijection -> KL is preserved exactly."""
    lam, mu = np.array([0.3, 1.0, 5.0]), np.array([1.0, 2.0, -1.0])
    for T in [0.5, 2.0, 5.0]:
        m, v = g.init_error_law(lam, mu, T, "ode")
        assert np.isclose(g.kl_gauss_diag(m, v, mu, lam), g.kl_prior_to_pT(lam, mu, T), rtol=1e-10)


def test_sde_init_error_rate_e_minus_4T():
    lam, mu = np.array([0.3, 5.0]), np.array([1.0, -1.0])
    kl = [g.kl_gauss_diag(*g.init_error_law(lam, mu, T, "sde"), mu, lam) for T in (6.0, 7.0)]
    assert np.isclose(kl[0] / kl[1], np.exp(4.0), rtol=1e-3)
    # and the data-processing bound is loose by ~e^{2T}/lam
    T = 6.0
    ratio = g.kl_prior_to_pT(lam, mu, T) / g.kl_gauss_diag(*g.init_error_law(lam, mu, T, "sde"), mu, lam)
    assert ratio > 1e4


# --- small-lambda regime -----------------------------------------------------
def test_small_lambda_ratio_sde_vs_ode():
    """P6: variance part of C ~ 1/(4 lam^2) (SDE) vs 1/(16 lam^2) (ODE)."""
    lam = np.array([1e-4])
    Cs = g.kl_constant(lam, 0.0, np.inf, "sde")
    Co = g.kl_constant(lam, 0.0, np.inf, "ode")
    assert np.isclose(Cs * lam[0] ** 2, 0.25, rtol=5e-3)
    assert np.isclose(Co * lam[0] ** 2, 1 / 16, rtol=5e-3)


def test_left_endpoint_scheme_is_stable_even_for_tiny_lambda():
    """P8 was REFUTED (see AUDIT.md): with the score queried only at t_k >= h,
    sigma_t^2 >= 1 - e^{-2h}, so |1 + h b_k| <= 1 for every lam > 0 (h <= 1).
    Tiny lam costs accuracy, not stability."""
    for lam in [1e-8, 1e-4, 1e-2]:
        for h in [0.5, 0.1, 0.01]:
            T = 2.0
            t = T - h * np.arange(g.n_steps(T, h))
            s2 = g.sigma2(t, lam)
            assert np.all(np.abs(1 + h * (1 - 2 / s2)) <= 1 + 1e-12)
            assert np.all(np.abs(1 + h * (1 - 1 / s2)) <= 1 + 1e-12)
            _, v = g.discrete_law(lam, 0.0, T, h, "sde", "exact")
            assert np.isfinite(v).all() and v[0] < 4 * h + lam


@pytest.mark.parametrize("scheme", g.SCHEMES)
@pytest.mark.parametrize("T", [0.7, 6.0, np.inf])
def test_branch_continuity_near_lambda_one(scheme, T):
    """Regression test for audit bug #2: series branch vs closed-form branch must agree."""
    # pairs straddling the series/closed-form switch at |lam - 1| = 1e-4, 2e-8 apart
    d = 1e-8
    for c in (-1e-4, 1e-4):
        lam = 1.0 + np.array([c - d, c + d])
        a, b = g.first_order_coeffs(lam, 1.0, T, scheme)
        lam_s = 1.0 + c + np.array([-1e-6, 1e-6])
        a_s, b_s = g.first_order_coeffs(lam_s, 1.0, T, scheme)
        for jump, slope in ((a[1] - a[0], (a_s[1] - a_s[0]) / 2e-6),
                            (b[1] - b[0], (b_s[1] - b_s[0]) / 2e-6)):
            assert abs(jump - slope * 2 * d) < 1e-10, (c, jump, slope * 2 * d)


def test_tiny_lambda_precision():
    """Regression test for audit bug #1: c_v(lam) -> 1 as lam -> 0 without cancellation."""
    lam = np.array([1e-6, 1e-9, 1e-12])
    _, b = g.first_order_coeffs(lam, 0.0, np.inf, "sde")
    exact = (2 * lam - 2 - lam**2 * np.log(lam)) / (2 * (lam - 1))
    assert np.allclose(b, exact, rtol=1e-12)
