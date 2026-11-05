import numpy as np
import pytest

import gaussian as g
import mixture as mx

MIX = mx.Mixture1D([0.3, 0.7], [-2.0, 1.5], [0.15, 0.4])


def test_score_derivatives_vs_complex_step():
    y = np.linspace(-4, 4, 41)
    d = 1e-30
    for t in [0.0, 0.3, 2.0]:
        S, Sp, Spp = MIX.score_derivs(t, y)
        lp_c = MIX.logpdf(t, y + 1j * d)
        assert np.allclose(np.imag(lp_c) / d, S, rtol=1e-10, atol=1e-10)
        S_c, Sp_c, _ = MIX.score_derivs(t, y + 1j * d)
        assert np.allclose(np.imag(S_c) / d, Sp, rtol=1e-10, atol=1e-10)
        assert np.allclose(np.imag(Sp_c) / d, Spp, rtol=1e-9, atol=1e-9)


def test_dt_score_matches_time_derivative():
    """Checks the Fokker–Planck-based formula for d/dt of the score."""
    y = np.linspace(-4, 4, 41)
    for t in [0.05, 0.5, 3.0]:
        dt = 1e-5
        fd = (MIX.score_derivs(t + dt, y)[0] - MIX.score_derivs(t - dt, y)[0]) / (2 * dt)
        assert np.allclose(MIX.dt_score(t, y), fd, rtol=1e-6, atol=1e-6)


def test_marginal_normalised():
    y = np.linspace(-15, 15, 30001)
    for t in [0.0, 1.0, 8.0]:
        assert np.isclose(np.trapezoid(MIX.pdf(t, y), y), 1.0, atol=1e-10)


@pytest.mark.parametrize("init", ["exact", "prior"])
def test_em_density_matches_exact_gaussian_recursion(init):
    lam, mu, T, h = 0.4, 0.8, 4.0, 0.02
    k_grid = mx.kl_sde(mx.Mixture1D([1], [mu], [lam]), T, h, init=init)
    k_exact = g.kl_output(lam, mu, T, h, "sde", init=init)
    assert np.isclose(k_grid, k_exact, rtol=1e-8)


def test_kl_ode_matches_exact_gaussian_recursion():
    for lam, mu in [(0.4, 0.8), (3.0, -1.0)]:
        T, h = 4.0, 0.02
        assert np.isclose(mx.kl_ode(mx.Mixture1D([1], [mu], [lam]), T, h),
                          g.kl_output(lam, mu, T, h, "ode", init="exact"), rtol=1e-8)


def test_ode_constant_reduces_to_gaussian_closed_form():
    for lam, mu in [(0.4, 0.8), (3.0, -1.0)]:
        C = mx.ode_first_order_constant(mx.Mixture1D([1], [mu], [lam]), 4.0, n_ref=400, dx=4e-3)
        assert np.isclose(C, g.kl_constant(lam, mu, 4.0, "ode"), rtol=1e-6)


def test_em_density_matches_monte_carlo_on_mixture():
    """The grid propagator must agree with brute-force particles of the same EM chain."""
    rng = np.random.default_rng(3)
    T, h, n = 3.0, 0.05, 400_000
    x, rho = mx.em_density(MIX, T, h, init="prior")
    dx = x[1] - x[0]
    Y = rng.standard_normal(n)
    for k in range(g.n_steps(T, h)):
        t = T - k * h
        S = MIX.score_derivs(t, Y)[0]
        Y = Y + h * (Y + 2 * S) + np.sqrt(2 * h) * rng.standard_normal(n)
    for p in (1, 2, 3):
        grid_moment = np.sum(rho * x**p) * dx
        se = np.std(Y**p) / np.sqrt(n)
        assert abs(np.mean(Y**p) - grid_moment) < 5 * se
    # probability of the left mode region
    pl = np.sum(rho[x < -0.6]) * dx
    assert abs(np.mean(Y < -0.6) - pl) < 5 * np.sqrt(pl * (1 - pl) / n)


def test_pushforward_identity_of_exact_flow():
    """Exact PF flow maps p_T to p_0:  log p_T(x) - log Phi'(x) - log p_0(Phi(x)) = 0,
    and the RK4 residual must shrink at 4th order."""
    errs = []
    for n in (300, 600, 1200):
        _, P = mx.ode_first_order_constant(MIX, 6.0, n_ref=n, dx=4e-3, return_parts=True)
        res = P["logq"] - P["logJ"] - MIX.logpdf(0.0, P["Y"])
        errs.append(np.max(np.abs(res) * np.exp(P["logq"])))
    # complex-step derivative of the RK4 map vs integrated log-Jacobian: both O(H^4)-close
    assert np.allclose(np.log(P["dPhi"]), P["logJ"], atol=1e-8)
    assert errs[-1] < 2e-7
    assert np.log2(errs[1] / errs[2]) > 3.5
