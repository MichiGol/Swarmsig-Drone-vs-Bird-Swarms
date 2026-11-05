"""13.1 integrator: Ornstein–Uhlenbeck-Varianz, schwache Ordnung ≈ 1, Zeitschrittregel."""

import numpy as np
import pytest

from swarmsig.agents.terms import Speed
from swarmsig.dynamics.integrator import (ALLOWED_DT_S, dt_from_rule, kappa_eff, simulate,
                                          spectral_radius, steps_per)
from swarmsig.dynamics.wind_transfer import WindTransfer
from swarmsig.state import SwarmState


class _OU:
    """dv = −θ v dt + σ dW, unabhängige Teilchen (Testmodell)."""

    def __init__(self, theta: float, sigma: float) -> None:
        self.theta, self.sigma = theta, sigma
        self.transfer = WindTransfer(0.0, 1.0)

    def start(self, state): pass
    def drift(self, state, step): return -self.theta * state.v
    def constrain(self, v): return v
    def after_step(self, state, step): pass


def _stationary_var(theta, sigma, dt, n, rng, t_end):
    model = _OU(theta, sigma)
    s0 = SwarmState.initial(0.0, np.zeros((n, 3)), np.zeros((n, 3)))
    s = simulate(model, s0, dt=dt, n_steps=int(round(t_end / dt)), noise_rng=rng)
    return float(np.var(s.v))


def test_ou_stationary_variance(rng):
    theta, sigma, dt = 1.0, 0.8, 0.01
    var = _stationary_var(theta, sigma, dt, 40000, rng, 12.0)
    exact_discrete = sigma**2 * dt / (1 - (1 - theta * dt) ** 2)
    se = exact_discrete * np.sqrt(2 / (3 * 40000))
    assert abs(var - exact_discrete) < 4 * se
    assert abs(var - sigma**2 / (2 * theta)) / (sigma**2 / (2 * theta)) < 0.02


def test_weak_order_about_one(rng):
    theta, sigma = 1.0, 1.0
    target = sigma**2 / (2 * theta)
    b1 = _stationary_var(theta, sigma, 0.1, 200000, rng, 10.0) - target
    b2 = _stationary_var(theta, sigma, 0.05, 200000, rng, 10.0) - target
    order = np.log2(b1 / b2)
    assert 0.7 < order < 1.3


def test_timestep_rule_marginal_speed():
    """Großer marginaler Speed-Ausschlag: Regel verlangt kleines Δt; zu großes Δt divergiert."""
    kappa, u_star, s0, dev = 1.0, 1.0, 100.0, 5.0
    k_eff = kappa_eff(kappa, "marginal", u_q=dev, u_star=u_star)
    assert k_eff == pytest.approx(75.0)
    term = Speed(kappa, s0, "marginal", u_star)

    class _M(_OU):
        def __init__(self): super().__init__(0.0, 0.0)
        def drift(self, state, step): return term(state)

    def run(dt):
        s = SwarmState.initial(0.0, np.zeros((1, 3)), np.array([[s0 + dev, 0.0, 0.0]]))
        with np.errstate(over="ignore", invalid="ignore"):
            s = simulate(_M(), s, dt=dt, n_steps=50, noise_rng=np.random.default_rng(0))
        return abs(np.linalg.norm(s.v[0]) - s0)

    dt_ok = 0.1 / k_eff
    assert run(dt_ok) < dev
    assert not np.isfinite(run(10.0 / k_eff)) or run(10.0 / k_eff) > 1e3


def test_dt_from_rule_and_grid():
    dt, ok = dt_from_rule(10.0)  # 0,1/10 = 10 ms
    assert dt == 0.010 and ok
    dt, ok = dt_from_rule(30.0)  # 3,3 ms -> 2,5 ms
    assert dt == 0.0025 and ok
    dt, ok = dt_from_rule(1000.0)
    assert dt == min(ALLOWED_DT_S) and not ok
    for dt in ALLOWED_DT_S:
        for f in (10, 20, 50):
            steps_per(1.0 / f, dt)  # darf nicht werfen
    with pytest.raises(ValueError):
        steps_per(0.05, 0.004)  # 4 ms ist kein zulässiges Raster (12,5 Schritte)


def test_spectral_radius_linear():
    A = np.diag([-3.0, -0.5, 2.0]) + np.triu(np.ones((3, 3)), 1)
    rho = spectral_radius(lambda z: A @ z, np.ones(3))
    assert rho == pytest.approx(3.0, rel=1e-6)
    big = np.diag(-np.linspace(0.1, 7.0, 120))
    assert spectral_radius(lambda z: big @ z, np.ones(120)) == pytest.approx(7.0, rel=1e-3)
