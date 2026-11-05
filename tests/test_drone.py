"""13.1 drone: φ(0) = 0, ρ_h stetig, D2-c-Navigation ändert δu nicht; Realismus-Schicht."""

import numpy as np
import pytest

from swarmsig.agents import neighbors as nb
from swarmsig.agents.drone import braking_curve, phi, rho_h, sigma_norm
from swarmsig.agents.realism import Comms, RealismParams, saturate_norm
from swarmsig.state import SwarmState


def test_phi_zero_at_origin():
    for a, b in ((5.0, 5.0), (1.0, 5.0), (2.0, 3.0)):
        assert abs(float(phi(np.array(0.0), a, b))) < 1e-12


def test_rho_h_continuity():
    h, eps = 0.2, 1e-9
    assert rho_h(np.array(0.0), h) == 1.0
    assert abs(rho_h(np.array(h - eps), h) - rho_h(np.array(h + eps), h)) < 1e-6
    assert abs(rho_h(np.array(1.0 - eps), h) - rho_h(np.array(1.0 + eps), h)) < 1e-6
    assert rho_h(np.array(1.5), h) == 0.0


def test_sigma_norm_and_braking_curve():
    assert sigma_norm(0.0, 0.1) == 0.0
    a, p = 4.16, 3.2
    r = a / p**2
    assert abs(braking_curve(np.array(r - 1e-9), a, p) - braking_curve(np.array(r + 1e-9), a, p)) < 1e-6
    assert braking_curve(np.array(-1.0), a, p) == 0.0


def test_d2c_navigation_is_uniform():
    # D2-c wirkt auf den Schwerpunkt: gleicher Vektor für alle -> δu (A7) unverändert
    from swarmsig.observables.modes import remove
    rng = np.random.default_rng(3)
    x = rng.normal(0, 10, (50, 3))
    u = rng.normal(0, 1, (50, 3)) + [8.0, 0, 0]
    nav = -0.04 * (x.mean(axis=0) - [5.0, 0, 0]) - 0.4 * (u.mean(axis=0) - [10.0, 0, 0])
    assert np.allclose(remove("A7", x, u).du, remove("A7", x, u + 0.1 * nav).du, atol=1e-12)


def _rp(**kw):
    base = dict(f_ctrl=20.0, tau_d=0.11, sigma_g=0.5, sigma_gv=0.1, tau_g=5.0, p_loss=0.0,
                tau_v=0.3, a_max=6.0, v_max=20.0)
    base.update(kw)
    return RealismParams(**base)


def _rng_for(seed):
    from swarmsig.rng import name_key
    return lambda name: np.random.default_rng([seed, name_key(name)])


def test_latency_rounding_and_grid():
    c = Comms(10, 0.005, _rp(), 1.0, _rng_for(1))
    assert c.n_ctrl == 10 and c.lag_d == 22
    assert c.latency_rounding_s == pytest.approx(0.0, abs=1e-12)
    c2 = Comms(10, 0.005, _rp(tau_d=0.1234), 1.0, _rng_for(1))
    assert c2.latency_rounding_s <= 0.0025


def test_packet_loss_ages():
    x = np.random.default_rng(0).uniform(0, 5, (20, 3))
    g = nb.metric(x, 3.0)
    c0 = Comms(20, 0.005, _rp(p_loss=0.0), 0.2, _rng_for(2))
    assert np.all(c0.packet_ages(g) == 0)
    c1 = Comms(20, 0.005, _rp(p_loss=1.0), 0.2, _rng_for(2))
    a1, a2 = c1.packet_ages(g), c1.packet_ages(g)
    assert np.all(a1 == c1.n_ctrl) and np.all(a2 == 2 * c1.n_ctrl)
    for _ in range(10):
        a = c1.packet_ages(g)
    assert np.all(a == c1.max_age)


def test_gps_ou_stationary_std():
    n = 4000
    c = Comms(n, 0.01, _rp(sigma_g=0.7, tau_g=0.5), 1.0, _rng_for(5))
    s = SwarmState.initial(0.0, np.zeros((n, 3)), np.ones((n, 3)))
    c.start(s)
    for _ in range(300):
        c.after_step(s)
    assert abs(np.std(c.eps_x) - 0.7) / 0.7 < 0.03
    obs = c.observe(nb.topological(np.random.default_rng(1).uniform(0, 50, (n, 3)), 3))
    assert obs.x_nb.shape == (3 * n, 3)


def test_saturate_norm():
    a = np.array([[3.0, 4.0, 0.0], [0.3, 0.4, 0.0]])
    s = saturate_norm(a, 1.0)
    assert np.allclose(np.linalg.norm(s, axis=1), [1.0, 0.5])
    assert np.allclose(s[0] / np.linalg.norm(s[0]), [0.6, 0.8, 0.0])
