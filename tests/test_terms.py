"""13.1 terms: Ausrichtung = 0 bei gleichen u; Speed = 0 bei s = s0; g ungerade; f(r0) = 0;
harter Kern gekappt; gleichförmige Navigation ändert δu nicht."""

import numpy as np

from swarmsig.agents import neighbors as nb
from swarmsig.agents.terms import (Align, Cohesion, Navigation, NeighborObs, Speed,
                                   cohesion_profile)
from swarmsig.observables.modes import remove
from swarmsig.state import SwarmState


def _state(x, v):
    return SwarmState.initial(0.0, x, v)


def test_align_zero_for_equal_velocities(rng):
    x = rng.uniform(0, 10, (40, 3))
    s = _state(x, np.tile([3.0, 4.0, 0.0], (40, 1)))
    obs = NeighborObs.exact(s, nb.topological(x, 6))
    assert np.allclose(Align(5.0)(s, obs), 0.0)


def test_speed_term_zero_at_s0_and_odd():
    v = np.array([[0.0, 10.0, 0.0], [3.0, 4.0, 0.0]])
    for mode in ("stiff", "marginal"):
        t = Speed(2.0, 10.0, mode, u_star=0.7)
        a = t(_state(np.zeros((2, 3)), np.array([[0.0, 10.0, 0.0], [6.0, 8.0, 0.0]])))
        assert np.allclose(a[0], 0.0)
        dev = np.linspace(-3, 3, 13)
        assert np.allclose(t.g(dev), -t.g(-dev))
    assert np.allclose(Speed(1.0, 5.0, "stiff")(_state(np.zeros((2, 3)), v))[1], 0.0)


def test_cohesion_profile():
    r0, r_w, r_hc = 1.0, 0.5, 0.2
    assert abs(cohesion_profile(np.array([r0]), r0, r_w, r_hc)[0]) < 1e-15
    r = np.array([0.01, 0.05, 0.0999])
    core = cohesion_profile(r, r0, r_w, r_hc) - np.tanh((r - r0) / r_w)
    assert np.allclose(core, core[0])  # unterhalb r_hc/2 gekappt
    eps = 1e-9
    left = cohesion_profile(np.array([2 * r_hc - eps]), r0, r_w, r_hc)[0]
    right = cohesion_profile(np.array([2 * r_hc + eps]), r0, r_w, r_hc)[0]
    assert abs(left - right) < 1e-6  # stetig bei 2 r_hc (Version 1.2)


def test_cohesion_attracts_and_repels():
    x = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    s = _state(x, np.zeros((2, 3)) + [0, 1, 0])
    obs = NeighborObs.exact(s, nb.topological(x, 1))
    a = Cohesion(1.0, 1.0, 0.5, 0.2)(s, obs)
    assert a[0, 0] > 0 and a[1, 0] < 0  # r > r0: Anziehung
    x2 = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    s2 = _state(x2, np.zeros((2, 3)) + [0, 1, 0])
    a2 = Cohesion(1.0, 1.0, 0.5, 0.2)(s2, NeighborObs.exact(s2, nb.topological(x2, 1)))
    assert a2[0, 0] < 0 and a2[1, 0] > 0  # r < r0: Abstoßung


def test_uniform_navigation_does_not_change_residuals(rng):
    x = rng.normal(0, 5, (100, 3))
    u = np.array([0.0, 10.0, 0.0]) + rng.normal(0, 1, (100, 3))
    s = _state(x, u)
    for nav in (Navigation("transit", z0=50.0, kappa_z=0.1, e_T=np.array([1.0, 0, 0]), kappa_n=0.5),
                Navigation("loiter", z0=50.0, kappa_z=0.1, x_c=np.array([30.0, 0, 0]), kappa_h=0.01)):
        a = nav(s)
        assert np.allclose(a, a[0])  # für alle gleich
        for v in ("A7", "M", "A12"):
            r1 = remove(v, x, u)
            r2 = remove(v, x, u + 0.3 * a)
            if v == "M":
                assert np.allclose(r1.du, r2.du)
            else:
                assert np.allclose(r1.du, r2.du, atol=1e-12)
