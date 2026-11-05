"""13.1 hierarchy (B3): η_h = 0, τ_0 = 0 -> identisch mit Standard-Ausrichtung."""

import numpy as np

from swarmsig.agents import neighbors as nb
from swarmsig.agents.hierarchy import Hierarchy, hierarchical_obs
from swarmsig.agents.terms import Align, NeighborObs
from swarmsig.dynamics.history import History
from swarmsig.state import SwarmState


def test_reduces_to_standard_alignment(rng):
    x = rng.uniform(0, 10, (60, 3))
    u = rng.normal(0, 1, (60, 3)) + [0, 10, 0]
    s = SwarmState.initial(0.0, x, u)
    g = nb.topological(x, 6)
    h = Hierarchy.draw(60, 0.0, 0.0, 0.005, rng)
    a_h = Align(3.0)(s, hierarchical_obs(s, g, h, None))
    a_0 = Align(3.0)(s, NeighborObs.exact(s, g))
    assert np.array_equal(a_h, a_0)


def test_weights_and_delays(rng):
    x = rng.uniform(0, 10, (30, 3))
    g = nb.topological(x, 5)
    h = Hierarchy.draw(30, 2.0, 0.3, 0.005, rng)
    w = h.weights(g)
    higher = h.rank[g.dst] > h.rank[g.src]
    assert np.all(w[higher] > 1) and np.all(w[~higher] <= 1)
    lags = h.lags(g)
    assert np.all(lags[~higher] == 0)
    assert lags.max() <= h.max_lag_steps == 60
    assert np.all(np.abs(lags * 0.005 - h.delays(g)) <= 0.0025 + 1e-12)


def test_delayed_values_come_from_history(rng):
    n = 5
    hist = History(10, n, fields=("u",))
    for k in range(10):
        hist.push(u=np.full((n, 3), float(k)))
    vals = hist.lagged("u", np.array([0, 3, 9, 20]), np.array([0, 1, 2, 3]))
    assert np.allclose(vals[:, 0], [9.0, 6.0, 0.0, 0.0])  # Verzögerung > Puffer -> ältester Wert
