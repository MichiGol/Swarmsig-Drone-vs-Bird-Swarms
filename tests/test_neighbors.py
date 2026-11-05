"""13.1 neighbors: topologisch/metrisch gegen Brute-Force."""

import numpy as np

from swarmsig.agents import neighbors as nb


def _brute(x):
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d


def test_topological_matches_bruteforce(rng):
    x = rng.uniform(0, 20, size=(200, 3))
    d = _brute(x)
    g = nb.topological(x, 7)
    expected = [set(np.argsort(d[i])[:7].tolist()) for i in range(200)]
    assert g.as_sets() == expected
    assert np.all(g.counts() == 7)


def test_metric_matches_bruteforce(rng):
    x = rng.uniform(0, 20, size=(200, 3))
    d = _brute(x)
    g = nb.metric(x, 4.0)
    expected = [set(np.flatnonzero(d[i] < 4.0).tolist()) for i in range(200)]
    assert g.as_sets() == expected


def test_limit_nearest(rng):
    x = rng.uniform(0, 10, size=(100, 3))
    d = _brute(x)
    g = nb.limit_nearest(nb.metric(x, 6.0), x, 5)
    for i, s in enumerate(g.as_sets()):
        inside = np.flatnonzero(d[i] < 6.0)
        best = inside[np.argsort(d[i, inside])][:5]
        assert s == set(best.tolist())


def test_duplicates_and_small_n():
    x = np.zeros((4, 3))
    g = nb.topological(x, 6)  # mehr Nachbarn verlangt als vorhanden
    assert np.all(g.counts() == 3)
    assert all(i not in s for i, s in enumerate(g.as_sets()))


def test_segment_mean_and_nearest(rng):
    x = rng.uniform(0, 10, size=(50, 3))
    g = nb.metric(x, 3.0)
    vals = rng.standard_normal((g.n_edges, 3))
    m = g.segment_mean(vals)
    for i in range(50):
        sel = g.src == i
        exp = vals[sel].mean(axis=0) if sel.any() else np.zeros(3)
        assert np.allclose(m[i], exp)
    nn, dist = nb.nearest_neighbor(x)
    d = _brute(x)
    assert np.array_equal(nn, d.argmin(axis=1))
    assert np.allclose(dist, d.min(axis=1))
