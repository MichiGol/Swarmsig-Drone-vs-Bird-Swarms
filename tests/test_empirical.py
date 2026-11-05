"""O3a: Hauptschwarm-Filter und Bias-Korrektur je Bild für echte Daten."""

import numpy as np

from swarmsig import nullmodels as nm
from swarmsig.observables.empirical import main_cluster, null_bias
from swarmsig.observables.modes import remove
from swarmsig.observables.smalln import c_nn

U = np.array([0.0, 10.0, 0.0])


def test_main_cluster_drops_straggler(rng):
    x = nm.positions(20, 1.0, rng)
    x = np.vstack([x, [500.0, 0.0, 0.0]])
    m = main_cluster(x)
    assert m[:20].all() and not m[20]


def test_main_cluster_keeps_larger_group(rng):
    a = nm.positions(15, 1.0, rng)
    b = nm.positions(8, 1.0, rng) + np.array([300.0, 0.0, 0.0])
    m = main_cluster(np.vstack([a, b]))
    assert m[:15].all() and not m[15:].any()


def test_null_bias_matches_direct_n0(rng):
    """Mittel der Bias-Korrektur je Bild über zufällige Positionen = E_N0[c_nn | N] (wie die Tabelle)."""
    n, draws = 30, 150
    cond = {"A7": [], "M": []}
    direct = {"A7": [], "M": []}
    for _ in range(draws):
        x = nm.positions(n, 1.0, rng)
        b = null_bias(x, U, ("A7", "M"), rng, reps=10)
        u = nm.n0(x, U, 1.0, 1.0, rng).u
        for v in cond:
            cond[v].append(b[v])
            direct[v].append(c_nn(x, remove(v, x, u).ds))
    for v in cond:
        se = np.std(direct[v]) / np.sqrt(draws)
        assert abs(np.mean(cond[v]) - np.mean(direct[v])) < 4 * se + 0.005
    assert abs(np.mean(cond["M"]) - (-1 / (n - 1))) < 0.01


def test_null_bias_planar(rng):
    """Horizontale Daten (z = 0) mit 2D-Schwankungen: M wie in 3D; A7 gleicht direktem planarem N0."""
    n, draws = 12, 200
    cond, direct = [], []
    for _ in range(draws):
        x = nm.positions(n, 1.0, rng)
        x[:, 2] = 0.0
        cond.append(null_bias(x, U, ("A7", "M"), rng, reps=5, dim=2))
        u = U + np.column_stack([rng.standard_normal((n, 2)), np.zeros(n)])
        direct.append(c_nn(x, remove("A7", x, u).ds))
    assert abs(np.mean([b["M"] for b in cond]) - (-1 / (n - 1))) < 0.02
    se = np.std(direct) / np.sqrt(draws)
    assert abs(np.mean([b["A7"] for b in cond]) - np.mean(direct)) < 4 * se + 0.01
