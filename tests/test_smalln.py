"""13.1 smalln: Mittelwertabzug, unabhängige Schwankungen -> E[c_nn] ≈ −1/(N−1);
nach N0-Korrektur E[T_nn] ≈ 0."""

import numpy as np
import pandas as pd

from swarmsig import nullmodels as nm
from swarmsig.observables.modes import remove
from swarmsig.observables.smalln import NNBiasTable, c_nn, t_nn

U = np.array([0.0, 10.0, 0.0])


def _mean_cnn(n, variant, reps, rng):
    vals = []
    for _ in range(reps):
        x = nm.positions(n, 1.0, rng)
        s = nm.n0(x, U, 1.0, 1.0, rng)
        vals.append(c_nn(x, remove(variant, x, s.u).ds))
    return np.mean(vals), np.std(vals) / np.sqrt(reps)


def test_mean_removal_bias(rng):
    n = 50
    m, se = _mean_cnn(n, "M", 2000, rng)
    assert abs(m - (-1 / (n - 1))) < 4 * se + 1e-3


def test_bias_table_correction(rng):
    rows = []
    for n in (30, 60):
        for v in ("A7", "M"):
            m, se = _mean_cnn(n, v, 800, rng)
            rows.append({"N": n, "variant": v, "mean_c_nn": m, "se_c_nn": se, "n_rep": 800})
    table = NNBiasTable(pd.DataFrame(rows))
    t = []
    for _ in range(800):
        x = nm.positions(30, 1.0, rng)
        s = nm.n0(x, U, 1.0, 1.0, rng)
        t.append(t_nn(c_nn(x, remove("A7", x, s.u).ds), 30, "A7", table))
    assert abs(np.mean(t)) < 4 * np.std(t) / np.sqrt(len(t))
    assert np.isnan(t_nn(0.1, 30, "A7", None))
    assert np.isnan(table.bias(1000, "A7"))  # keine Extrapolation
