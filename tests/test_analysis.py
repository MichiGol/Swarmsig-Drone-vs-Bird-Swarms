"""13.1 analysis: eigene U-Statistik = Mann–Whitney auf unabhängigen Stichproben; Diagonale
ausgeschlossen; Designpunkt-Bootstrap (Überdeckung als langsamer Test)."""

import numpy as np
import pytest
from scipy import stats

from swarmsig.experiment.analysis import (auc_u, bootstrap_auc, error_rates, required_runs,
                                          roc_curve, wilson_interval)


def test_auc_matches_mann_whitney_without_diagonal_effect(rng):
    b = rng.normal(1.0, 1.0, 300)
    d = rng.normal(0.0, 1.0, 300)
    U = stats.mannwhitneyu(b, d).statistic / (300 * 300)
    assert auc_u(b, d, exclude_diagonal=False) == pytest.approx(U, rel=1e-12)


def test_diagonal_excluded():
    b = np.array([5.0, 0.0, 0.0])
    d = np.array([4.0, 1.0, 1.0])
    # ohne Diagonale: Paare (0,1), (0,2) -> 1; (1,0), (1,2), (2,0), (2,1) -> 0  => 2/6
    assert auc_u(b, d) == pytest.approx(2 / 6)
    assert auc_u(b, d, exclude_diagonal=False) == pytest.approx(3 / 9)


def test_ties_and_nan():
    b = np.array([1.0, 2.0, np.nan, 3.0])
    d = np.array([1.0, np.nan, 0.0, 3.0])
    # gültige Kreuzpaare a≠b: (0,2):1, (0,3):0, (1,0):1, (1,2):1, (1,3):0, (3,0):1, (3,2):1 ; Diagonale (0,0),(3,3) raus
    assert auc_u(b, d) == pytest.approx(5 / 7)


def test_weights_equal_duplication(rng):
    b = rng.normal(size=20)
    d = rng.normal(size=20)
    w = rng.integers(0, 4, size=20).astype(float)
    idx = np.repeat(np.arange(20), w.astype(int))
    # Kopien desselben Designpunkts dürfen nicht miteinander verglichen werden
    bb, dd = b[idx], d[idx]
    m = len(idx)
    num = den = 0.0
    for i in range(m):
        for j in range(m):
            if idx[i] == idx[j]:
                continue
            num += 1.0 if bb[i] > dd[j] else (0.5 if bb[i] == dd[j] else 0.0)
            den += 1
    assert auc_u(b, d, weights=w) == pytest.approx(num / den)


def test_error_rates_roc_wilson():
    b = np.array([1.0, 2.0, 3.0, 4.0])
    d = np.array([0.0, 1.5, 2.5, 10.0])
    pfa, pmiss = error_rates(b, d, 2.0)
    assert pfa == 0.5 and pmiss == 0.5
    f, t, _ = roc_curve(b, d)
    assert f[0] == 0 and t[0] == 0 and f[-1] == 1 and t[-1] == 1
    lo, hi = wilson_interval(5, 100)
    assert 0.02 < lo < 0.05 < hi < 0.12
    assert required_runs(0.05, 0.01) in (1825, 1825 + 1)


def test_bootstrap_contains_estimate(rng):
    b = rng.normal(0.8, 1, 150)
    d = rng.normal(0.0, 1, 150)
    res = bootstrap_auc(b, d, rng, n_boot=300)
    assert res.low < res.estimate < res.high


@pytest.mark.slow
def test_bootstrap_coverage():
    """Überdeckung ≈ 95 % auf synthetischen Daten mit bekannter AUC (13.1; langsam)."""
    rng = np.random.default_rng(7)
    delta, c = 0.8, 0.7
    # Begegnungen unabhängiger Designpunkte: B − D = δ + c(θ_a − θ_b) + ε_a − ε_b
    true_auc = stats.norm.cdf(delta / np.sqrt(2 + 2 * c**2))
    hits = 0
    n_sets = 300
    for _ in range(n_sets):
        cov = rng.normal(size=120)  # gemeinsame Kovariate je Designpunkt -> Abhängigkeit
        b = delta + c * cov + rng.normal(size=120)
        d = c * cov + rng.normal(size=120)
        r = bootstrap_auc(b, d, rng, n_boot=1000)
        hits += r.low <= true_auc <= r.high
    assert 0.93 <= hits / n_sets <= 0.97
