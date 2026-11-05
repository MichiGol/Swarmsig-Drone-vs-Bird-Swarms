"""13.1 correlation: Identität χ_s(∞) = −(1/N)Σδs²; N1/N2-Skalierung von ξ̂.

Befund beim Aufbau (Oktober 2026): Für Matérn-Felder (N1) wächst die Nullstellen-Länge ξ̂
mit L (log-log-Steigung ≈ 0,5–0,9 bei N = 200…1600), weil der Mittelwertabzug einen Versatz
≈ ∫C d³r / V erzeugt. Die Erwartung „Steigung ≈ 0“ aus 13.1 gilt so nur für Felder mit
∫C d³r = 0 (Loch-Kern). Geprüft wird daher im Standardlauf: Loch-Kern -> Steigung ≈ 0,
N2 -> Steigung ≈ 1; der Matérn-Vergleich läuft als langsamer Test.
"""

import numpy as np
import pytest

from swarmsig import nullmodels as nm
from swarmsig.observables import correlation as corr
from swarmsig.observables.modes import remove
from swarmsig.observables.size import L_g

U = np.array([0.0, 10.0, 0.0])


def test_chi_identity(rng):
    x = rng.normal(0, 5, (150, 3))
    u = rng.normal(0, 1, (150, 3)) + U
    res = remove("A7", x, u)
    ps = corr.pairs(x)
    num, _ = corr.binned_products(ps, res.ds, 0.7)
    chi = corr.chi_curve(num, ps)
    assert chi[-1] == pytest.approx(-np.sum(res.ds**2) / 150, rel=1e-10)
    assert corr.chi_total(res.ds) == pytest.approx(chi[-1], rel=1e-10)


def test_binned_curve_matches_definition(rng):
    x = rng.uniform(0, 6, (40, 3))
    a = rng.normal(size=40)
    ps = corr.pairs(x)
    num, cnt = corr.binned_products(ps, a, 1.0)
    C = corr.curve(num, cnt)
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    for b in range(C.size):
        m = (d >= b) & (d < b + 1)
        np.fill_diagonal(m, False)
        if m.any():
            assert C[b] == pytest.approx((np.outer(a, a)[m]).mean())


def test_first_zero():
    r = np.array([0.5, 1.5, 2.5, 3.5])
    assert corr.first_zero(r, np.array([1.0, 0.5, -0.5, -1.0])) == (2.0, "ok")
    assert corr.first_zero(r, np.array([np.nan, 0.5, -0.5, 1.0]))[0] == pytest.approx(2.0)
    assert corr.first_zero(r, np.array([-0.1, 0.5, -0.5, 1.0])) == (0.5, "nonpositive_start")
    xi, flag = corr.first_zero(r, np.array([1.0, 0.5, 0.2, 0.1]))
    assert np.isnan(xi) and flag == "no_crossing"


def _xi(snap, dr=0.5):
    res = remove("A7", snap.x, snap.u)
    ps = corr.pairs(snap.x)
    num, cnt = corr.binned_products(ps, res.ds, dr)
    C = corr.curve(num, cnt)
    return corr.first_zero(corr.bin_centers(C.size, dr), C)[0]


def _slope(maker, sizes=(200, 1600), reps=4):
    rng = np.random.default_rng(1)
    logL, logxi = [], []
    for n in sizes:
        L, xs = [], []
        for _ in range(reps):
            x = nm.positions(n, 1.0, rng)
            L.append(L_g(x))
            xs.append(_xi(maker(x, rng)))
        logL.append(np.log(np.mean(L)))
        logxi.append(np.log(np.mean(xs)))
    return (logxi[1] - logxi[0]) / (logL[1] - logL[0])


def test_n2_scale_free_and_hole_kernel_finite():
    s_n2 = _slope(lambda x, r: nm.n2(x, U, 1.0, 1.0, 1.0, 1.0, r))
    s_hole = _slope(lambda x, r: nm.n1(x, U, 1.5, 1.5, 1.0, 1.0, r, kernel="hole"))
    assert 0.7 < s_n2 < 1.4
    assert abs(s_hole) < 0.25


@pytest.mark.slow
def test_n1_matern_drifts_hole_does_not():
    """Matérn-N1 gegen N2 über einen größeren L-Bereich mit mehr Wiederholungen.

    Beim Aufbau gemessen (je 4 Wiederholungen, N = 200 vs. 1600): Steigung N1 ≈ 0,5–0,9,
    N2 ≈ 0,9–1,2 — die Trennung über die Nullstellen-Länge ist bei diesen Größen schwach.
    Der Test dokumentiert, ob sie mit größerem L-Bereich deutlicher wird.
    """
    s_n2 = _slope(lambda x, r: nm.n2(x, U, 1.0, 1.0, 1.0, 1.0, r), sizes=(200, 3200), reps=12)
    s_n1 = _slope(lambda x, r: nm.n1(x, U, 1.5, 1.5, 1.0, 1.0, r), sizes=(200, 3200), reps=12)
    print(f"Steigung N1 (Matérn) = {s_n1:.2f}, N2 = {s_n2:.2f}")
    assert s_n1 < s_n2 - 0.25


def test_random_fourier_features_match_matern(rng):
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    f = np.stack([nm.gaussian_field(x, 2.0, 0.5, rng, cholesky_max_n=0, n_features=512)[:, 0]
                  for _ in range(3000)])
    emp = np.mean(f[:, :1] * f[:, 1:], axis=0)
    assert np.allclose(emp, nm.matern(np.array([1.0, 3.0]), 2.0, 0.5), atol=0.07)
    assert abs(np.var(f[:, 0]) - 1.0) < 0.08
