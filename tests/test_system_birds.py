"""O21 Stufe 4: Säulendichte, Zuggeschwindigkeit und unbewertbare Vogelspuren."""

import numpy as np
import pytest

from swarmsig.system import birds as B

DENS = np.array([[0.0, 10.0, 20.0, 30.0, 40.0]])  # Vögel/km³ in 0–200, 200–400, …, 800–1000 m


def test_column_density_fill_and_window():
    assert B.column_density(DENS, 600.0)[0] == pytest.approx(0.2 * 10 + 0.2 * 10 + 0.2 * 20)  # unterste Stufe gefüllt
    assert B.column_density(DENS, 600.0, fill_lowest=False)[0] == pytest.approx(0.2 * 10 + 0.2 * 20)
    assert B.column_density(DENS, 1000.0)[0] == pytest.approx(0.2 * 10 + 0.2 * (10 + 20 + 30 + 40))
    assert B.column_density(DENS, 500.0)[0] == pytest.approx(0.2 * 10 + 0.2 * 10 + 0.1 * 20)  # angeschnitten


def test_column_density_missing_and_bad_window():
    d = DENS.copy()
    d[0, 2] = np.nan
    assert np.isnan(B.column_density(d, 600.0)[0])
    assert B.column_density(d, 400.0)[0] == pytest.approx(0.2 * 10 + 0.2 * 10)  # Lücke liegt außerhalb
    with pytest.raises(ValueError):
        B.column_density(DENS, 150.0)


def test_mean_speed():
    ff = np.array([[np.nan, 10.0, 20.0, 5.0, 5.0]])
    assert B.mean_speed(DENS, ff, 600.0)[0] == pytest.approx((10 * 10 + 20 * 20) / 30)
    ff_gap = np.array([[np.nan, 10.0, np.nan, 5.0, 5.0]])
    assert B.mean_speed(DENS, ff_gap, 600.0)[0] == pytest.approx(10.0)
    assert B.mean_speed(np.zeros((1, 5)), ff, 600.0)[0] == pytest.approx(B.DEFAULT_SPEED)


def test_ring_and_rate_formulas():
    assert B.ring_count(10.0, 3.0, 2.0) == pytest.approx(10 * np.pi * 5)
    assert B.ring_count(10.0, 2.0, 3.0) == 0.0
    assert B.never_classified_rate(10.0, 36.0, 3.0, 2.0) == pytest.approx(10 * 36 * 2 * 1)
    assert B.never_classified_rate(10.0, 36.0, 2.0, 3.0) == 0.0


def test_never_classified_rate_monte_carlo():
    """Zufällige Vögel auf geraden Bahnen: Wie viele treten in einer Stunde in den Erkennungskreis ein und bleiben dabei
    immer weiter als R_U entfernt? Erwartung n · v · T · 2 (R_V − R_U)."""
    rng = np.random.default_rng(1)
    n, v, t, r_v, r_u = 50.0, 40.0, 1.0, 3.0, 2.0  # Vögel/km², km/h, h, km, km
    x_lo, x_hi, y_max = -v * t - r_v, r_v, 4.0
    count = rng.poisson(n * (x_hi - x_lo) * 2 * y_max)
    x0 = rng.uniform(x_lo, x_hi, count)
    y = rng.uniform(-y_max, y_max, count)
    inside = np.abs(y) < r_v
    t_in = (-np.sqrt(np.maximum(r_v**2 - y**2, 0.0)) - x0) / v  # Zeitpunkt des Eintritts in den Erkennungskreis
    hits = inside & (t_in >= 0) & (t_in <= t) & (np.abs(y) >= r_u)
    expected = B.never_classified_rate(n, v, r_v, r_u) * t
    assert hits.sum() == pytest.approx(expected, rel=0.05)
