"""13.1 observables.modes: exakte starre Rotation + Dilatation -> Residuum 0; N3 -> wie N1."""

import numpy as np

from swarmsig import nullmodels as nm
from swarmsig.observables.modes import fit_rigid, remove


def test_exact_rigid_modes_removed(rng):
    x = rng.normal(0, 8, (300, 3))
    U = np.array([1.0, 11.0, -0.5])
    Omega = np.array([0.05, -0.02, 0.3])
    D = 0.07
    u = nm.add_collective_modes(x, np.broadcast_to(U, x.shape).copy(), Omega, D)
    U_f, Om_f, D_f = fit_rigid(x, u)
    assert np.allclose(U_f, U) and np.allclose(Om_f, Omega) and abs(D_f - D) < 1e-12
    res = remove("A7", x, u)
    assert np.max(np.abs(res.du)) < 1e-10
    assert np.max(np.abs(remove("A12", x, u).du)) < 1e-10


def test_zero_sum_residuals(rng):
    x = rng.normal(0, 5, (200, 3))
    u = rng.normal(0, 1, (200, 3)) + [0, 10, 0]
    for v in ("A7", "M", "A12"):
        r = remove(v, x, u)
        assert abs(r.ds.sum()) < 1e-9
        if v != "M":
            assert np.allclose(r.du.sum(axis=0), 0.0, atol=1e-9)
            assert np.allclose(r.du_perp @ r.n_hat, 0.0, atol=1e-12)


def test_n3_residual_equals_n1(rng):
    x = nm.positions(400, 1.0, rng)
    base = nm.n1(x, np.array([0.0, 10.0, 0.0]), 2.0, 3.0, 0.5, 0.5, rng)
    with_modes = nm.n3(base, np.array([0.0, 0.0, 0.2]), 0.05)
    r1 = remove("A7", base.x, base.u)
    r3 = remove("A7", with_modes.x, with_modes.u)
    assert np.allclose(r1.du, r3.du, atol=1e-10)
    # Ohne Pflichtabzug (nur Mittelwert) bleiben die Moden im Residuum:
    assert np.max(np.abs(remove("M", with_modes.x, with_modes.u).du - remove("M", x, base.u).du)) > 0.1


def _velocities(speed, heading):
    return np.column_stack([speed * np.sin(heading), speed * np.cos(heading), np.zeros(speed.size)])


def test_speed_linear_removes_linear_speed_field(rng):
    """O3b: L entfernt ein lineares Speed-Gefälle exakt, unabhängig vom Kursfeld."""
    x = nm.positions(60, 1.0, rng)
    r = x - x.mean(axis=0)
    speed = 10.0 + r @ np.array([0.3, -0.2, 0.1])
    u = _velocities(speed, np.radians(20.0) * r[:, 1] / 3.0)
    assert np.max(np.abs(remove("L", x, u).ds)) < 1e-10


def test_speed_quadratic_removes_quadratic_field(rng):
    x = nm.positions(80, 1.0, rng)
    r = x - x.mean(axis=0)
    speed = 10.0 + 0.2 * r[:, 0] + 0.05 * r[:, 0] * r[:, 1] - 0.03 * r[:, 2] ** 2
    u = _velocities(speed, np.zeros(80))
    assert np.max(np.abs(remove("Q", x, u).ds)) < 1e-10


def test_rigid_speed_removes_exact_rigid_modes(rng):
    """RL: starre Bewegung (auch schnelle Drehung) hinterlässt kein Speed-Residuum."""
    x = rng.normal(0, 5, (100, 3))
    U = np.array([0.0, 8.0, 0.0])
    u = nm.add_collective_modes(x, np.broadcast_to(U, x.shape).copy(), np.array([0.05, 0.0, 0.4]), 0.06)
    assert np.max(np.abs(remove("RL", x, u).ds)) < 1e-9


def test_speed_variants_ignore_nonrigid_turning(rng):
    """Gleiche Speed für alle, Kurs ändert sich über den Schwarm: L und Q ohne Residuum, A7 nicht."""
    x = nm.positions(50, 1.0, rng)
    r = x - x.mean(axis=0)
    u = _velocities(np.full(50, 10.0), np.radians(40.0) * r[:, 1] / np.abs(r[:, 1]).max())
    for v in ("L", "Q"):
        assert np.max(np.abs(remove(v, x, u).ds)) < 1e-10
    assert np.std(remove("A7", x, u).ds) > 0.05


def test_speed_variants_planar_and_zero_sum(rng):
    x = nm.positions(30, 1.0, rng)
    x[:, 2] = 0.0
    u = rng.normal(0, 0.5, (30, 3)) + [0, 10, 0]
    u[:, 2] = 0.0
    for v in ("L", "Q", "RL"):
        ds = remove(v, x, u).ds
        assert np.all(np.isfinite(ds)) and abs(ds.sum()) < 1e-8
