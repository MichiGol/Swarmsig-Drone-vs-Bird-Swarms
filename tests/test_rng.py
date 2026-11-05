"""13.1 rng: gleicher Seed -> identisch; Klassen eines Designpunkts unkorreliert;
Sweep-/Kontrollvarianten verwenden identische Ströme."""

import numpy as np

from swarmsig import rng as R


def test_same_seed_identical():
    a = R.point_seeds(7, 3).for_class("B1").rng("agent", "process_noise").standard_normal(100)
    b = R.point_seeds(7, 3).for_class("B1").rng("agent", "process_noise").standard_normal(100)
    assert np.array_equal(a, b)


def test_stateless_children_independent_of_call_order():
    ps = R.point_seeds(7, 3)
    first = ps.for_class("D1").rng("par", "tau_d").random(5)
    _ = ps.for_class("B1").rng("par", "J").random(5)
    again = ps.for_class("D1").rng("par", "tau_d").random(5)
    assert np.array_equal(first, again)


def test_classes_of_design_point_uncorrelated():
    ps = R.point_seeds(11, 0)
    n = 20000
    a = ps.for_class("B1").rng("agent", "process_noise").standard_normal(n)
    b = ps.for_class("D1").rng("agent", "process_noise").standard_normal(n)
    r = np.corrcoef(a, b)[0, 1]
    assert abs(r) < 4 / np.sqrt(n)


def test_streams_and_points_differ():
    ps0, ps1 = R.point_seeds(11, 0), R.point_seeds(11, 1)
    x = ps0.for_class("B1").rng("agent").random(4)
    assert not np.array_equal(x, ps1.for_class("B1").rng("agent").random(4))
    assert not np.array_equal(x, ps0.for_class("B1").rng("sensor").random(4))
    assert not np.array_equal(ps0.cov_rng("N").random(4), ps0.cov_rng("R").random(4))


def test_sweep_variants_reuse_identical_streams():
    """CRN: dieselben PointSeeds für alle Sweep-Werte -> bitgleiche Ströme."""
    seeds = R.design_seeds(5, 3)
    for ps in seeds:
        s1 = ps.for_class("I").rng("agent", "gps_position").standard_normal(10)
        s2 = ps.for_class("I").rng("agent", "gps_position").standard_normal(10)
        assert np.array_equal(s1, s2)


def test_registry_is_append_only_stable():
    # Index fest: ein Teil der Klassen zu simulieren ändert die übrigen Ströme nicht.
    assert R.CLASS_REGISTRY[:8] == ("B1", "B2", "B3", "D1", "D2", "D2c", "D3", "I")


def test_name_key_stable():
    assert R.name_key("process_noise") == R.name_key("process_noise")
    assert R.name_key("a") != R.name_key("b")
