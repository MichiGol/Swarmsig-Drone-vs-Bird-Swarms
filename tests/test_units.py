import numpy as np

from swarmsig import units


def test_db_roundtrip():
    x = np.array([1e-12, 0.5, 1.0, 3.0, 1e9])
    assert np.allclose(units.db_to_lin(units.lin_to_db(x)), x, rtol=1e-12, atol=0)
    assert np.allclose(units.db_to_amp(units.amp_to_db(x)), x, rtol=1e-12, atol=0)
    assert units.lin_to_db(100.0) == 20.0
    assert abs(units.amp_to_db(10.0) - 20.0) < 1e-12


def test_ft_roundtrip_and_value():
    assert abs(units.ft_to_m(1000.0) - 304.8) < 1e-12
    x = np.linspace(0, 1e4, 7)
    assert np.allclose(units.m_to_ft(units.ft_to_m(x)), x, rtol=1e-12, atol=1e-12)


def test_scalar_returns_float():
    assert isinstance(units.db_to_lin(3.0), float)
    assert isinstance(units.deg_to_rad(180.0), float)
    assert abs(units.deg_to_rad(180.0) - np.pi) < 1e-15
