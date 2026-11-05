"""Einheiten und Konstanten (Regel 3).

Intern wird ausschließlich in SI gerechnet (m, s, kg, Hz, W, rad). Dezibel, Fuß und Grad
kommen nur an den Rändern vor (Configs, Plots, MIL-Formeln) und werden ausschließlich
über die Funktionen dieses Moduls umgerechnet.
"""

from __future__ import annotations

import numpy as np

C_LIGHT = 299_792_458.0  # m/s; Quelle: SI-Definition (exakt)
K_BOLTZMANN = 1.380_649e-23  # J/K; Quelle: SI-Definition (exakt)
T0_K = 290.0  # K; Quelle: IEEE-Referenztemperatur für Rauschzahlen
FOOT_M = 0.3048  # m; Quelle: internationaler Fuß (exakt)
G0 = 9.806_65  # m/s²; Quelle: Normfallbeschleunigung (exakt)


def _ret(a: np.ndarray) -> float | np.ndarray:
    return float(a) if a.ndim == 0 else a


def db_to_lin(x_db: float | np.ndarray) -> float | np.ndarray:
    """Leistungsgröße: dB -> linear (10^(x/10))."""
    return _ret(10.0 ** (np.asarray(x_db, dtype=float) / 10.0))


def lin_to_db(x: float | np.ndarray) -> float | np.ndarray:
    """Leistungsgröße: linear -> dB (10 lg x)."""
    return _ret(10.0 * np.log10(np.asarray(x, dtype=float)))


def db_to_amp(x_db: float | np.ndarray) -> float | np.ndarray:
    """Amplitudengröße: dB -> linear (10^(x/20))."""
    return _ret(10.0 ** (np.asarray(x_db, dtype=float) / 20.0))


def amp_to_db(a: float | np.ndarray) -> float | np.ndarray:
    """Amplitudengröße: linear -> dB (20 lg a)."""
    return _ret(20.0 * np.log10(np.asarray(a, dtype=float)))


def ft_to_m(ft: float | np.ndarray) -> float | np.ndarray:
    return _ret(np.asarray(ft, dtype=float) * FOOT_M)


def m_to_ft(m: float | np.ndarray) -> float | np.ndarray:
    return _ret(np.asarray(m, dtype=float) / FOOT_M)


def deg_to_rad(deg: float | np.ndarray) -> float | np.ndarray:
    return _ret(np.deg2rad(np.asarray(deg, dtype=float)))


def rad_to_deg(rad: float | np.ndarray) -> float | np.ndarray:
    return _ret(np.rad2deg(np.asarray(rad, dtype=float)))
