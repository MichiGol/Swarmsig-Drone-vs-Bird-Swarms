"""Windfelder des Kernpfads (5.2). In P1 und P2 gilt: kein Wind (9.4).

Turbulenz (von Kármán, 5.3) folgt in P3 in ``spectra.py``, ``wind_grid.py`` und
``wind_kraichnan.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Wind(Protocol):
    """Windfeld: Geschwindigkeit (N, 3) in m/s an Positionen x (N, 3) zur Zeit t."""

    def sample(self, x: np.ndarray, t: float) -> np.ndarray: ...


class NoWind:
    def sample(self, x: np.ndarray, t: float) -> np.ndarray:  # noqa: ARG002
        return np.zeros_like(x, dtype=float)


@dataclass(frozen=True)
class ConstantWind:
    w: tuple[float, float, float]

    def sample(self, x: np.ndarray, t: float) -> np.ndarray:  # noqa: ARG002
        return np.broadcast_to(np.asarray(self.w, dtype=float), x.shape).copy()


@dataclass(frozen=True)
class SineWind:
    """Räumlich gleichförmiger Sinus-Wind (für Tests der Windübertragung)."""

    amplitude: tuple[float, float, float]
    period_s: float

    def sample(self, x: np.ndarray, t: float) -> np.ndarray:
        a = np.asarray(self.amplitude, dtype=float) * np.sin(2 * np.pi * t / self.period_s)
        return np.broadcast_to(a, x.shape).copy()


@dataclass(frozen=True)
class PowerLawWind:
    """Mittleres Profil U(z) = U10 (z / 10 m)^α, Richtung höhenunabhängig (5.2).

    ``direction_rad`` ist die Richtung, *in die* der Wind weht (mathematisch: Azimut von
    Nord im Uhrzeigersinn, wie in 2.1). z ≤ 0 wird auf 0 gesetzt.
    """

    U10: float
    alpha: float
    direction_rad: float

    def speed(self, z: np.ndarray) -> np.ndarray:
        z = np.maximum(np.asarray(z, dtype=float), 0.0)
        return self.U10 * (z / 10.0) ** self.alpha

    def sample(self, x: np.ndarray, t: float) -> np.ndarray:  # noqa: ARG002
        e = np.array([np.sin(self.direction_rad), np.cos(self.direction_rad), 0.0])
        return self.speed(x[:, 2])[:, None] * e[None, :]
