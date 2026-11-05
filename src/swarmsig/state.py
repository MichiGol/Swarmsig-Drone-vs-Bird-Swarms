"""Zustände und Trajektorien (Abschnitt 10.1/10.2).

``SwarmState`` ist unveränderlich (``frozen=True``); jeder Integrationsschritt erzeugt ein
neues Objekt. Die enthaltenen Arrays werden per Konvention nicht in-place verändert.
Interne Zustände von Reglern (gehaltener Befehl, GPS-Fehler, Paketalter, Verzögerungs-
puffer) liegen bewusst *nicht* hier, sondern im jeweiligen Modellobjekt eines Laufs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SwarmState:
    """Zustand aller Individuen zur Zeit ``t``.

    x      Positionen (N, 3) in m
    v      Eigengeschwindigkeiten (N, 3) in m/s (Vögel: Luft-, Drohnen: geregelte Geschw.)
    u      Geschwindigkeiten über Grund (N, 3) in m/s (gemessene Größe)
    w_hat  tiefpassgefilterte Windschätzungen (N, 3) in m/s (4.1)
    """

    t: float
    x: np.ndarray
    v: np.ndarray
    u: np.ndarray
    w_hat: np.ndarray

    def __post_init__(self) -> None:
        n = self.x.shape[0]
        for name in ("x", "v", "u", "w_hat"):
            a = getattr(self, name)
            if a.shape != (n, 3):
                raise ValueError(f"SwarmState.{name}: Form {a.shape}, erwartet ({n}, 3)")

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    def replace(self, **changes: object) -> "SwarmState":
        return dataclasses.replace(self, **changes)

    @staticmethod
    def initial(t: float, x: np.ndarray, v: np.ndarray, w: np.ndarray | None = None,
                c_comp: np.ndarray | float = 0.0) -> "SwarmState":
        """Startzustand; Windschätzung startet beim lokalen Wind (eingeschwungener Schätzer)."""
        x = np.asarray(x, dtype=float)
        v = np.asarray(v, dtype=float)
        w = np.zeros_like(x) if w is None else np.asarray(w, dtype=float)
        c = np.broadcast_to(np.asarray(c_comp, dtype=float).reshape(-1, 1), (x.shape[0], 1))
        w_hat = w.copy()
        return SwarmState(float(t), x, v, v + w - c * w_hat, w_hat)


@dataclass(frozen=True)
class Trajectory:
    """Zustände auf Frame-Zeitpunkten (nach Hermite-Interpolation, 4.9)."""

    t: np.ndarray  # (K,)
    x: np.ndarray  # (K, N, 3)
    u: np.ndarray  # (K, N, 3)

    @property
    def n_frames(self) -> int:
        return int(self.t.shape[0])

    @property
    def n(self) -> int:
        return int(self.x.shape[1])


@dataclass(frozen=True)
class Observation:
    """Ausgabe eines Sensors: gemessene Positionen/Geschwindigkeiten je Frame.

    valid (K, N) markiert Individuen, die im Frame gemessen wurden (Fehldetektionen, 6.1).
    """

    t: np.ndarray  # (K,)
    x: np.ndarray  # (K, N, 3)
    u: np.ndarray  # (K, N, 3)
    valid: np.ndarray  # (K, N) bool

    @property
    def n_frames(self) -> int:
        return int(self.t.shape[0])

    def frame(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        m = self.valid[k]
        return self.x[k][m], self.u[k][m]
