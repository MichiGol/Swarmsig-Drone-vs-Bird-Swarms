"""B3: hierarchische, verzögerte Ausrichtung (4.4).

Ränge h_i ~ U[0, 1]; Gewichte ω_ij = exp(η_h (h_j − h_i)); Verzögerungen
τ_ij = τ_0 (h_j − h_i)_+, gerundet auf Vielfache von Δt (Rundungsfehler wird protokolliert).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dynamics.history import History
from ..state import SwarmState
from .neighbors import NeighborGraph
from .terms import NeighborObs


@dataclass(frozen=True)
class Hierarchy:
    rank: np.ndarray  # (N,)
    eta_h: float
    tau0: float
    dt: float

    @staticmethod
    def draw(n: int, eta_h: float, tau0: float, dt: float, rng: np.random.Generator) -> "Hierarchy":
        return Hierarchy(rng.uniform(0.0, 1.0, size=n), float(eta_h), float(tau0), float(dt))

    @property
    def max_lag_steps(self) -> int:
        return int(np.ceil(self.tau0 / self.dt - 1e-12)) if self.tau0 > 0 else 0

    def weights(self, g: NeighborGraph) -> np.ndarray:
        return np.exp(self.eta_h * (self.rank[g.dst] - self.rank[g.src]))

    def delays(self, g: NeighborGraph) -> np.ndarray:
        return self.tau0 * np.maximum(self.rank[g.dst] - self.rank[g.src], 0.0)

    def lags(self, g: NeighborGraph) -> np.ndarray:
        return np.rint(self.delays(g) / self.dt).astype(np.int64)

    def max_rounding_error(self) -> float:
        """Größter möglicher Rundungsfehler der Verzögerungen in s (≤ Δt/2)."""
        return 0.5 * self.dt if self.tau0 > 0 else 0.0


def hierarchical_obs(state: SwarmState, g: NeighborGraph, h: Hierarchy,
                     history: History | None) -> NeighborObs:
    """Nachbarbeobachtung für B3: verzögerte Bodengeschwindigkeiten und Gewichte ω_ij."""
    w = h.weights(g)
    if history is None or h.tau0 <= 0:
        u_nb = state.u[g.dst]
    else:
        u_nb = history.lagged("u", h.lags(g), g.dst)
    return NeighborObs(g, state.x[g.dst], u_nb, w)
