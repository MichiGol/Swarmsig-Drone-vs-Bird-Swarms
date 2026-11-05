"""Verhaltensterme der Agentenmodelle (4.1, 4.2, 4.4).

Jeder Term ist eine austauschbare Komponente (Strategy-Pattern, 10.3) und liefert eine
Beschleunigung (N, 3) in m/s². Ausrichtungsterme verwenden Geschwindigkeiten **über
Grund** ``u`` (Wahrnehmung im Bodensystem, 4.1), der Speed-Term wirkt auf die
Eigen-Speed ``s = |v|``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..state import SwarmState
from .neighbors import NeighborGraph


@dataclass(frozen=True)
class NeighborObs:
    """Was jedes Individuum über seine Nachbarn weiß (je Kante ``e = (i <- j)``).

    x_nb, u_nb  beobachtete Position bzw. Bodengeschwindigkeit des Nachbarn j (E, 3);
                bei Vögeln die wahren Werte (B3: u verzögert), bei Drohnen verrauscht/verzögert
    weight      optionale Kantengewichte (E,), z. B. Hierarchiegewichte ω_ij (B3)
    """

    graph: NeighborGraph
    x_nb: np.ndarray
    u_nb: np.ndarray
    weight: np.ndarray | None = None

    @staticmethod
    def exact(state: SwarmState, graph: NeighborGraph) -> "NeighborObs":
        return NeighborObs(graph, state.x[graph.dst], state.u[graph.dst])


class Term(Protocol):
    def __call__(self, state: SwarmState, obs: NeighborObs) -> np.ndarray: ...


@dataclass(frozen=True)
class Align:
    """a_i = J (⟨u⟩_N_i − u_i), optional gewichtet (B3: Σω(u_j(t−τ_ij) − u_i)/Σω)."""

    J: float

    def __call__(self, state: SwarmState, obs: NeighborObs) -> np.ndarray:
        g = obs.graph
        if g.n_edges == 0:
            return np.zeros_like(state.u)
        diff = obs.u_nb - state.u[g.src]
        return self.J * g.segment_mean(diff, obs.weight)


def g_stiff(dev: np.ndarray, u_star: float) -> np.ndarray:  # noqa: ARG001
    return dev


def g_marginal(dev: np.ndarray, u_star: float) -> np.ndarray:
    return dev**3 / u_star**2


@dataclass(frozen=True)
class Speed:
    """a_i = −κ_s g(s_i − s_0) v̂_i mit g(u) = u (steif) oder u³/u_*² (marginal), 4.2."""

    kappa_s: float
    s0: float
    mode: str  # "stiff" | "marginal"
    u_star: float = 1.0

    def g(self, dev: np.ndarray) -> np.ndarray:
        if self.mode == "stiff":
            return g_stiff(dev, self.u_star)
        if self.mode == "marginal":
            return g_marginal(dev, self.u_star)
        raise ValueError(f"unbekannte Speed-Regelung {self.mode!r}")

    def __call__(self, state: SwarmState, obs: NeighborObs | None = None) -> np.ndarray:
        s = np.linalg.norm(state.v, axis=1)
        ok = s > 0
        vhat = np.zeros_like(state.v)
        vhat[ok] = state.v[ok] / s[ok, None]
        return -self.kappa_s * self.g(s - self.s0)[:, None] * vhat


def cohesion_profile(r: np.ndarray, r0: float, r_w: float, r_hc: float) -> np.ndarray:
    """f(r) aus 4.2: tanh((r−r0)/r_w) − [(r_hc/max(r, r_hc/2))^6 − (1/2)^6]·1{r < 2 r_hc}.

    Die Verschiebung um (1/2)^6 macht die Kraft bei r = 2 r_hc stetig, die Kappung bei
    r_hc/2 („Softening“) beschränkt die Steifigkeit (4.9).
    """
    r = np.asarray(r, dtype=float)
    hc = (r_hc / np.maximum(r, 0.5 * r_hc)) ** 6 - 0.5**6
    return np.tanh((r - r0) / r_w) - np.where(r < 2.0 * r_hc, hc, 0.0)


@dataclass(frozen=True)
class Cohesion:
    """a_i = β/|N_i| Σ_j f(r_ij) ê_ij (paarweise über dieselben Nachbarn, 4.2)."""

    beta: float
    r0: float
    r_w: float
    r_hc: float

    def __call__(self, state: SwarmState, obs: NeighborObs) -> np.ndarray:
        g = obs.graph
        if g.n_edges == 0:
            return np.zeros_like(state.x)
        d = obs.x_nb - state.x[g.src]
        r = np.linalg.norm(d, axis=1)
        ok = r > 0
        e = np.zeros_like(d)
        e[ok] = d[ok] / r[ok, None]
        f = cohesion_profile(r, self.r0, self.r_w, self.r_hc)
        return self.beta * g.segment_mean(f[:, None] * e)


@dataclass(frozen=True)
class Navigation:
    """Gleichförmiger Navigationsterm (4.2), für alle Individuen identisch.

    transit: a = −κ_n P_h(ū − |P_h ū| ê_T) − κ_z (z̄ − z0) ẑ
    loiter:  a = −κ_h P_h(x̄ − x_c)        − κ_z (z̄ − z0) ẑ

    Hängt nur von Schwerpunkt x̄ und Mittelgeschwindigkeit ū ab und verschwindet daher
    exakt beim Abzug kollektiver Moden.
    """

    mode: str  # "transit" | "loiter"
    z0: float
    kappa_z: float
    e_T: np.ndarray | None = None  # Zielkurs (horizontal, Einheitsvektor)
    kappa_n: float = 0.0
    x_c: np.ndarray | None = None  # Kreismittelpunkt (horizontal)
    kappa_h: float = 0.0

    def acceleration(self, xbar: np.ndarray, ubar: np.ndarray) -> np.ndarray:
        a = np.zeros(3)
        if self.mode == "transit":
            if self.e_T is None:
                raise ValueError("transit braucht e_T")
            uh = np.array([ubar[0], ubar[1], 0.0])
            a -= self.kappa_n * (uh - np.linalg.norm(uh) * self.e_T)
        elif self.mode == "loiter":
            if self.x_c is None:
                raise ValueError("loiter braucht x_c")
            a[:2] -= self.kappa_h * (xbar[:2] - self.x_c[:2])
        else:
            raise ValueError(f"unbekannter Flugmodus {self.mode!r}")
        a[2] -= self.kappa_z * (xbar[2] - self.z0)
        return a

    def __call__(self, state: SwarmState, obs: NeighborObs | None = None) -> np.ndarray:
        a = self.acceleration(state.x.mean(axis=0), state.u.mean(axis=0))
        return np.broadcast_to(a, state.x.shape).copy()
