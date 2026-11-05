"""Realismus-Schicht für Drohnen (4.7): Takt, Latenz, GPS-Rauschen, Paketverlust,
Trägheit, Sättigung.

* **Regeltakt:** Befehl alle ``n_ctrl = (1/f_ctrl)/Δt`` Schritte neu, dazwischen gehalten
  (Zero-Order-Hold); 1/f_ctrl muss ein ganzzahliges Vielfaches von Δt sein.
* **Latenz:** Der Regler sieht Nachbarzustände von t − τ_d (Ringpuffer); τ_d wird auf ein
  Vielfaches von Δt gerundet, der Rundungsfehler wird protokolliert.
* **GPS:** x̃_j = x_j + ε_j, ũ_j = u_j + η_j, ε und η Ornstein–Uhlenbeck-Prozesse mit
  Zeitkonstante τ_g und stationären Standardabweichungen σ_g bzw. σ_gv (exakte
  Diskretisierung). Der Puffer speichert die gemessenen (verrauschten) Zustände.
* **Paketverlust:** Jede Verbindung i ← j fällt pro Regel-Update mit Wahrscheinlichkeit
  p_loss aus; dann gilt der zuletzt empfangene Wert (Alter wächst um 1/f_ctrl).
* **Gemeinsamer GPS-Fehler (Erweiterung 2026-10-11, P3):** ε_j = √(1−ρ)·ε_j^unabh + √ρ·ε^gem mit
  einem für alle Drohnen gleichen OU-Prozess ε^gem (gleiche σ und τ_g); analog für die
  Geschwindigkeit. ρ = 0 ist das Modell aus 4.7.
* **Eigener Zustand (Erweiterung, optional):** Mit ``own_state_noise`` nutzt der Regler auch für
  sich selbst die eigene GPS-Schätzung x_i + ε_i, u_i + η_i (und v_i + η_i im inneren
  Geschwindigkeitsregelkreis), so wie er sie an die Nachbarn sendet. Der Kommunikationsgraph
  (Funkreichweite) und der für alle gleiche Navigationsterm bleiben bei den wahren Größen.
* **Trägheit/Sättigung:** dv = [(v_cmd − v)/τ_v]_sat dt + σ_d dW, |dv/dt| ≤ a_max, |v| ≤ v_max.

Alle Zufallsquellen haben eigene benannte Unterströme, damit CRN-Sweeps (P4) über einen
Parameter die übrigen Zufallspfade nicht verschieben. Insbesondere wird für jede Kante
immer eine Verlust-Zufallszahl gezogen, auch wenn p_loss = 0 ist.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from ..config import RealismCfg, sample
from ..dynamics.history import History
from ..dynamics.integrator import steps_per
from ..state import SwarmState
from .neighbors import NeighborGraph
from .terms import NeighborObs

RngFor = Callable[[str], np.random.Generator]


@dataclass(frozen=True)
class RealismParams:
    f_ctrl: float
    tau_d: float
    sigma_g: float
    sigma_gv: float
    tau_g: float
    p_loss: float
    tau_v: float
    a_max: float
    v_max: float
    gps_shared_frac: float = 0.0
    own_state_noise: bool = False

    def as_row(self, prefix: str = "d_") -> dict[str, object]:
        return {prefix + k: v for k, v in asdict(self).items()}


def draw_realism_params(cfg: RealismCfg, rng_for: RngFor) -> RealismParams:
    d = lambda name: float(sample(getattr(cfg, name), rng_for(name)))  # noqa: E731
    return RealismParams(
        f_ctrl=d("f_ctrl_hz"), tau_d=d("tau_d_s"), sigma_g=d("sigma_g_m"),
        sigma_gv=d("sigma_gv_mps"), tau_g=d("tau_g_s"), p_loss=d("p_loss"),
        tau_v=d("tau_v_s"), a_max=d("a_max_mps2"), v_max=d("v_max_mps"),
        gps_shared_frac=d("gps_shared_frac"), own_state_noise=bool(cfg.own_state_noise),
    )


def saturate_norm(a: np.ndarray, limit: float) -> np.ndarray:
    """Begrenzt die Zeilennorm von ``a`` auf ``limit`` (Richtung bleibt erhalten)."""
    norm = np.linalg.norm(a, axis=1)
    scale = np.where(norm > limit, limit / np.where(norm > 0, norm, 1.0), 1.0)
    return a * scale[:, None]


class Comms:
    """GPS-Messung, Verzögerungspuffer und Paketverlust eines Drohnenlaufs."""

    def __init__(self, n: int, dt: float, rp: RealismParams, max_packet_age_s: float,
                 rng_for: RngFor) -> None:
        self.n = int(n)
        self.dt = float(dt)
        self.rp = rp
        self.n_ctrl = steps_per(1.0 / rp.f_ctrl, dt)
        self.lag_d = int(round(rp.tau_d / dt))
        self.latency_rounding_s = abs(self.lag_d * dt - rp.tau_d)
        self.max_age = max(int(round(max_packet_age_s / dt)), self.n_ctrl)
        self.history = History(self.lag_d + self.max_age + 1, self.n, fields=("x", "u"))
        self._rng_gx = rng_for("gps_position")
        self._rng_gv = rng_for("gps_velocity")
        self._rng_gxs = rng_for("gps_position_shared")
        self._rng_gvs = rng_for("gps_velocity_shared")
        self._rng_loss = rng_for("packet_loss")
        # Exakte OU-Diskretisierung; Start im stationären Zustand.
        self._a = float(np.exp(-dt / rp.tau_g))
        self._b = float(np.sqrt(1.0 - self._a**2))
        self.eps_x = rp.sigma_g * self._rng_gx.standard_normal((self.n, 3))
        self.eps_u = rp.sigma_gv * self._rng_gv.standard_normal((self.n, 3))
        # Gemeinsamer Anteil (für alle Drohnen gleich) und Gewichte √(1−ρ), √ρ
        self.eps_xs = rp.sigma_g * self._rng_gxs.standard_normal(3)
        self.eps_us = rp.sigma_gv * self._rng_gvs.standard_normal(3)
        self.w_ind = float(np.sqrt(1.0 - rp.gps_shared_frac))
        self.w_sh = float(np.sqrt(rp.gps_shared_frac))
        self._keys = np.zeros(0, np.int64)
        self._ages = np.zeros(0, np.int64)
        self.n_dropped = 0
        self.n_links = 0

    def err_x(self) -> np.ndarray:
        """GPS-Positionsfehler aller Drohnen (N, 3): unabhängiger plus gemeinsamer Anteil."""
        return self.w_ind * self.eps_x + self.w_sh * self.eps_xs

    def err_u(self) -> np.ndarray:
        return self.w_ind * self.eps_u + self.w_sh * self.eps_us

    def own_estimate(self, state: SwarmState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Eigener Zustand, wie ihn der Regler kennt: (x, u, v) exakt oder mit eigenem GPS-Fehler."""
        if not self.rp.own_state_noise:
            return state.x, state.u, state.v
        eu = self.err_u()
        return state.x + self.err_x(), state.u + eu, state.v + eu

    def _push(self, state: SwarmState) -> None:
        self.history.push(x=state.x + self.err_x(), u=state.u + self.err_u())

    def start(self, state: SwarmState) -> None:
        self._push(state)

    def after_step(self, state: SwarmState) -> None:
        self.eps_x = self._a * self.eps_x + self.rp.sigma_g * self._b * self._rng_gx.standard_normal((self.n, 3))
        self.eps_u = self._a * self.eps_u + self.rp.sigma_gv * self._b * self._rng_gv.standard_normal((self.n, 3))
        self.eps_xs = self._a * self.eps_xs + self.rp.sigma_g * self._b * self._rng_gxs.standard_normal(3)
        self.eps_us = self._a * self.eps_us + self.rp.sigma_gv * self._b * self._rng_gvs.standard_normal(3)
        self._push(state)

    def packet_ages(self, g: NeighborGraph) -> np.ndarray:
        """Alter (in Schritten) des zuletzt empfangenen Pakets je Kante; aktualisiert den Speicher."""
        keys = g.src * self.n + g.dst
        drop = self._rng_loss.random(keys.size) < self.rp.p_loss
        prev = np.zeros(keys.size, np.int64)
        if self._keys.size:
            pos = np.searchsorted(self._keys, keys)
            pos_c = np.minimum(pos, self._keys.size - 1)
            found = self._keys[pos_c] == keys
            prev = np.where(found, self._ages[pos_c], 0)
        ages = np.where(drop, np.minimum(prev + self.n_ctrl, self.max_age), 0)
        order = np.argsort(keys, kind="stable")
        self._keys, self._ages = keys[order], ages[order]
        self.n_dropped += int(drop.sum())
        self.n_links += int(keys.size)
        return ages

    def observe(self, g: NeighborGraph) -> NeighborObs:
        """Verrauschte, verzögerte Nachbarzustände (Latenz + Paketalter) je Kante."""
        lags = self.lag_d + self.packet_ages(g)
        return NeighborObs(g, self.history.lagged("x", lags, g.dst),
                           self.history.lagged("u", lags, g.dst))
