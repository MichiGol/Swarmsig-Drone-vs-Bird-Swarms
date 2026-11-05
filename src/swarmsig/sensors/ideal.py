"""Idealer Sensor / Tracking (6.1, P1; Referenz „Stereokamera“).

* Alle Individuen sichtbar während T_obs; Bildrate f_cam.
* Positionsrauschen x̃ = x + ε, ε ~ N(0, σ_pos² I).
* Geschwindigkeit per zentraler Differenz über ±q Bilder:
  ũ(t) = (x̃(t + qδ) − x̃(t − qδ)) / (2qδ), δ = 1/f_cam.
* Optionale Fehldetektionen mit Wahrscheinlichkeit p_miss (je Individuum und Bild); ein
  Individuum gilt in einem Frame als gemessen, wenn es in t − qδ, t und t + qδ erkannt wurde.

Die Rauschzahlen werden immer gezogen (auch bei σ_pos = 0, p_miss = 0), damit Varianten
mit gemeinsamen Zufallszahlen vergleichbar bleiben.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..state import Observation, Trajectory


@dataclass(frozen=True)
class IdealSensor:
    sigma_pos: float
    f_cam: float
    q: int = 1
    p_miss: float = 0.0

    @property
    def delta(self) -> float:
        return 1.0 / self.f_cam

    def frame_times(self, t_start: float, t_obs: float) -> np.ndarray:
        """Aufnahmezeitpunkte inkl. q Randbildern vor und nach dem Beobachtungsfenster."""
        k_obs = int(round(t_obs * self.f_cam)) + 1
        k = np.arange(-self.q, k_obs + self.q)
        return t_start + k * self.delta

    def observe(self, traj: Trajectory, rng: np.random.Generator) -> Observation:
        q = self.q
        if traj.n_frames < 2 * q + 1:
            raise ValueError("zu wenige Frames für die zentrale Differenz")
        xt = traj.x + self.sigma_pos * rng.standard_normal(traj.x.shape)
        present = rng.random(traj.x.shape[:2]) >= self.p_miss
        u = (xt[2 * q:] - xt[: -2 * q]) / (2 * q * self.delta)
        valid = present[q:-q] & present[: -2 * q] & present[2 * q:]
        return Observation(traj.t[q:-q].copy(), xt[q:-q], u, valid)
