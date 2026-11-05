"""Kubische Hermite-Interpolation auf Frame-Zeitpunkte (4.9).

Simulations- und Sensorzeit sind entkoppelt: Frames t_k müssen nicht auf dem
Integrationsraster liegen. Zwischen zwei Schritten (t_n, x_n, u_n) und (t_{n+1}, x_{n+1},
u_{n+1}) wird mit dem kubischen Hermite-Polynom (Stützwerte x, Steigungen u) interpoliert;
die Geschwindigkeit ist dessen Ableitung.
"""

from __future__ import annotations

import numpy as np

from ..state import SwarmState, Trajectory


def hermite(t0: float, x0: np.ndarray, u0: np.ndarray, t1: float, x1: np.ndarray,
            u1: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    h = t1 - t0
    s = (t - t0) / h
    s2, s3 = s * s, s * s * s
    h00, h10, h01, h11 = 2 * s3 - 3 * s2 + 1, s3 - 2 * s2 + s, -2 * s3 + 3 * s2, s3 - s2
    d00, d10, d01, d11 = 6 * s2 - 6 * s, 3 * s2 - 4 * s + 1, -6 * s2 + 6 * s, 3 * s2 - 2 * s
    x = h00 * x0 + h10 * h * u0 + h01 * x1 + h11 * h * u1
    u = (d00 * x0 + d10 * h * u0 + d01 * x1 + d11 * h * u1) / h
    return x, u


class FrameRecorder:
    """Callback für :func:`~swarmsig.dynamics.integrator.simulate`: interpoliert alle
    Frame-Zeitpunkte, die in einen Integrationsschritt fallen."""

    def __init__(self, frame_times: np.ndarray, n: int, tol: float = 1e-9) -> None:
        self.times = np.sort(np.asarray(frame_times, dtype=float))
        self.x = np.empty((self.times.size, n, 3))
        self.u = np.empty((self.times.size, n, 3))
        self.k = 0
        self.tol = tol

    @property
    def done(self) -> bool:
        return self.k >= self.times.size

    def __call__(self, prev: SwarmState, new: SwarmState, step: int) -> None:  # noqa: ARG002
        while self.k < self.times.size and self.times[self.k] <= new.t + self.tol:
            t = self.times[self.k]
            if t < prev.t - self.tol:
                raise RuntimeError(f"Frame t={t} liegt vor dem Aufzeichnungsbeginn t={prev.t}")
            t = min(max(t, prev.t), new.t)
            self.x[self.k], self.u[self.k] = hermite(prev.t, prev.x, prev.u, new.t, new.x, new.u, t)
            self.k += 1

    def trajectory(self) -> Trajectory:
        if not self.done:
            raise RuntimeError(f"nur {self.k} von {self.times.size} Frames aufgezeichnet")
        return Trajectory(self.times.copy(), self.x, self.u)
