"""Einheitliche Windübertragung (4.1), für alle Klassen gleich aufgebaut.

    u_i = v_i + w(x_i, t) − c_i ŵ_i,     dŵ_i = (w(x_i, t) − ŵ_i) / τ_est,i dt

Diskretisierung wie in 4.9: ŵ wird mit dem Wind am alten Ort/zur alten Zeit
fortgeschrieben, u mit dem Wind am alten Ort zur neuen Zeit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindTransfer:
    c_comp: np.ndarray | float  # Kompensationsanteil (0 = volle Drift, 1 = volle Kompensation)
    tau_est: np.ndarray | float  # Zeitkonstante des Windschätzers in s

    def _col(self, a: np.ndarray | float, n: int) -> np.ndarray:
        return np.broadcast_to(np.asarray(a, dtype=float).reshape(-1, 1), (n, 1))

    def apply(self, v_new: np.ndarray, w_old: np.ndarray, w_new: np.ndarray,
              w_hat: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Liefert (u^{n+1}, ŵ^{n+1}).

        w_old = w(x^n, t_n) für den Schätzer, w_new = w(x^n, t_{n+1}) für u.
        """
        n = v_new.shape[0]
        tau = self._col(self.tau_est, n)
        c = self._col(self.c_comp, n)
        w_hat_new = w_hat + (dt / tau) * (w_old - w_hat)
        return v_new + w_new - c * w_hat_new, w_hat_new
