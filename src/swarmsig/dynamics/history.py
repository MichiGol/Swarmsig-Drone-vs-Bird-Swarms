"""Ringpuffer vergangener Zustände für Latenz (Drohnen, 4.7) und Verzögerungen (B3, 4.4).

Verzögerungen werden in ganzen Integrationsschritten angegeben (``lag = round(τ/Δt)``,
Rundungsfehler protokolliert der Aufrufer). Ist der Puffer noch nicht so weit gefüllt,
wird der älteste vorhandene Eintrag verwendet (Beginn der Einschwingphase).
"""

from __future__ import annotations

import numpy as np


class History:
    """Speichert die letzten ``capacity`` Einträge von bis zu mehreren (N, 3)-Feldern."""

    def __init__(self, capacity: int, n: int, fields: tuple[str, ...] = ("x", "u")) -> None:
        if capacity < 1:
            raise ValueError("capacity muss >= 1 sein")
        self.capacity = int(capacity)
        self.n = int(n)
        self._buf = {f: np.zeros((self.capacity, self.n, 3)) for f in fields}
        self._head = -1  # Index des neuesten Eintrags
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def push(self, **arrays: np.ndarray) -> None:
        self._head = (self._head + 1) % self.capacity
        for f, buf in self._buf.items():
            buf[self._head] = arrays[f]
        self._count = min(self._count + 1, self.capacity)

    def lagged(self, field: str, lags: np.ndarray | int, idx: np.ndarray) -> np.ndarray:
        """Werte von ``field`` für Individuen ``idx`` vor ``lags`` Schritten (je Eintrag).

        ``lags`` und ``idx`` werden elementweise kombiniert (Form (E,)). ``lags = 0`` ist
        der neueste Eintrag.
        """
        if self._count == 0:
            raise RuntimeError("History ist leer")
        lags = np.minimum(np.asarray(lags, dtype=np.int64), self._count - 1)
        rows = (self._head - lags) % self.capacity
        return self._buf[field][rows, idx]

    def latest(self, field: str) -> np.ndarray:
        return self._buf[field][self._head]
