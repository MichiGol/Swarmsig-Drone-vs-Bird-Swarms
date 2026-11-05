"""Anfangsbedingungen, Einschwingen und Stationaritätsprüfung (4.9).

1. Positionen gleichverteilt in einem Ellipsoid; Volumen aus der Zieldichte der Klasse
   (Dichte 1/spacing³, **ANNAHME**), Halbachsenverhältnis ``aspect`` (**ANNAHME**, Config),
   längste Achse längs der Flugrichtung, kürzeste vertikal.
2. Geschwindigkeiten s0·ê + kleines Rauschen.
3. Einschwingzeit T_burn wird verworfen; Stationarität: Steigung einer linearen Regression
   der Kontextgrößen (Φ, s̄, L, d_nn) in den letzten 20 % der Einschwingzeit nicht
   signifikant.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def ellipsoid_positions(n: int, center: np.ndarray, spacing: float,
                        aspect: tuple[float, float, float], heading: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    aspect_a = np.asarray(aspect, dtype=float)
    volume = n * spacing**3
    s = (3.0 * volume / (4.0 * np.pi * np.prod(aspect_a))) ** (1.0 / 3.0)
    axes = s * aspect_a
    z = rng.standard_normal((n, 3))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    rad = rng.uniform(0.0, 1.0, size=n) ** (1.0 / 3.0)
    local = z * rad[:, None] * axes[None, :]
    # lokale Achsen: 0 = Flugrichtung (horizontal), 1 = horizontal quer, 2 = vertikal
    e0 = np.array([heading[0], heading[1], 0.0])
    nrm = np.linalg.norm(e0)
    e0 = e0 / nrm if nrm > 0 else np.array([0.0, 1.0, 0.0])
    e2 = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(e2, e0)
    rot = np.column_stack([e0, e1, e2])
    return np.asarray(center, dtype=float)[None, :] + local @ rot.T


def initial_velocities(n: int, s0: float, direction: np.ndarray, rel_noise: float,
                       rng: np.random.Generator) -> np.ndarray:
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    return s0 * d[None, :] + rel_noise * s0 * rng.standard_normal((n, 3))


def stationarity_ok(t: np.ndarray, series: dict[str, np.ndarray], window_fraction: float,
                    alpha: float) -> tuple[bool, dict[str, float]]:
    """Prüft je Reihe, ob die Regressionssteigung im Endfenster nicht signifikant ist.

    Rückgabe: (alle stationär?, p-Werte je Reihe). Hinweis: Autokorrelation macht den
    naiven t-Test antikonservativ; das Flag ist eine Diagnose, kein Ausschlusskriterium.
    """
    t = np.asarray(t, dtype=float)
    if t.size < 4:
        return False, {}
    t_start = t[-1] - window_fraction * (t[-1] - t[0])
    m = t >= t_start
    pvals: dict[str, float] = {}
    for k, y in series.items():
        y = np.asarray(y, dtype=float)[m]
        tt = t[m]
        ok = np.isfinite(y)
        if ok.sum() < 4 or np.ptp(y[ok]) == 0:
            pvals[k] = 1.0
            continue
        pvals[k] = float(stats.linregress(tt[ok], y[ok]).pvalue)
    return all(p > alpha for p in pvals.values()), pvals
