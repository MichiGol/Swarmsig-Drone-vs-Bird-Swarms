"""Vögel als Fehlalarmquelle (O21 Stufe 4): Säulendichte aus Wetterradar-Profilen und unbewertbare Vogelspuren.

Ein Vogel in der Entfernung r ist unbewertbar, wenn das Radar ihn erkennt (r ≤ R_V) und kein Merkmal klären kann, ob
dort eine Drohne fliegt (r > R_U). Bei horizontal gleichverteilten Vögeln (Breitfrontzug) mit Säulendichte n gilt:

* gleichzeitig unbewertbare Vögel im Ring R_U < r ≤ R_V:  N = n · π · max(R_V² − R_U², 0)
* nie bewertbare neue Spuren pro Zeit (gerade Bahnen mit Abstand R_U ≤ |y| < R_V vom Radar):  Φ = n · v · 2 · max(R_V − R_U, 0)
"""

from __future__ import annotations

import numpy as np

BIN_M = 200.0  # Höhenstufe der Aloft-Profile
DEFAULT_SPEED = 12.0  # m/s, typische Grundgeschwindigkeit ziehender Vögel


def column_density(dens: np.ndarray, top_m: float, fill_lowest: bool = True) -> np.ndarray:
    """Säulendichte in Vögeln/km² im Höhenfenster 0 … top_m.

    ``dens`` hat je Zeile die Dichten (Vögel/km³) der Stufen 0–200, 200–400, … m. Die unterste Stufe ist im Wetterradar
    nicht beobachtbar und wird mit der zweiten gefüllt (``fill_lowest``) oder gleich 0 gesetzt. Fehlt eine beobachtete
    Stufe im Fenster (NaN), ist das Ergebnis NaN.
    """
    dens = np.atleast_2d(np.asarray(dens, dtype=float))
    k = int(np.ceil(top_m / BIN_M))
    if k < 2 or k > dens.shape[1]:
        raise ValueError("Höhenfenster muss mindestens zwei und höchstens alle Stufen umfassen")
    observed = dens[:, 1:k]
    thickness = np.full(k - 1, BIN_M / 1e3)
    thickness[-1] = (top_m - (k - 1) * BIN_M) / 1e3  # angeschnittene oberste Stufe
    lowest = observed[:, 0] if fill_lowest else np.zeros(len(dens))
    return lowest * BIN_M / 1e3 + observed @ thickness


def mean_speed(dens: np.ndarray, ff: np.ndarray, top_m: float, default: float = DEFAULT_SPEED) -> np.ndarray:
    """Dichtegewichtete Grundgeschwindigkeit (m/s) der beobachteten Stufen im Fenster; fehlt sie, ``default``."""
    k = int(np.ceil(top_m / BIN_M))
    d = np.nan_to_num(np.atleast_2d(np.asarray(dens, dtype=float))[:, 1:k])
    f = np.atleast_2d(np.asarray(ff, dtype=float))[:, 1:k]
    w = np.where(np.isfinite(f), d, 0.0)
    num = np.sum(w * np.nan_to_num(f), axis=1)
    den = np.sum(w, axis=1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), default)


def ring_count(n_col: np.ndarray, r_v_km: np.ndarray, r_u_km: np.ndarray) -> np.ndarray:
    """Gleichzeitig unbewertbare Vögel N = n · π · max(R_V² − R_U², 0)."""
    return np.asarray(n_col) * np.pi * np.maximum(np.asarray(r_v_km) ** 2 - np.asarray(r_u_km) ** 2, 0.0)


def never_classified_rate(n_col: np.ndarray, v_kmh: np.ndarray, r_v_km: np.ndarray, r_u_km: np.ndarray) -> np.ndarray:
    """Neue Spuren pro Stunde, die den Erkennungskreis durchqueren, ohne je näher als R_U zu kommen."""
    return np.asarray(n_col) * np.asarray(v_kmh) * 2.0 * np.maximum(np.asarray(r_v_km) - np.asarray(r_u_km), 0.0)
