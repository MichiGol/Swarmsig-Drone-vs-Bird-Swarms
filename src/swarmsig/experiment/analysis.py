"""Auswertung (8.1–8.3): Fehlerraten, ROC, AUC als U-Statistik ohne Diagonale,
Designpunkt-Bootstrap, Wilson-Intervalle.

Konvention (8.1): Entscheidung „Vögel“, falls T > τ, sonst „Drohnengruppe“
(für Statistiken, bei denen Vögel größere Werte erwarten lassen; sonst Vorzeichen tauschen).

    P_FA(τ)   = P(T ≤ τ | B)   (Fehlalarm → unnötige Pistensperrung)
    P_Miss(τ) = P(T > τ | D)   (Drohnengruppe übersehen)

AUC = P(T_B > T_D) für **unabhängige** Begegnungen: Mann–Whitney-U über alle Kreuzpaare
verschiedener Designpunkte (a ≠ b). Arrays ``t_bird[a]``, ``t_drone[a]`` sind nach
Designpunkt ausgerichtet; NaN (z. B. nicht erfasst) wird ausgelassen.
GAM-Landkarte und Sobol-Analyse (8.4) folgen nach P1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def _weighted_greater(b: np.ndarray, wb: np.ndarray, d: np.ndarray, wd: np.ndarray) -> float:
    """Σ_a Σ_b wb_a wd_b [1{b_a > d_b} + ½ 1{b_a = d_b}] in O(M log M)."""
    order = np.argsort(d, kind="stable")
    ds, ws = d[order], wd[order]
    cw = np.concatenate([[0.0], np.cumsum(ws)])
    lo = np.searchsorted(ds, b, side="left")
    hi = np.searchsorted(ds, b, side="right")
    return float(np.sum(wb * (cw[lo] + 0.5 * (cw[hi] - cw[lo]))))


def auc_u(t_bird: np.ndarray, t_drone: np.ndarray, weights: np.ndarray | None = None,
          exclude_diagonal: bool = True) -> float:
    """AUC-Schätzer (8.3); ``weights`` = Vielfachheiten der Designpunkte (Bootstrap)."""
    b = np.asarray(t_bird, dtype=float)
    d = np.asarray(t_drone, dtype=float)
    if b.shape != d.shape:
        raise ValueError("t_bird und t_drone müssen nach Designpunkt ausgerichtet sein")
    w = np.ones_like(b) if weights is None else np.asarray(weights, dtype=float)
    wb = np.where(np.isfinite(b), w, 0.0)
    wd = np.where(np.isfinite(d), w, 0.0)
    bf = np.where(np.isfinite(b), b, 0.0)
    df = np.where(np.isfinite(d), d, 0.0)
    num = _weighted_greater(bf[wb > 0], wb[wb > 0], df[wd > 0], wd[wd > 0])
    den = wb.sum() * wd.sum()
    if exclude_diagonal:
        both = wb * wd  # Paare zwischen Kopien desselben Designpunkts
        h = np.where(bf > df, 1.0, np.where(bf == df, 0.5, 0.0))
        num -= float(np.sum(both * h))
        den -= float(np.sum(both))
    return num / den if den > 0 else float("nan")


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    low: float
    high: float
    n_boot: int
    samples: np.ndarray


def bootstrap_auc(t_bird: np.ndarray, t_drone: np.ndarray, rng: np.random.Generator,
                  n_boot: int = 2000, level: float = 0.95) -> BootstrapResult:
    """Designpunkt-Bootstrap (8.3): Indizes mit Zurücklegen, alle Klassen eines Punkts
    gemeinsam; Perzentil-Intervall."""
    m = len(t_bird)
    est = auc_u(t_bird, t_drone)
    reps = np.empty(n_boot)
    for k in range(n_boot):
        w = np.bincount(rng.integers(0, m, size=m), minlength=m).astype(float)
        reps[k] = auc_u(t_bird, t_drone, weights=w)
    a = (1 - level) / 2
    lo, hi = np.nanquantile(reps, [a, 1 - a])
    return BootstrapResult(est, float(lo), float(hi), n_boot, reps)


def error_rates(t_bird: np.ndarray, t_drone: np.ndarray, tau: float) -> tuple[float, float]:
    """(P_FA, P_Miss) bei Schwelle τ, bedingt auf gültige (erfasste) Läufe."""
    b = np.asarray(t_bird, float)
    d = np.asarray(t_drone, float)
    b, d = b[np.isfinite(b)], d[np.isfinite(d)]
    return float(np.mean(b <= tau)), float(np.mean(d > tau))


def roc_curve(t_bird: np.ndarray, t_drone: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC-Punkte (P_FA(τ), 1 − P_Miss(τ)) über alle Schwellen; Rückgabe (pfa, tpr, tau)."""
    b = np.asarray(t_bird, float)
    d = np.asarray(t_drone, float)
    b, d = b[np.isfinite(b)], d[np.isfinite(d)]
    tau = np.unique(np.concatenate([[-np.inf], b, d, [np.inf]]))
    pfa = np.searchsorted(np.sort(b), tau, side="right") / b.size
    tpr = np.searchsorted(np.sort(d), tau, side="right") / d.size
    return pfa, tpr, tau


def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson-Intervall für eine Wahrscheinlichkeit k/n (8.2)."""
    if n == 0:
        return float("nan"), float("nan")
    z = float(stats.norm.ppf(0.5 + level / 2))
    p = k / n
    den = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return float(center - half), float(center + half)


def required_runs(p: float, half_width: float, level: float = 0.95) -> int:
    """M ≥ z² P(1−P)/ε² (8.2); Beispiel P = 0,05, ε = 0,01 → M ≈ 1825."""
    z = float(stats.norm.ppf(0.5 + level / 2))
    return int(np.ceil(z**2 * p * (1 - p) / half_width**2))
