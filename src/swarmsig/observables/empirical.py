"""Hilfen für echte Schwarmdaten (O3a): Hauptschwarm je Bild und Bias-Korrektur je Bild.

Echte Daten haben wechselnde Individuenzahlen, Nachzügler und teils nur horizontale Positionen.
Die N0-Tabelle (7.5) gilt nur für 3D und N = 10–100. Deshalb wird E[c_nn] unter unabhängigen
Schwankungen je Bild auf *denselben* Positionen geschätzt, mit gleichem Abzug und in der Dimension
der Daten:

    T_nn = Mittel_Bilder(c_nn) − Mittel_Bilder(E_N0[c_nn | x])
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..agents.neighbors import nearest_neighbor
from .modes import remove
from .smalln import c_nn


def main_cluster(x: np.ndarray, factor: float = 5.0) -> np.ndarray:
    """Maske der größten Gruppe bei Einzelverknüpfung mit Schwelle ``factor`` × medianer Nachbarabstand."""
    n = x.shape[0]
    if n < 3:
        return np.ones(n, bool)
    _, d = nearest_neighbor(x)
    thr = factor * float(np.median(d))
    if not np.isfinite(thr) or thr <= 0:
        return np.ones(n, bool)
    pairs = cKDTree(x).query_pairs(thr, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels == np.bincount(labels).argmax()


def null_bias(x: np.ndarray, U: np.ndarray, variants: Iterable[str], rng: np.random.Generator, *,
              reps: int = 20, dim: int = 3, rel_sigma: float = 0.05) -> dict[str, float]:
    """E[c_nn] unter unabhängigen isotropen Schwankungen (σ = rel_sigma·|U|) auf den Positionen ``x``.

    ``dim = 2``: Schwankungen nur in x und y (für horizontale Daten mit z = 0).
    """
    variants = list(variants)
    n = x.shape[0]
    U = np.asarray(U, float)
    sigma = rel_sigma * float(np.linalg.norm(U))
    nn, _ = nearest_neighbor(x)
    vals: dict[str, list[float]] = {v: [] for v in variants}
    for _ in range(reps):
        z = np.zeros((n, 3))
        z[:, :dim] = rng.standard_normal((n, dim))
        u = U[None, :] + sigma * z
        for v in variants:
            vals[v].append(c_nn(x, remove(v, x, u).ds, nn=nn))
    return {v: float(np.nanmean(a)) for v, a in vals.items()}
