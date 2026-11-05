"""Schwarmgröße, Dichte und Kontextgrößen (7.3, 7.6). Dichte und d_nn sind nur
Kovariaten, nie Klassifikationsmerkmale (Regel 6)."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ..agents.neighbors import nearest_neighbor


def radius_of_gyration(x: np.ndarray) -> float:
    r = x - x.mean(axis=0)
    return float(np.sqrt(np.mean(np.einsum("ij,ij->i", r, r))))


def L_g(x: np.ndarray) -> float:
    """L_g = 2 √(5/3) R_g (für eine homogene Kugel exakt der Durchmesser)."""
    return float(2.0 * np.sqrt(5.0 / 3.0) * radius_of_gyration(x))


def _hull(x: np.ndarray) -> ConvexHull | None:
    if x.shape[0] < 4:
        return None
    try:
        return ConvexHull(x)
    except QhullError:
        return None


def L_max(x: np.ndarray) -> float:
    """Größter Paarabstand (ausreißerempfindlich), über die Ecken der konvexen Hülle."""
    h = _hull(x)
    pts = x if h is None else x[h.vertices]
    if pts.shape[0] < 2:
        return 0.0
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    return float(d.max())


def nn_distance_mean(x: np.ndarray) -> float:
    _, d = nearest_neighbor(x)
    return float(np.nanmean(d)) if d.size else float("nan")


def density(x: np.ndarray) -> float:
    """Anzahldichte N / Volumen der konvexen Hülle (D: Definition; Kovariate)."""
    h = _hull(x)
    return float(x.shape[0] / h.volume) if h is not None and h.volume > 0 else float("nan")


def n_clusters(x: np.ndarray, link: float) -> int:
    """Zahl der Teilgruppen (Zusammenhangskomponenten bei Kanten bis ``link``), Fragmentierung 14.23."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = x.shape[0]
    pairs = cKDTree(x).query_pairs(link, output_type="ndarray")
    if len(pairs) == 0:
        return n
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return int(connected_components(g, directed=False)[0])


def polarization(u: np.ndarray) -> float:
    s = np.linalg.norm(u, axis=1)
    ok = s > 0
    if not ok.any():
        return float("nan")
    return float(np.linalg.norm((u[ok] / s[ok, None]).sum(axis=0)) / u.shape[0])


def mean_speed(u: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(u, axis=1)))
