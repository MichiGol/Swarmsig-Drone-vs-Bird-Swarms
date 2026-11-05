"""Nachbarsuche (4.9): topologisch (n_c nächste) oder metrisch (Radius r_c) über cKDTree.

Ergebnis ist ein gerichteter Graph als Kantenliste ``(src, dst)``: Kante ``e`` bedeutet
„Individuum ``src[e]`` nimmt Nachbar ``dst[e]`` wahr“. Kanten sind nach ``src`` sortiert.
Alle Summen über Nachbarn laufen vektorisiert über ``np.bincount`` (Regel: keine
Python-Schleifen über Agenten).

Die optionale Wiederverwendung der Listen über mehrere Schritte (Verlet-Prinzip) ist noch
nicht umgesetzt; derzeit wird jede Liste neu aufgebaut.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class NeighborGraph:
    n: int
    src: np.ndarray  # (E,) int64
    dst: np.ndarray  # (E,) int64

    @property
    def n_edges(self) -> int:
        return int(self.src.shape[0])

    def counts(self) -> np.ndarray:
        return np.bincount(self.src, minlength=self.n)

    def segment_sum(self, values: np.ndarray) -> np.ndarray:
        """Summe von Kantenwerten je Empfänger; ``values`` hat Form (E,) oder (E, d)."""
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            return np.bincount(self.src, weights=values, minlength=self.n)
        out = np.empty((self.n, values.shape[1]))
        for c in range(values.shape[1]):
            out[:, c] = np.bincount(self.src, weights=values[:, c], minlength=self.n)
        return out

    def segment_mean(self, values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """(Gewichtetes) Mittel je Empfänger; Individuen ohne Nachbarn erhalten 0."""
        values = np.asarray(values, dtype=float)
        if weights is None:
            num = self.segment_sum(values)
            den = self.counts().astype(float)
        else:
            w = np.asarray(weights, dtype=float)
            num = self.segment_sum(values * (w[:, None] if values.ndim == 2 else w))
            den = self.segment_sum(w)
        den_safe = np.where(den > 0, den, 1.0)
        return num / (den_safe[:, None] if num.ndim == 2 else den_safe)

    def as_sets(self) -> list[set[int]]:
        sets: list[set[int]] = [set() for _ in range(self.n)]
        for i, j in zip(self.src.tolist(), self.dst.tolist()):
            sets[i].add(j)
        return sets


def _sorted_graph(n: int, src: np.ndarray, dst: np.ndarray) -> NeighborGraph:
    order = np.lexsort((dst, src))
    return NeighborGraph(n, src[order].astype(np.int64), dst[order].astype(np.int64))


def topological(x: np.ndarray, n_c: int, tree: cKDTree | None = None) -> NeighborGraph:
    """Die ``n_c`` nächsten Nachbarn je Individuum (ohne sich selbst)."""
    n = x.shape[0]
    k = min(int(n_c), n - 1)
    if k <= 0:
        return NeighborGraph(n, np.zeros(0, np.int64), np.zeros(0, np.int64))
    tree = cKDTree(x) if tree is None else tree
    _, idx = tree.query(x, k=k + 1)
    idx = np.asarray(idx).reshape(n, k + 1)
    self_mask = idx == np.arange(n)[:, None]
    # Bei Duplikaten kann der eigene Index fehlen: dann die letzte Spalte verwerfen.
    no_self = ~self_mask.any(axis=1)
    self_mask[no_self, -1] = True
    # Genau ein Eintrag je Zeile entfernen (falls der eigene Index mehrfach vorkäme).
    first = np.argmax(self_mask, axis=1)
    keep = np.ones_like(self_mask)
    keep[np.arange(n), first] = False
    dst = idx[keep].reshape(n, k)
    src = np.repeat(np.arange(n), k)
    return _sorted_graph(n, src, dst.ravel())


def metric(x: np.ndarray, r_c: float, tree: cKDTree | None = None) -> NeighborGraph:
    """Alle Nachbarn mit Abstand < ``r_c`` (ohne sich selbst), beidseitig gerichtet."""
    n = x.shape[0]
    tree = cKDTree(x) if tree is None else tree
    pairs = tree.query_pairs(float(r_c), output_type="ndarray")
    if pairs.size == 0:
        return NeighborGraph(n, np.zeros(0, np.int64), np.zeros(0, np.int64))
    i, j = pairs[:, 0], pairs[:, 1]
    # query_pairs nutzt <= r; die Definition verlangt < r_c.
    d = np.linalg.norm(x[j] - x[i], axis=1)
    m = d < r_c
    i, j = i[m], j[m]
    return _sorted_graph(n, np.concatenate([i, j]), np.concatenate([j, i]))


def build(x: np.ndarray, kind: str, *, n_c: int | None = None, r_c: float | None = None,
          max_neighbors: int | None = None) -> NeighborGraph:
    """Nachbargraph nach Art ``kind`` ("topological" | "metric").

    ``max_neighbors`` begrenzt bei metrischer Nachbarschaft zusätzlich auf die nächsten
    ``max_neighbors`` innerhalb von ``r_c`` (wie in SwarmLab, optional).
    """
    if kind == "topological":
        if n_c is None:
            raise ValueError("topologische Nachbarschaft braucht n_c")
        return topological(x, n_c)
    if kind == "metric":
        if r_c is None:
            raise ValueError("metrische Nachbarschaft braucht r_c")
        g = metric(x, r_c)
        if max_neighbors is not None:
            g = limit_nearest(g, x, int(max_neighbors))
        return g
    raise ValueError(f"unbekannte Nachbarschaft {kind!r}")


def limit_nearest(g: NeighborGraph, x: np.ndarray, k: int) -> NeighborGraph:
    """Behält je Empfänger nur die ``k`` nächsten Kanten."""
    if g.n_edges == 0:
        return g
    d = np.linalg.norm(x[g.dst] - x[g.src], axis=1)
    order = np.lexsort((d, g.src))  # nach src, innerhalb nach Abstand
    src, dst = g.src[order], g.dst[order]
    starts = np.searchsorted(src, np.arange(g.n))
    rank = np.arange(src.size) - starts[src]
    m = rank < k
    return _sorted_graph(g.n, src[m], dst[m])


def nearest_neighbor(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index und Abstand des nächsten Nachbarn je Individuum (für d_nn und T_nn)."""
    n = x.shape[0]
    if n < 2:
        return np.zeros(n, np.int64), np.full(n, np.nan)
    tree = cKDTree(x)
    d, idx = tree.query(x, k=2)
    # Spalte 0 ist im Normalfall das Individuum selbst; bei Duplikaten korrigieren.
    self_first = idx[:, 0] == np.arange(n)
    nn = np.where(self_first, idx[:, 1], idx[:, 0])
    dist = np.where(self_first, d[:, 1], d[:, 0])
    return nn.astype(np.int64), dist
