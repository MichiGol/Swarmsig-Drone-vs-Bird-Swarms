"""Korrelationsfunktionen, Korrelationslänge und Suszeptibilität (7.2).

    C(r) = Σ_{i≠j} a_i·a_j Θ_r(r_ij) / Σ_{i≠j} Θ_r(r_ij),  Θ_r = 1{r ≤ r_ij < r + Δr}
    χ_s(r) = (1/N) Σ_i Σ_{j≠i} δs_i δs_j 1{r_ij < r}

Summen über ungeordnete Paare (i < j) liefern dasselbe C(r); für χ zählt jedes Paar doppelt.
Für große N (über ``max_n_all_pairs``) wird eine Zufallsstichprobe von Paaren verwendet
(7.2; der Stichprobenfehler ist zu dokumentieren).

Korrelationslänge ξ: erste Nullstelle von C(r), lineare Interpolation zwischen
Klassenmitten. Sonderfälle werden mit einem Flag zurückgegeben (siehe :func:`first_zero`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class PairSet:
    i: np.ndarray
    j: np.ndarray
    d: np.ndarray
    n: int
    sampled: bool

    @property
    def scale(self) -> float:
        """Faktor von der Stichprobe auf alle ungeordneten Paare (1 bei Vollzählung)."""
        total = self.n * (self.n - 1) / 2
        return total / self.i.size if self.sampled and self.i.size else 1.0


@lru_cache(maxsize=32)
def _triu(n: int) -> tuple[np.ndarray, np.ndarray]:
    i, j = np.triu_indices(n, k=1)
    i.setflags(write=False)
    j.setflags(write=False)
    return i, j


def pairs(x: np.ndarray, *, max_n_all: int = 5000, n_samples: int = 2_000_000,
          rng: np.random.Generator | None = None) -> PairSet:
    n = x.shape[0]
    if n <= max_n_all:
        i, j = _triu(n)
        sampled = False
    else:
        if rng is None:
            raise ValueError("Paar-Stichprobe braucht einen Generator")
        i = rng.integers(0, n, size=n_samples)
        j = rng.integers(0, n - 1, size=n_samples)
        j = j + (j >= i)  # j ≠ i, gleichverteilt
        sampled = True
    d = np.linalg.norm(x[i] - x[j], axis=1)
    return PairSet(i.astype(np.int64), j.astype(np.int64), d, n, sampled)


def bin_index(ps: PairSet, dr: float) -> np.ndarray:
    return np.floor(ps.d / dr).astype(np.int64)


def binned_products(ps: PairSet, a: np.ndarray, dr: float,
                    bins: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Summe der Produkte a_i·a_j und Paarzahl je Abstandsklasse [bΔr, (b+1)Δr)."""
    b = bin_index(ps, dr) if bins is None else bins
    nb = int(b.max()) + 1 if b.size else 0
    prod = a[ps.i] * a[ps.j] if a.ndim == 1 else np.einsum("ij,ij->i", a[ps.i], a[ps.j])
    num = np.bincount(b, weights=prod, minlength=nb)
    cnt = np.bincount(b, minlength=nb).astype(float)
    return num, cnt


def curve(num: np.ndarray, cnt: np.ndarray, min_pairs: int = 1) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt >= min_pairs, num / cnt, np.nan)


def chi_curve(num: np.ndarray, ps: PairSet) -> np.ndarray:
    """χ(r) an den oberen Klassengrenzen (b+1)Δr: (2/N)·kumulierte Paarsumme."""
    return 2.0 * ps.scale * np.cumsum(num) / ps.n


def bin_centers(nbins: int, dr: float) -> np.ndarray:
    return (np.arange(nbins) + 0.5) * dr


def first_zero(r: np.ndarray, c: np.ndarray) -> tuple[float, str]:
    """Erste Nullstelle von c(r) (lineare Interpolation). NaN-Klassen werden übersprungen.

    Flags: ``ok``; ``nonpositive_start`` (schon die erste besetzte Klasse ≤ 0: ξ = deren
    Mitte, also die kleinste auflösbare Länge); ``no_crossing`` (keine Nullstelle im
    berechneten Bereich: ξ = NaN); ``empty``.
    """
    ok = np.isfinite(c)
    if not ok.any():
        return float("nan"), "empty"
    rv, cv = r[ok], c[ok]
    if cv[0] <= 0:
        return float(rv[0]), "nonpositive_start"
    k = np.flatnonzero(cv <= 0)
    if k.size == 0:
        return float("nan"), "no_crossing"
    k0 = int(k[0])
    r1, r2, c1, c2 = rv[k0 - 1], rv[k0], cv[k0 - 1], cv[k0]
    return float(r1 + c1 * (r2 - r1) / (c1 - c2)), "ok"


def chi_total(ds: np.ndarray) -> float:
    """χ_s(∞) = (1/N)[(Σδs)² − Σδs²]; bei Σδs = 0 gleich −(1/N)Σδs² (Nullstellen-Argument 7.2)."""
    n = ds.size
    return float((ds.sum() ** 2 - np.sum(ds**2)) / n)
