"""Nachbar-Speed-Korrelation T_nn für kleine Schwärme (7.5).

    c_nn = [ (1/|P1|) Σ_{(i,j)∈P1} δs_i δs_j ] / [ (1/N) Σ_i δs_i² ],   T_nn = c_nn − E_N0[c_nn | N]

P1 = Paare (i, j), in denen j nächster Nachbar von i ist. Der Bias durch die
Nullsummen-Bedingung wird **empirisch** über das Nullmodell N0 unter identischem Abzug
bestimmt (Tabelle über N, erzeugt mit ``scripts/build_nn_bias_table.py``), nicht analytisch.
Fehlt die Tabelle, wird T_nn als NaN gespeichert; c_nn selbst wird immer gespeichert, damit
T_nn nachträglich berechnet werden kann.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..agents.neighbors import nearest_neighbor


def c_nn(x: np.ndarray, ds: np.ndarray, nn: np.ndarray | None = None) -> float:
    """``nn`` (Index des nächsten Nachbarn je Individuum) kann vorab berechnet übergeben werden."""
    if x.shape[0] < 3:
        return float("nan")
    if nn is None:
        nn, _ = nearest_neighbor(x)
    den = float(np.mean(ds**2))
    if den == 0:
        return float("nan")
    return float(np.mean(ds * ds[nn]) / den)


@dataclass(frozen=True)
class NNBiasTable:
    """E_N0[c_nn | N] je Abzugsvariante; Interpolation linear in log N."""

    table: pd.DataFrame  # Spalten: N, variant, mean_c_nn, se_c_nn, n_rep

    @staticmethod
    def load(path: str | Path) -> "NNBiasTable":
        return NNBiasTable(pd.read_parquet(path))

    def bias(self, n: int, variant: str) -> float:
        t = self.table[self.table["variant"] == variant].sort_values("N")
        if t.empty:
            return float("nan")
        ln = np.log(t["N"].to_numpy(dtype=float))
        if not ln[0] <= np.log(n) <= ln[-1]:
            return float("nan")  # keine Extrapolation
        return float(np.interp(np.log(n), ln, t["mean_c_nn"].to_numpy(dtype=float)))


def t_nn(c: float, n: int, variant: str, table: NNBiasTable | None) -> float:
    if table is None or not np.isfinite(c):
        return float("nan")
    return float(c - table.bias(n, variant))
