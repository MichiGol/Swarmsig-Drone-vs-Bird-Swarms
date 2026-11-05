"""Teststatistiken für den idealen Sensor (7.4) und Zeitmittelung über T_obs (7.2).

Pro Frame werden für jede Abzugsvariante (A7, M, A12) C_s(r), C_⊥(r), χ_s(r) und c_nn
berechnet. Hauptstatistiken stammen aus den über die Frames gemittelten Funktionen:

    T_s = ξ_s / L̄_g,   T_⊥ = ξ_⊥ / L̄_g,   T_χ = max_r χ̄_s(r) / ⟨(1/N) Σ δs²⟩,   T_nn = c̄_nn − Bias

Zusätzlich werden Verteilungsmaße der Einzelframe-Werte von ξ_s gespeichert.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..agents.neighbors import nearest_neighbor
from ..state import Observation
from . import correlation as corr
from . import size
from .modes import remove
from .smalln import NNBiasTable, c_nn, t_nn


def _nanmean_rows(arrs: list[np.ndarray]) -> np.ndarray:
    """Mittel über Frames je Abstandsklasse; Klassen ohne Werte bleiben NaN (ohne Warnung)."""
    if not arrs:
        return np.array([])
    stack = _pad(arrs, np.nan)
    cnt = np.isfinite(stack).sum(axis=0)
    tot = np.where(np.isfinite(stack), stack, 0.0).sum(axis=0)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def _pad(arrs: list[np.ndarray], fill: str | float) -> np.ndarray:
    """Stapelt Kurven unterschiedlicher Länge; ``fill='last'`` setzt mit dem letzten Wert fort."""
    m = max((a.size for a in arrs), default=0)
    out = np.full((len(arrs), m), np.nan)
    for k, a in enumerate(arrs):
        out[k, : a.size] = a
        if a.size and fill == "last":
            out[k, a.size:] = a[-1]
    return out


@dataclass
class _VariantAcc:
    C_s: list[np.ndarray] = field(default_factory=list)
    C_perp: list[np.ndarray] = field(default_factory=list)
    chi: list[np.ndarray] = field(default_factory=list)
    var_s: list[float] = field(default_factory=list)
    c_nn: list[float] = field(default_factory=list)
    xi_s_frame: list[float] = field(default_factory=list)
    Omega_abs: list[float] = field(default_factory=list)
    D: list[float] = field(default_factory=list)


class IdealStatistics:
    """Sammelt Frame-Größen und berechnet daraus die Lauf-Statistiken."""

    def __init__(self, variants: list[str], dr: float, *, max_n_all_pairs: int = 5000,
                 n_pair_samples: int = 2_000_000, min_pairs_per_bin: int = 1,
                 rng: np.random.Generator | None = None, nn_table: NNBiasTable | None = None,
                 context_stride: int = 10) -> None:
        self.context_stride = max(1, int(context_stride))  # Hüllen-Größen (L_max, Dichte) nur jedes k-te Frame
        self.variants = list(variants)
        self.dr = float(dr)
        self.max_n_all = int(max_n_all_pairs)
        self.n_samples = int(n_pair_samples)
        self.min_pairs = int(min_pairs_per_bin)
        self.rng = rng
        self.nn_table = nn_table
        self.acc = {v: _VariantAcc() for v in self.variants}
        self.ctx: dict[str, list[float]] = {k: [] for k in
                                            ("L_g", "L_max", "Phi", "mean_speed", "nn_dist", "density", "n_obs")}
        self.sampled_pairs = False

    def add_frame(self, x: np.ndarray, u: np.ndarray) -> None:
        n = x.shape[0]
        if n < 5:
            return
        ps = corr.pairs(x, max_n_all=self.max_n_all, n_samples=self.n_samples, rng=self.rng)
        self.sampled_pairs |= ps.sampled
        bins = corr.bin_index(ps, self.dr)
        nn, nn_d = nearest_neighbor(x)
        for v in self.variants:
            res = remove(v, x, u)
            a = self.acc[v]
            num_s, cnt = corr.binned_products(ps, res.ds, self.dr, bins=bins)
            num_p, _ = corr.binned_products(ps, res.du_perp, self.dr, bins=bins)
            Cs = corr.curve(num_s, cnt, self.min_pairs)
            a.C_s.append(Cs)
            a.C_perp.append(corr.curve(num_p, cnt, self.min_pairs))
            a.chi.append(corr.chi_curve(num_s, ps))
            a.var_s.append(float(np.mean(res.ds**2)))
            a.c_nn.append(c_nn(x, res.ds, nn=nn))
            a.xi_s_frame.append(corr.first_zero(corr.bin_centers(Cs.size, self.dr), Cs)[0])
            a.Omega_abs.append(float(np.linalg.norm(res.Omega)))
            a.D.append(float(res.D))
        k = len(self.ctx["L_g"])
        self.ctx["L_g"].append(size.L_g(x))
        self.ctx["Phi"].append(size.polarization(u))
        self.ctx["mean_speed"].append(size.mean_speed(u))
        self.ctx["nn_dist"].append(float(np.mean(nn_d)))
        self.ctx["n_obs"].append(float(n))
        if k % self.context_stride == 0:
            self.ctx["L_max"].append(size.L_max(x))
            self.ctx["density"].append(size.density(x))

    def add_observation(self, obs: Observation, stride: int = 1) -> None:
        for k in range(0, obs.n_frames, max(1, int(stride))):
            x, u = obs.frame(k)
            self.add_frame(x, u)

    @property
    def n_frames(self) -> int:
        return len(self.ctx["L_g"])

    def finalize(self, keep_curves: bool = False) -> tuple[dict[str, float | str | int], dict[str, np.ndarray]]:
        row: dict[str, float | str | int] = {}
        curves: dict[str, np.ndarray] = {}
        nf = self.n_frames
        row["n_frames"] = nf
        for k, vals in self.ctx.items():
            arr = np.asarray(vals, dtype=float)
            row[k] = float(np.mean(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else float("nan")
        row["pairs_sampled"] = bool(self.sampled_pairs)
        if nf == 0:
            return row, curves
        Lg = row["L_g"]
        n_mean = int(round(float(row["n_obs"])))
        for v in self.variants:
            a = self.acc[v]
            Cs = _nanmean_rows(a.C_s)
            Cp = _nanmean_rows(a.C_perp)
            chi = np.mean(_pad(a.chi, "last"), axis=0)
            r = corr.bin_centers(Cs.size, self.dr)
            xi_s, f_s = corr.first_zero(r, Cs)
            xi_p, f_p = corr.first_zero(corr.bin_centers(Cp.size, self.dr), Cp)
            var_s = float(np.mean(a.var_s))
            chi_max = float(max(0.0, np.nanmax(chi))) if chi.size else float("nan")
            cn = float(np.nanmean(a.c_nn)) if np.isfinite(a.c_nn).any() else float("nan")
            row[f"xi_s_{v}"] = xi_s
            row[f"xi_perp_{v}"] = xi_p
            row[f"xi_s_flag_{v}"] = f_s
            row[f"xi_perp_flag_{v}"] = f_p
            row[f"chi_s_max_{v}"] = chi_max
            row[f"var_s_{v}"] = var_s
            row[f"T_s_{v}"] = xi_s / Lg if Lg > 0 else float("nan")
            row[f"T_perp_{v}"] = xi_p / Lg if Lg > 0 else float("nan")
            row[f"T_chi_{v}"] = chi_max / var_s if var_s > 0 else float("nan")
            row[f"c_nn_{v}"] = cn
            row[f"T_nn_{v}"] = t_nn(cn, n_mean, v, self.nn_table)
            xf = np.asarray(a.xi_s_frame, dtype=float)
            row[f"xi_s_frame_median_{v}"] = float(np.nanmedian(xf)) if np.isfinite(xf).any() else float("nan")
            row[f"xi_s_frame_iqr_{v}"] = (float(np.subtract(*np.nanpercentile(xf, [75, 25])))
                                         if np.isfinite(xf).any() else float("nan"))
            if v == "A7":
                row["Omega_abs"] = float(np.mean(a.Omega_abs))
                row["dilatation_D"] = float(np.mean(a.D))
            if keep_curves:
                curves[f"r_{v}"] = r
                curves[f"C_s_{v}"] = Cs
                curves[f"C_perp_{v}"] = Cp
                curves[f"chi_s_{v}"] = chi
        return row, curves
