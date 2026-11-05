"""O4 (explorativ): Nachbarkorrelation T_nn an echten Drohnenschwarm-Flügen (Vásárhelyi et al. 2018).

Ablauf je Flug (4 Flüge, je ≈ 30 Drohnen, GPS-Positionen mit ≈ 5 Hz):

1. **Zeitstempel korrigieren.** Die Positionen sind gleichmäßig abgetastet (Δ ≈ 0,2 s), die
   protokollierten Zeitstempel schwanken aber um einige Hundertstelsekunden. Je Drohne wird die
   Probennummer (mit Lücken) bestimmt und die Zeit als gleitender Median des Versatzes plus
   Δ·Probennummer geglättet.
2. **Schwarmflugphase:** Drohne fliegt mit 0,5–1,5 × Zielgeschwindigkeit und mindestens 5 m über
   ihrer Starthöhe.
3. **Gemeinsames Raster** (0,2 s) durch lineare Interpolation; Fenster von 10 s (wie T_obs der
   Simulation), schrittweise um 10 s versetzt; nur Drohnen, die im ganzen Fenster fliegen (≥ 20).
4. **Kenngrößen mit dem Code der Simulation** (`IdealStatistics`): Geschwindigkeit per zentraler
   Differenz über ±q Rasterpunkte, Pflichtabzug A7 (sowie M, A12), biaskorrigiertes T_nn.

Vergleich mit der Simulation bei N = 30 (P3-Hauptlauf mit und ohne Wind).

    python scripts/analyze_vasarhelyi.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from analyze_p1 import md  # noqa: E402
from report_figures import COLORS, INK, INK2, MUTED, SURFACE, style  # noqa: E402
from swarmsig.config import load_config  # noqa: E402
from swarmsig.io import read_runs  # noqa: E402
from swarmsig.observables.smalln import NNBiasTable  # noqa: E402
from swarmsig.observables.statistics import IdealStatistics  # noqa: E402
from swarmsig.state import Observation  # noqa: E402

TARGET = {"4mps_diagonal": 4.0, "6mps_circular": 6.0, "6mps_obstacles": 6.0, "8mps_circular": 8.0}
GRID = 0.2  # s
WIN = 10.0  # s


def correct_times(t: np.ndarray) -> np.ndarray:
    """Geglättete Zeit je Probe: Probennummer mit Lücken, Versatz als gleitender Median."""
    dt = np.diff(t)
    step = float(np.median(dt))
    n = np.maximum(1, np.rint(dt / step)).astype(int)
    k = np.concatenate([[0], np.cumsum(n)])
    off = pd.Series(t - step * k).rolling(51, center=True, min_periods=1).median().to_numpy()
    return off + step * k


def prepare(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Je Flug: Rasterzeiten (K,), Positionen (K, D, 3) mit NaN außerhalb, Flugmaske (K, D)."""
    out = {}
    for flight, g in df.groupby("flight"):
        target = TARGET[flight]
        drones = sorted(g["drone"].unique())
        t_end = g["t"].max()
        grid = np.arange(0.0, t_end, GRID)
        X = np.full((grid.size, len(drones), 3), np.nan)
        F = np.zeros((grid.size, len(drones)), bool)
        for j, dr in enumerate(drones):
            d = g[g["drone"] == dr].sort_values("t")
            tc = correct_times(d["t"].to_numpy())
            xyz = d[["x", "y", "z"]].to_numpy()
            inside = (grid >= tc[0]) & (grid <= tc[-1])
            for c in range(3):
                X[inside, j, c] = np.interp(grid[inside], tc, xyz[:, c])
            v = np.full(grid.size, np.nan)
            v[1:-1] = np.hypot(X[2:, j, 0] - X[:-2, j, 0], X[2:, j, 1] - X[:-2, j, 1]) / (2 * GRID)
            z0 = np.nanquantile(X[:, j, 2], 0.05)
            F[:, j] = (v > 0.5 * target) & (v < 1.5 * target) & (X[:, j, 2] > z0 + 5.0)
        out[flight] = (grid, X, F)
    return out


def windows(grid, X, F, q: int, nn_table, variants, dr):
    """Statistiken je 10-s-Fenster (Drohnen, die im ganzen Fenster samt Rand fliegen)."""
    n_win = int(round(WIN / GRID))  # 50 Intervalle = 51 Bilder
    rows = []
    k0 = q
    while k0 + n_win + q < grid.size:
        sl = slice(k0 - q, k0 + n_win + q + 1)
        ok = F[sl].all(axis=0) & np.isfinite(X[sl]).all(axis=(0, 2))
        if ok.sum() >= 20:
            Xw = X[sl][:, ok]
            u = (Xw[2 * q:] - Xw[:-2 * q]) / (2 * q * GRID)
            xo = Xw[q:-q]
            obs = Observation(grid[sl][q:-q], xo, u, np.ones(xo.shape[:2], bool))
            st = IdealStatistics(variants, dr, nn_table=nn_table)
            st.add_observation(obs)
            srow, _ = st.finalize(False)
            rows.append({"t0": float(grid[k0]), "N": int(ok.sum()), **srow})
        k0 += n_win
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "vasarhelyi2018_raw.parquet")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "p3_wind.yaml")
    ap.add_argument("--sim", type=Path, default=ROOT / "results" / "P3_wind_v1")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "O4_vasarhelyi_results.md")
    ap.add_argument("--figdir", type=Path, default=ROOT / "reports" / "figures")
    args = ap.parse_args()
    cfg = load_config(args.config)
    nn_table = NNBiasTable.load(cfg.observables.nn_bias_table)
    variants, dr = cfg.observables.mode_removal, cfg.observables.dr_m
    flights = prepare(pd.read_parquet(args.data))
    rows = []
    for flight, (grid, X, F) in flights.items():
        for q in (1, 5):
            for r in windows(grid, X, F, q, nn_table, variants, dr):
                rows.append({"flight": flight, "q": q, **r})
    W = pd.DataFrame(rows)
    W.to_csv(args.report.with_name("O4_vasarhelyi_windows.csv"), index=False)

    sim = read_runs(args.sim)
    sim = sim[sim["N"] == 30]
    sim["fam"] = np.where(sim["class"].isin(["B1", "B3"]), "Vögel", sim["class"])
    L = ["# O4 (explorativ): T_nn an echten Drohnenschwarm-Flügen (Vásárhelyi et al. 2018)\n",
         "Fenster: 10 s, Raster 0,2 s (5 Hz), Geschwindigkeit per zentraler Differenz über ±q Rasterpunkte "
         "(q = 1: ±0,2 s; q = 5: ±1,0 s). Kenngrößen mit dem Simulationscode (Abzug A7, Bias-Korrektur).\n"]
    agg = (W.groupby(["flight", "q"])
           .agg(Fenster=("T_nn_A7", "size"), N_Median=("N", "median"),
                T_nn_Median=("T_nn_A7", "median"), T_nn_q25=("T_nn_A7", lambda s: s.quantile(0.25)),
                T_nn_q75=("T_nn_A7", lambda s: s.quantile(0.75)), c_nn_Median=("c_nn_A7", "median"),
                sd_ds=("var_s_A7", lambda s: float(np.sqrt(s.median()))), Phi=("Phi", "median"),
                d_nn=("nn_dist", "median"), Speed=("mean_speed", "median")).round(3))
    L.append("## Echte Flüge\n")
    L.append(md(agg))
    simagg = (sim.groupby(["control", "fam"])
              .agg(T_nn_Median=("T_nn_A7", "median"), T_nn_q25=("T_nn_A7", lambda s: s.quantile(0.25)),
                   T_nn_q75=("T_nn_A7", lambda s: s.quantile(0.75)),
                   sd_ds=("var_s_A7", lambda s: float(np.sqrt(s.median()))), d_nn=("nn_dist", "median"),
                   n=("T_nn_A7", "size")).round(3))
    L.append("\n## Simulation bei N = 30 (P3-Hauptlauf; K0 = mit Wind, K-Calm = ohne Wind)\n")
    L.append(md(simagg))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    figure(W, sim, args.figdir)


def figure(W: pd.DataFrame, sim: pd.DataFrame, figdir: Path) -> None:
    """Abb. 16: T_nn echter Flüge (je Fenster) gegen simulierte Vögel und D3 bei N = 30."""
    rng = np.random.default_rng(0)
    groups = [("Vögel, simuliert (Wind)", sim[(sim.fam == "Vögel") & (sim.control == "K0")]["T_nn_A7"], COLORS["Vögel"]),
              ("D3, simuliert (Wind)", sim[(sim["class"] == "D3") & (sim.control == "K0")]["T_nn_A7"], COLORS["D3"]),
              ("D3, simuliert (ohne Wind)", sim[(sim["class"] == "D3") & (sim.control == "K-Calm")]["T_nn_A7"], COLORS["D3"])]
    for flight in sorted(W.flight.unique()):
        for q in (1, 5):
            groups.append((f"echt: {flight}, q = {q}", W[(W.flight == flight) & (W.q == q)]["T_nn_A7"], INK2))
    fig, ax = plt.subplots(figsize=(8.6, 5.6), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
    for k, (label, vals, col) in enumerate(groups):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        y = k + rng.uniform(-0.18, 0.18, v.size)
        ax.plot(v, y, "o", color=col, markersize=3, alpha=0.35, zorder=2)
        if v.size:
            q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
            ax.hlines(k, q1, q3, color=INK, linewidth=3, zorder=3)
            ax.plot(med, k, "|", color=INK, markersize=14, markeredgewidth=2.5, zorder=4)
    ax.axvline(0.0, color=MUTED, linewidth=1, linestyle=(0, (2, 3)))
    ax.set_yticks(range(len(groups)), [g[0] for g in groups], fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("T_nn (A7, biaskorrigiert) · Punkte: Designpunkte bzw. 10-s-Fenster · Balken: Quartile, Strich: Median",
                  color=INK2, fontsize=8.5)
    ax.set_title("Echte Drohnenflüge (Vásárhelyi 2018, ≈ 30 Drohnen) gegen Simulation bei N = 30",
                 color=INK, fontsize=10.5, loc="left")
    fig.tight_layout()
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "fig16_vasarhelyi_tnn.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
