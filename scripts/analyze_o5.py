"""O5 (vorregistriert, ``reports/O5_prereg.md``): Formationsdrohnen DF ohne Nachbarkopplung gegen Vögel.

AUC′ = P(T_DF > T_Vogel) je Positionsvariante (base, rtk, gps) und N, Designpunkt-Bootstrap; primär Q bei
N = 30/50 und L bei N = 20. Vögel und D1–D3 aus O3c_robust_v1 (gleiche Designpunkte).

    python scripts/analyze_o5.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from analyze_o3c import reading_auc  # noqa: E402
from analyze_p1 import md  # noqa: E402
from analyze_p3 import BIRDS, boot_aucs, cell, ci, series  # noqa: E402
from report_figures import COLORS, INK, INK2, MUTED, SURFACE, style  # noqa: E402
from swarmsig.io import read_runs  # noqa: E402

NS = (20, 30, 50)
VARS = ("base", "rtk", "gps")
LABEL = {"base": "DF, eigener Zustand exakt", "rtk": "DF, RTK (2–6 cm)", "gps": "DF, Standard-GPS (1–3 m)"}
PRIMARY = {20: "L", 30: "Q", 50: "Q"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", type=Path, default=ROOT / "results" / "O3c_robust_v1")
    ap.add_argument("--real", type=Path, default=ROOT / "reports" / "O3b_echte_Voegel_units.csv")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "O5_results.md")
    ap.add_argument("--figdir", type=Path, default=ROOT / "reports" / "figures")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    ref = read_runs(args.ref)
    ref = ref[ref["control"] == "K0"]
    df = {v: read_runs(ROOT / "results" / f"O5_df_{v}") for v in VARS}
    rng = np.random.default_rng(20261016)

    L = ["# O5: Formationsdrohnen ohne Nachbarkopplung – Ergebnisse (automatisch erzeugt)\n",
         "Vorregistrierung: `reports/O5_prereg.md`. Primär: Q bei N = 30/50, L bei N = 20. "
         "Vögel und D1–D3 aus O3c_robust_v1 (gleiche Designpunkte).\n",
         "## 0. Vollständigkeit\n",
         ", ".join(f"{v}: {df[v].groupby('N')['design_point_id'].nunique().to_dict()}" for v in VARS) + "\n"]

    rows = []
    for v in VARS:
        for n in NS:
            for stat in ("Q", "L", "A7", "M"):
                col = f"T_nn_{stat}"
                b = series(ref, BIRDS, "K0", n, col)
                f = series(df[v], "DF", "K0", n, col)
                d3 = series(ref, "D3", "K0", n, col)
                est, reps, ids = boot_aucs([(b, f), (f, d3)], rng, args.n_boot)
                lo, hi = ci(reps[:, 0])
                lo2, hi2 = ci(reps[:, 1])
                rows.append({"Variante": v, "N": n, "Kenngröße": stat, "primär": stat == PRIMARY[n],
                             "AUC_DF_Vogel": est[0], "lo": lo, "hi": hi, "Lesart": reading_auc(est[0], lo, hi),
                             "AUC_D3_DF": est[1], "lo2": lo2, "hi2": hi2, "n": len(ids)})
    A = pd.DataFrame(rows)
    A.to_csv(args.report.with_name("O5_auc.csv"), index=False)
    L.append("## 1. AUC′ = P(T_DF > T_Vogel) (primäre Kenngröße je N fett markiert)\n")
    t = []
    for (v, n), g in A.groupby(["Variante", "N"], sort=False):
        g = g.set_index("Kenngröße")
        p = PRIMARY[n]
        t.append({"Variante": LABEL[v], "N": n, f"primär ({p})": cell(g.loc[p, "AUC_DF_Vogel"], g.loc[p, "lo"], g.loc[p, "hi"]),
                  "Lesart": g.loc[p, "Lesart"],
                  **{s: f"{g.loc[s, 'AUC_DF_Vogel']:.2f}" for s in ("Q", "L", "A7", "M")},
                  "P(T_D3 > T_DF) primär": cell(g.loc[p, "AUC_D3_DF"], g.loc[p, "lo2"], g.loc[p, "hi2"])})
    L.append(md(pd.DataFrame(t).set_index(["Variante", "N"])))

    ref = ref.assign(fam=np.where(ref["class"].isin(BIRDS), "Vögel", ref["class"]))
    cols = ["N", "fam", "T_nn_Q", "T_nn_L", "T_nn_A7", "T_nn_M"]
    allr = pd.concat([ref[cols]] + [df[v].assign(fam=LABEL[v])[cols] for v in VARS])
    desc = allr.groupby(["N", "fam"])[["T_nn_Q", "T_nn_L", "T_nn_A7", "T_nn_M"]].median().round(3)
    L.append("\n## 2. Mediane von T je Klasse\n")
    L.append(md(desc))

    U = pd.read_csv(args.real, dtype={"vel": str})
    real = {"echte Tauben (alle, Kontrolle)": U[(U.dataset == "Tauben") & (U.condition == "c") & (U.q == 5)],
            "echte Drohnen (3D, q = 5)": U[(U.dataset == "Drohnen") & (U.q == 5) & (U.dim == 3)],
            "echte Dohlen (Schwärme)": U[(U.dataset == "Dohlen") & (U.unit == "Schwarm") & (U.vel == "auth")]}
    L.append("\n## 3. Zum Vergleich: echte Daten (O3b), Median\n")
    L.append(md(pd.DataFrame([{"echt": k, "Q": round(float(g["T_Q"].median()), 3), "L": round(float(g["T_L"].median()), 3)}
                              for k, g in real.items()]).set_index("echt")))
    args.report.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    figure(allr, real, args.figdir)


def figure(allr: pd.DataFrame, real: dict[str, pd.DataFrame], figdir: Path) -> None:
    """Abb. 20: T unter Q (N = 50) und L (N = 20) für Vögel, D1–D3, DF und echte Daten."""
    rng = np.random.default_rng(0)
    groups = [("Vögel, simuliert", "Vögel", COLORS["Vögel"]), ("D1, simuliert", "D1", COLORS["D1"]),
              ("D2, simuliert", "D2", COLORS["D2"]), ("D3, simuliert", "D3", COLORS["D3"])]
    groups += [(LABEL[v] + ", simuliert", LABEL[v], INK2) for v in VARS]
    panels = [("Q", 50, "Q, N = 50 (primär)"), ("L", 20, "L, N = 20 (primär)")]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.8), sharey=True, facecolor=SURFACE)
    labels = [g[0] for g in groups] + list(real)
    for ax, (stat, n, title) in zip(axes, panels):
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
        vals = [allr[(allr.N == n) & (allr.fam == key)][f"T_nn_{stat}"] for _, key, _ in groups]
        vals += [g[f"T_{stat}"] for g in real.values()]
        cols = [g[2] for g in groups] + [INK] * len(real)
        marks = ["o"] * len(groups) + ["s"] * len(real)
        for k, (v, col, mk) in enumerate(zip(vals, cols, marks)):
            a = np.asarray(v, float)
            a = a[np.isfinite(a)]
            y = k + rng.uniform(-0.18, 0.18, a.size)
            ax.plot(a, y, mk, color=col, markersize=3.5 if a.size > 20 else 6, alpha=0.4 if a.size > 20 else 0.8, zorder=2)
            if a.size:
                q1, med, q3 = np.quantile(a, [0.25, 0.5, 0.75])
                halo = [pe.withStroke(linewidth=6, foreground=SURFACE)]
                ax.hlines(k, q1, q3, color=INK, linewidth=3, zorder=3, path_effects=halo)
                ax.plot(med, k, "|", color=INK, markersize=14, markeredgewidth=2.5, zorder=4, path_effects=halo)
        ax.axvline(0.0, color=MUTED, linewidth=1, linestyle=(0, (2, 3)))
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("T_nn (biaskorrigiert)", color=INK2, fontsize=9)
    axes[0].set_yticks(range(len(labels)), labels, fontsize=8.5, color=INK)
    axes[0].invert_yaxis()
    fig.suptitle("Formationsdrohnen (DF) gegen Vögel und Schwarmregler · echte Daten: alle Gruppengrößen · "
                 "Balken: Quartile, Strich: Median", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "fig20_formation.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
