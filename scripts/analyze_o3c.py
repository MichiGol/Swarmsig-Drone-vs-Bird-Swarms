"""O3c-1 (vorregistriert, ``reports/O3c1_prereg.md``): Simulation unter robusten Kenngrößen.

(0) Reproduktion von P3_wind_v1 (A7, M, A12); (1) AUC′ = P(T_Drohne > T_Vogel) unter Q (primär), L, RL und A7
je N und Drohnenklasse (Designpunkt-Bootstrap); (2) Realismus der Modelle gegen echte Daten aus O3b.

    python scripts/analyze_o3c.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import analyze_real_birds as arb  # noqa: E402
from analyze_p1 import md  # noqa: E402
from analyze_p3 import BIRDS, DRONES, boot_aucs, cell, ci, series  # noqa: E402
from swarmsig.io import read_runs  # noqa: E402

STATS = ("Q", "L", "RL", "A7")
NS = (20, 30, 50)


def reading_auc(est: float, lo: float, hi: float) -> str:
    if lo > 0.5:
        return "Drohnen stärker gekoppelt" if est >= 0.8 else "schwache Tendenz zu Drohnen"
    if hi < 0.5:
        return "Vögel stärker gekoppelt"
    return "kein belastbarer Unterschied"


def reading_real(med: float, q10: float, q90: float) -> str:
    if med > q90:
        return "Modell zu schwach gekoppelt"
    if med < q10:
        return "Modell zu stark gekoppelt"
    return "realistisch"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "results" / "O3c_robust_v1")
    ap.add_argument("--p3", type=Path, default=ROOT / "results" / "P3_wind_v1")
    ap.add_argument("--real", type=Path, default=ROOT / "reports" / "O3b_echte_Voegel_units.csv")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "O3c1_results.md")
    ap.add_argument("--figdir", type=Path, default=ROOT / "reports" / "figures")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    sim = read_runs(args.runs)
    sim = sim[sim["control"] == "K0"]
    p3 = read_runs(args.p3)
    p3 = p3[p3["control"] == "K0"]

    L = ["# O3c-1: Simulation unter robusten Kenngrößen – Ergebnisse (automatisch erzeugt)\n",
         "Vorregistrierung: `reports/O3c1_prereg.md`. Mit Wind (K0), dieselben Designpunkte wie P3_wind_v1.\n"]

    # (0) Reproduktion
    key = ["design_point_id", "class"]
    m = sim.merge(p3, on=key, suffixes=("", "_p3"))
    rep = {c: float(np.nanmax(np.abs(m[c] - m[f"{c}_p3"]))) for c in ("T_nn_A7", "T_nn_M", "T_nn_A12")}
    cnt = sim.groupby("N")["design_point_id"].nunique().to_dict()
    L.append("## 0. Vollständigkeit und Reproduktion\n")
    L.append(f"Designpunkte je N: {cnt}; Zeilen: {len(sim)}; davon mit P3 abgeglichen: {len(m)}.\n")
    L.append("Größte Abweichung zu P3: " + ", ".join(f"{c} {v:.1e}" for c, v in rep.items())
             + f" → **{'exakt reproduziert' if max(rep.values()) == 0 else 'NICHT exakt'}**\n")

    # (1) AUC′ je N, Kenngröße, Drohnenklasse
    rng = np.random.default_rng(20261015)
    rows = []
    for n in NS:
        for v in STATS:
            stat = f"T_nn_{v}"
            pairs = [(series(sim, BIRDS, "K0", n, stat), series(sim, d, "K0", n, stat)) for d in DRONES]
            est, reps, ids = boot_aucs(pairs, rng, args.n_boot)
            for k, d in enumerate(DRONES):
                lo, hi = ci(reps[:, k])
                rows.append({"N": n, "Kenngröße": v, "Drohne": d, "AUC": est[k], "lo": lo, "hi": hi,
                             "n": len(ids), "Lesart": reading_auc(est[k], lo, hi)})
    A = pd.DataFrame(rows)
    A.to_csv(args.report.with_name("O3c1_auc.csv"), index=False)
    L.append("\n## 1. AUC′ = P(T_Drohne > T_Vogel) (Designpunkt-Bootstrap)\n")
    for n in NS:
        t = []
        for v in STATS:
            g = A[(A.N == n) & (A["Kenngröße"] == v)].set_index("Drohne")
            t.append({"Kenngröße": v, **{d: cell(g.loc[d, "AUC"], g.loc[d, "lo"], g.loc[d, "hi"]) for d in DRONES},
                      **{f"Lesart {d}": g.loc[d, "Lesart"] for d in DRONES}})
        L.append(f"\n**N = {n}** (Designpunkte: {int(A[A.N == n]['n'].min())})\n")
        L.append(md(pd.DataFrame(t).set_index("Kenngröße")))

    # Beschreibung
    sim = sim.assign(fam=np.where(sim["class"].isin(BIRDS), "Vögel", sim["class"]))
    desc = sim.groupby(["N", "fam"])[[f"T_nn_{v}" for v in ("Q", "L", "RL", "A7", "M")]].median().round(3)
    L.append("\n## Beschreibung: Mediane von T je Klasse\n")
    L.append(md(desc))

    # (2) Realismus gegen echte Daten
    U = pd.read_csv(args.real, dtype={"vel": str})
    pig = U[(U.dataset == "Tauben") & (U.condition == "c") & (U.q == 5)]
    jfa = U[(U.dataset == "Dohlen") & (U.unit == "Schwarm") & (U.vel == "auth")]
    d35 = U[(U.dataset == "Drohnen") & (U.q == 5) & (U.dim == 3)]
    birds = sim[sim["fam"] == "Vögel"]
    d3 = sim[sim["class"] == "D3"]
    rows = []
    for v in ("Q", "L", "RL", "A7", "M"):
        for name, real, model, n in (("Tauben klein", pig[pig["size"] == "s"], birds, 20),
                                     ("Tauben groß", pig[pig["size"] == "b"], birds, 30),
                                     ("Dohlen", jfa, birds, 50),
                                     ("echte Drohnen (3D, q = 5)", d35, d3, 30)):
            s = model[model["N"] == n][f"T_nn_{v}"]
            q10, q50, q90 = (float(s.quantile(p)) for p in (0.1, 0.5, 0.9))
            med = float(real[f"T_{v}"].median())
            rows.append({"Kenngröße": v, "echt": name, "Modell": "D3" if model is d3 else "Vögel", "N": n,
                         "echter Median": round(med, 3), "Sim. 10 %": round(q10, 3), "Sim. Median": round(q50, 3),
                         "Sim. 90 %": round(q90, 3), "Lesart": reading_real(med, q10, q90)})
    R = pd.DataFrame(rows)
    R.to_csv(args.report.with_name("O3c1_realism.csv"), index=False)
    L.append("\n## 2. Realismus der Modelle gegen echte Daten (ideal simuliert gegen echt gemessen)\n")
    L.append(md(R.set_index(["Kenngröße", "echt"])))
    args.report.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    # Abbildung: wie Abb. 18, jetzt mit Simulation unter Q und L
    arb.VARIANTS = ("Q", "L", "A7")
    jac = U[U.dataset == "Dohlen"]
    W = arb.views(U[U.dataset == "Tauben"], U[U.dataset == "Drohnen"], jac[jac.unit == "Schwarm"],
                  jac[jac.unit == "2-s-Fenster"])
    arb.figure(W, birds, d3, ("Q", "L", "A7"), args.figdir / "fig19_simulation_robust.png")


if __name__ == "__main__":
    main()
