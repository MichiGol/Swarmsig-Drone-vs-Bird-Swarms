"""O5-Diagnose (nicht vorregistriert): Kopplung mit und ohne Wind unter Q/L für Vögel, D1–D3 und DF.

Mit Wind: O3c_robust_v1 (K0) und O5_df_base; ohne Wind: O5_calm_all (K-Calm) und O5_df_calm.

    python scripts/analyze_o5_calm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from analyze_o3c import reading_auc  # noqa: E402
from analyze_p1 import md  # noqa: E402
from analyze_p3 import BIRDS, boot_aucs, cell, ci, series  # noqa: E402
from swarmsig.io import read_runs  # noqa: E402

PRIMARY = {20: "L", 30: "Q", 50: "Q"}


def main() -> None:
    wind = pd.concat([read_runs(ROOT / "results" / "O3c_robust_v1"), read_runs(ROOT / "results" / "O5_df_base")])
    calm = pd.concat([read_runs(ROOT / "results" / "O5_calm_all"), read_runs(ROOT / "results" / "O5_df_calm")])
    wind = wind[wind["control"] == "K0"]
    calm = calm[calm["control"] == "K-Calm"]
    rng = np.random.default_rng(20261017)
    L = ["# O5-Diagnose: Kopplung mit und ohne Wind (automatisch erzeugt, nicht vorregistriert)\n",
         f"Designpunkte ohne Wind je N: {calm.groupby('N')['design_point_id'].nunique().to_dict()}\n"]
    rows = []
    for n in (20, 30, 50):
        p = PRIMARY[n]
        for cls in ("D1", "D2", "D3", "DF"):
            r = {"N": n, "Kenngröße": p, "Drohne": cls}
            for name, d, ctrl in (("mit Wind", wind, "K0"), ("ohne Wind", calm, "K-Calm")):
                est, reps, _ = boot_aucs([(series(d, BIRDS, ctrl, n, f"T_nn_{p}"), series(d, cls, ctrl, n, f"T_nn_{p}"))],
                                         rng, 2000)
                lo, hi = ci(reps[:, 0])
                r[f"AUC′ {name}"] = cell(est[0], lo, hi)
                r[f"Lesart {name}"] = reading_auc(est[0], lo, hi)
            rows.append(r)
    L.append("## AUC′ = P(T_Drohne > T_Vogel), primäre Kenngröße (L bei N = 20, sonst Q)\n")
    L.append(md(pd.DataFrame(rows).set_index(["N", "Drohne"])))
    med = []
    for name, d in (("mit Wind", wind), ("ohne Wind", calm)):
        d = d.assign(fam=np.where(d["class"].isin(BIRDS), "Vögel", d["class"]), Wind=name)
        med.append(d.groupby(["N", "fam", "Wind"])[["T_nn_Q", "T_nn_L", "T_nn_A7", "T_nn_M"]].median())
    M = pd.concat(med).round(3).sort_index()
    L.append("\n## Mediane von T\n")
    L.append(md(M))
    out = ROOT / "reports" / "O5_calm_results.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
