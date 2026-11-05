"""O19 (vorregistriert, ``reports/O19_Segler_prereg.md``): Schornsteinsegler als dritte echte Vogelart.

Daten: Evangelista et al. 2017 (Zenodo 5082229, CC0), ``A_nodes.hdf5``: je Bild eine HDF5-Gruppe mit tid, x, y, z,
vx, vy, vz (Meter, Meter/s; von den Autoren geglättet), 29,97 Bilder/s. Gelesen nur als Zahlenfelder (h5py).
Auswertung je Bild wie O3b (``analyze_real_birds.frame_rec``), Einheiten 10-s-Fenster.

    python scripts/analyze_swifts.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import analyze_real_birds as arb  # noqa: E402
from analyze_p1 import md  # noqa: E402
from run_phase import keep_awake  # noqa: E402
from swarmsig.config import load_config  # noqa: E402
from swarmsig.observables.modes import fit_rigid  # noqa: E402
from swarmsig.observables.smalln import NNBiasTable  # noqa: E402

FPS = 29.97
VARS = ("Q", "L", "RL", "A7", "M")


def reading(q: float, l_: float) -> str:
    if q >= 0.20 or l_ >= 0.35:
        return "stark gekoppelt (wie Dohlen)"
    if q < 0.05 and l_ < 0.10:
        return "schwach gekoppelt (wie Tauben)"
    return "dazwischen (wie echte Drohnen)"


def frames(path: Path, stride: int) -> list[tuple[float, np.ndarray, np.ndarray, float]]:
    out = []
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys(), key=int)
        for k in keys[::stride]:
            g = f[k]
            x = np.column_stack([g["x"][()], g["y"][()], g["z"][()]]).astype(float)
            u = np.column_stack([g["vx"][()], g["vy"][()], g["vz"][()]]).astype(float)
            ok = np.isfinite(x).all(axis=1) & np.isfinite(u).all(axis=1)
            x, u = x[ok], u[ok]
            om = float(np.linalg.norm(fit_rigid(x, u)[1])) if len(x) >= 6 else float("nan")
            out.append((int(k) / FPS, x, u, om))
    return out


def frame_records(fr: list, stride: int) -> list:
    rng = arb.rng_for(f"swift|{stride}")
    return [(t, arb.frame_rec(x, u, 3, rng) if len(x) >= arb.MIN_N else None, om) for t, x, u, om in fr]


def units(recs: list, stride: int, win_s: float, table: NNBiasTable) -> pd.DataFrame:
    per = int(round(win_s * FPS / stride))
    rows = []
    for i0 in range(0, len(recs) - per + 1, per):
        win = [r for _, r, _ in recs[i0:i0 + per] if r is not None]
        if len(win) >= 0.8 * per:
            row = arb.unit_row(win, 3, table)
            row["t0"] = recs[i0][0]
            row["Omega"] = float(np.nanmedian([o for _, _, o in recs[i0:i0 + per]]))
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "evangelista2017_swifts" / "A_nodes.hdf5")
    ap.add_argument("--real", type=Path, default=ROOT / "reports" / "O3b_echte_Voegel_units.csv")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "O19_Segler_results.md")
    args = ap.parse_args()
    arb.VARIANTS = VARS
    table = NNBiasTable.load(load_config(ROOT / "configs" / "o3c_robust.yaml").observables.nn_bias_table)
    keep_awake(True)
    t0 = time.time()
    runs = {}
    recs3 = frame_records(frames(args.data, 3), 3)
    print(f"Bilder (jedes 3.) fertig ({time.time() - t0:.0f} s)", flush=True)
    recs6 = recs3[::2]  # jedes 6. Bild = jedes zweite der 3er-Auswahl (gleiche Zufallszahlen)
    for (stride, win), recs in (((3, 10.0), recs3), ((6, 10.0), recs6), ((3, 2.0), recs3)):
        runs[(stride, win)] = units(recs, stride, win, table)
        print(f"jedes {stride}. Bild, {win:g}-s-Fenster: {len(runs[(stride, win)])} Fenster", flush=True)
    keep_awake(False)
    P = runs[(3, 10.0)]
    pd.concat([d.assign(stride=k[0], win_s=k[1]) for k, d in runs.items()]).to_csv(
        args.report.with_name("O19_Segler_units.csv"), index=False)

    def med(d: pd.DataFrame, c: str) -> float:
        return float(d[c].median())

    L = ["# O19: Schornsteinsegler – Ergebnisse (automatisch erzeugt)\n",
         "Vorregistrierung: `reports/O19_Segler_prereg.md`. Median [Quartile] über Fenster.\n",
         "## Primär: jedes 3. Bild, 10-s-Fenster\n"]
    rows = []
    for (stride, win), d in runs.items():
        rows.append({"Variante": f"jedes {stride}. Bild, {win:g} s", "Fenster": len(d), "N (Median)": round(med(d, "N"), 0),
                     **{v: f"{med(d, f'T_{v}'):.2f} [{d[f'T_{v}'].quantile(.25):.2f}; {d[f'T_{v}'].quantile(.75):.2f}]"
                        for v in VARS},
                     "Lesart": reading(med(d, "T_Q"), med(d, "T_L")),
                     "|Ω| [rad/s]": round(med(d, "Omega"), 3), "Speed": round(med(d, "speed"), 1),
                     "d_nn [m]": round(med(d, "nn_dist"), 2), "Φ": round(med(d, "Phi"), 3)})
    L.append(md(pd.DataFrame(rows).set_index("Variante")))

    U = pd.read_csv(args.real, dtype={"vel": str})
    real = {"Tauben (Kontrolle, 2D, q = 5)": U[(U.dataset == "Tauben") & (U.condition == "c") & (U.q == 5)],
            "echte Drohnen (3D, q = 5)": U[(U.dataset == "Drohnen") & (U.q == 5) & (U.dim == 3)],
            "Dohlen (Schwärme)": U[(U.dataset == "Dohlen") & (U.unit == "Schwarm") & (U.vel == "auth")]}
    comp = [{"echt": k, **{v: round(float(g[f"T_{v}"].median()), 3) for v in VARS}} for k, g in real.items()]
    comp.append({"echt": "Schornsteinsegler (primär)", **{v: round(med(P, f"T_{v}"), 3) for v in VARS}})
    L.append("\n## Vergleich mit den anderen echten Daten (Median)\n")
    L.append(md(pd.DataFrame(comp).set_index("echt")))
    dr = real["echte Drohnen (3D, q = 5)"]
    L.append("\n## AUC′ = P(T_Drohne > T_Segler), Punktwert über Fenster\n")
    L.append(", ".join(f"{v}: {arb.auc(dr[f'T_{v}'].to_numpy(), P[f'T_{v}'].to_numpy()):.2f}" for v in ("Q", "L")) + "\n")
    L.append(f"\nRechenzeit: {time.time() - t0:.0f} s.\n")
    args.report.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
