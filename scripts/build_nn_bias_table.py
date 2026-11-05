"""N0-Biastabelle für T_nn (7.5): E_N0[c_nn | N] je Abzugsvariante, empirisch.

Für jedes N aus den N-Bändern der Config werden ``--reps`` Schnappschüsse des Nullmodells
N0 (unabhängige Schwankungen, zufällige Positionen) erzeugt und c_nn unter identischem
Abzug (A7, M, A12) berechnet. Rechenzeit: Minuten (große N dominieren).

    python scripts/build_nn_bias_table.py --config configs/base.yaml \
        --out data/processed/nn_bias_table.parquet --reps 2000

Danach in der Config ``observables.nn_bias_table`` auf die Datei setzen.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows-Konsole (cp1252) kann z. B. ρ, Δ nicht ausgeben

from swarmsig import nullmodels as nm  # noqa: E402
from swarmsig.config import load_config  # noqa: E402
from swarmsig.observables.modes import remove  # noqa: E402
from swarmsig.observables.smalln import c_nn  # noqa: E402
from swarmsig.rng import experiment_rng  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--spacing", type=float, default=1.0, help="Abstand der N0-Positionen in m (D)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.scenario.N_plan:
        Ns = sorted(cfg.scenario.N_plan)
    else:
        Ns = sorted({n for band in cfg.scenario.N_bands.values() for n in band.values})
    U = np.array([0.0, 10.0, 0.0])
    rows = []
    for n in Ns:
        rng = experiment_rng(cfg.experiment.seed, f"nn_bias_table_N{n}")
        vals: dict[str, list[float]] = {v: [] for v in cfg.observables.mode_removal}
        for _ in range(args.reps):
            x = nm.positions(n, args.spacing, rng, aspect=cfg.init.aspect)
            u = nm.n0(x, U, 1.0, 1.0, rng).u
            for v in vals:
                vals[v].append(c_nn(x, remove(v, x, u).ds))
        for v, arr in vals.items():
            a = np.asarray(arr)
            rows.append({"N": n, "variant": v, "mean_c_nn": float(a.mean()),
                         "se_c_nn": float(a.std(ddof=1) / np.sqrt(a.size)), "n_rep": int(a.size),
                         "spacing_m": args.spacing})
        print(f"N={n}: " + ", ".join(f"{r['variant']} {r['mean_c_nn']:+.5f}" for r in rows[-len(vals):]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"geschrieben: {args.out}")


if __name__ == "__main__":
    main()
