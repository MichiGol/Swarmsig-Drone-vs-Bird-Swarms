"""O3a: Echte Vogelschwärme einlesen (Dohlen: Ling et al. 2019; Tauben: Papadopoulou et al. 2022).

Quellen (CC0; Zenodo-Spiegel der Dryad-Datensätze, Dryad selbst blockiert Downloads per Skript):

- **Dohlen:** Zenodo 5007024 (Dryad doi:10.5061/dryad.kb8js06), Datei ``DataS1.txt``. Je Zeile ID, x, y, z [m],
  t [s], u, v, w [m/s], ax, ay, az [m/s²], f_wb [Hz]. Die Schwärme sind durch Zeilen „*****flock NN******“
  getrennt, z ist die Senkrechte. 60 Bilder/s.
- **Tauben:** Zenodo 6027115 (Dryad doi:10.5061/dryad.905qfttkc), Datei ``all_emp_data.csv``, Spalten laut README.
  Ein Flug ist (condition, small.big, group.num, study.flight); ``unique.flight`` ist dagegen eine einzelne Taube.
  GPS mit 5 Hz, nur horizontal. Länge/Breite werden je Flug in lokale Meter (Ost, Nord) um den Flugmittelpunkt
  umgerechnet (gleichabständige Näherung, wie bei den Drohnen in O4).

Beide Dateien werden nur als Text bzw. CSV gelesen.

Ausgabe: ``data/processed/ling2019_jackdaws.parquet`` und ``data/processed/papadopoulou2022_pigeons.parquet``.

    python scripts/bird_data.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
R_EARTH = 6371000.0
JD_COLS = ["id", "x", "y", "z", "t", "u", "v", "w", "ax", "ay", "az", "f_wb"]


def read_jackdaws(path: Path) -> pd.DataFrame:
    rows, flock = [], 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("*"):
                flock = int("".join(c for c in s if c.isdigit()))
                continue
            parts = s.split()
            if flock == 0 or len(parts) != len(JD_COLS):
                continue
            try:
                rows.append([flock] + [float(p) for p in parts])
            except ValueError:
                continue
    df = pd.DataFrame(rows, columns=["flock"] + JD_COLS)
    df["id"] = df["id"].astype(int)
    df["frame"] = np.rint(df["t"] * 60.0).astype(int)
    return df


def read_pigeons(path: Path) -> pd.DataFrame:
    P = pd.read_csv(path)
    key = ["condition", "small.big", "group.num", "study.flight"]
    P["flight"] = P[key].astype(str).agg("_".join, axis=1)
    out = []
    for _, g in P.groupby("flight"):
        g = g.copy()
        lat0, lon0 = np.radians(g["lat"].median()), np.radians(g["lon"].median())
        g["x"] = R_EARTH * np.cos(lat0) * (np.radians(g["lon"]) - lon0)
        g["y"] = R_EARTH * (np.radians(g["lat"]) - lat0)
        out.append(g)
    P = pd.concat(out, ignore_index=True)
    P["frame"] = np.rint(P["time"] / 0.2).astype(int)
    P = P.rename(columns={"small.big": "size", "group.num": "group", "study.flight": "study_flight",
                          "speed": "speed_auth"})
    return P[["flight", "condition", "size", "group", "study_flight", "pigeon", "frame", "time", "x", "y",
              "dist2pred", "speed_auth", "nn1dist"]].sort_values(["flight", "pigeon", "frame"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jackdaws", type=Path, default=ROOT / "data" / "raw" / "ling2019_jackdaws" / "DataS1.txt")
    ap.add_argument("--pigeons", type=Path,
                    default=ROOT / "data" / "raw" / "papadopoulou2022_pigeons" / "all_emp_data.csv")
    ap.add_argument("--outdir", type=Path, default=ROOT / "data" / "processed")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    J = read_jackdaws(args.jackdaws)
    J.to_parquet(args.outdir / "ling2019_jackdaws.parquet", index=False)
    for f, g in J.groupby("flock"):
        per = g.groupby("frame")["id"].nunique()
        print(f"Dohlen Schwarm {f}: {g['id'].nunique():4d} Bahnen | {g['t'].min():5.2f}–{g['t'].max():5.2f} s | "
              f"Vögel/Bild Median {per.median():.0f} (max {per.max()})")

    P = read_pigeons(args.pigeons)
    P.to_parquet(args.outdir / "papadopoulou2022_pigeons.parquet", index=False)
    fl = P.groupby("flight").agg(cond=("condition", "first"), size=("size", "first"), n=("pigeon", "nunique"))
    print(fl.groupby(["cond", "size"])["n"].agg(["size", "min", "max"]).rename(columns={"size": "Flüge"}))
    print(f"geschrieben: {len(J)} Dohlen-Zeilen, {len(P)} Tauben-Zeilen")


if __name__ == "__main__":
    main()
