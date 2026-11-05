"""Auswertung von P1 nach Vorregistrierung (``reports/P1_prereg.md``).

AUC = P(T_Vogel > T_Drohne) je (N, Drohnenregler) als U-Statistik ohne Diagonale mit
Designpunkt-Bootstrap; Go/No-go; Schnitte nach Vogelmodell; Auswertung nur mit
zusammenhängenden Schwärmen; Diagnosen (Zerfall, Zeitschritte, Rechenzeit).

    python scripts/analyze_p1.py --runs results/P1_small_v1 --out reports/P1_results.md
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
    sys.stdout.reconfigure(encoding="utf-8")

from swarmsig.experiment.analysis import auc_u, bootstrap_auc  # noqa: E402
from swarmsig.io import read_runs  # noqa: E402

DRONES = ("D1", "D2", "D3")
PRIMARY = "T_nn_A7"
SECONDARY = ("T_s_A7", "T_perp_A7", "T_chi_A7", "T_nn_M", "T_nn_A12")


def md(t: pd.DataFrame) -> str:
    """DataFrame als Markdown-Tabelle (ohne Zusatzpaket ``tabulate``)."""
    t = t.reset_index()
    cols = [str(c) for c in t.columns]

    def cell(v: object) -> str:
        if isinstance(v, float):
            return "–" if not np.isfinite(v) else f"{v:.3g}"
        return str(v)

    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(cell(v) for v in row) + " |" for row in t.itertuples(index=False)]
    return "\n".join(out)


def bird_label(row: pd.Series) -> str:
    if row["class"] == "B1":
        return "B1"
    return f"B3-{'steif' if row['b_speed_control'] == 'stiff' else 'marginal'}"


def aligned(df: pd.DataFrame, stat: str, drone: str, mask_points: pd.Index | None = None) -> tuple[np.ndarray, np.ndarray]:
    birds = df[df["class"].isin(["B1", "B3"])].set_index("design_point_id")[stat]
    drones = df[df["class"] == drone].set_index("design_point_id")[stat]
    ids = birds.index.intersection(drones.index)
    if mask_points is not None:
        ids = ids.intersection(mask_points)
    return birds.loc[ids].to_numpy(float), drones.loc[ids].to_numpy(float)


def auc_table(df: pd.DataFrame, stat: str, rng: np.random.Generator, n_boot: int,
              mask_points: pd.Index | None = None) -> pd.DataFrame:
    rows = []
    for N, g in df.groupby("N"):
        for d in DRONES:
            b, dr = aligned(g, stat, d, mask_points)
            if len(b) < 10:
                rows.append({"N": N, "Drohne": d, "n": len(b), "AUC": np.nan, "lo": np.nan, "hi": np.nan})
                continue
            res = bootstrap_auc(b, dr, rng, n_boot=n_boot)
            rows.append({"N": N, "Drohne": d, "n": len(b), "AUC": res.estimate, "lo": res.low, "hi": res.high})
    return pd.DataFrame(rows)


def fmt(t: pd.DataFrame) -> pd.DataFrame:
    """Breite Tabelle: Zeilen N, Spalten Drohnenregler, Zellen 'AUC [lo, hi]'."""
    t = t.copy()
    t["Zelle"] = [f"{a:.2f} [{lo:.2f}, {hi:.2f}]" if np.isfinite(a) else "–"
                  for a, lo, hi in zip(t.AUC, t.lo, t.hi)]
    w = t.pivot(index="N", columns="Drohne", values="Zelle")
    w.insert(0, "n", t.groupby("N")["n"].max())
    return w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "results" / "P1_small_v1")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "P1_results.md")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    df = read_runs(args.runs)
    rng = np.random.default_rng(20261013)
    df["bird_label"] = ""
    is_bird = df["class"].isin(["B1", "B3"])
    df.loc[is_bird, "bird_label"] = df[is_bird].apply(bird_label, axis=1)
    lines: list[str] = []
    p = lines.append

    # --- Primärstatistik ---------------------------------------------------------------
    prim = auc_table(df, PRIMARY, rng, args.boot)
    go_n = []
    for N, g in prim.groupby("N"):
        if len(g) == len(DRONES) and (g.AUC >= 0.8).all() and (g.lo > 0.7).all():
            go_n.append(int(N))
    nogo = bool((prim.AUC.dropna() < 0.7).all())
    verdict = "GO" if go_n else ("NO-GO" if nogo else "weder Go noch No-go (Grauzone)")

    p("# P1-Ergebnisse: kleine Schwärme (P1_small_v1)\n")
    n_pts = df["design_point_id"].nunique()
    p(f"Ausgewertet: {n_pts} Designpunkte, {len(df)} Läufe. Vorregistrierung: `reports/P1_prereg.md`.\n")
    p(f"## Primärstatistik T_nn (A7): AUC = P(T_Vogel > T_Drohne), 95-%-Bootstrap-Intervall\n")
    p(md(fmt(prim)))
    p(f"\n**Entscheidung nach Vorregistrierung: {verdict}**"
      + (f" (erfüllt bei N = {', '.join(map(str, go_n))})" if go_n else "") + "\n")

    # --- Sekundärstatistiken (Punktschätzer) --------------------------------------------
    p("## Sekundärstatistiken (AUC, Mittel über die drei Drohnenregler)\n")
    sec = []
    for stat in SECONDARY:
        t = auc_table(df, stat, rng, 200)
        sec.append(t.groupby("N")["AUC"].mean().rename(stat))
    p(md(pd.concat(sec, axis=1).round(2)))

    # --- Schnitte nach Vogelmodell ------------------------------------------------------
    p("\n## Primärstatistik getrennt nach Vogelmodell (Punktschätzer, Mittel über Drohnenregler)\n")
    cut = []
    for lab in ("B1", "B3-steif", "B3-marginal"):
        ids = pd.Index(df.loc[df.bird_label == lab, "design_point_id"].unique())
        t = auc_table(df, PRIMARY, rng, 200, mask_points=ids)
        cut.append(t.groupby("N").agg(AUC=("AUC", "mean"), n=("n", "max")).add_prefix(f"{lab} "))
    p(md(pd.concat(cut, axis=1).round(2)))

    # --- Nur zusammenhängende Schwärme --------------------------------------------------
    ok_pts = df.groupby("design_point_id").apply(lambda g: bool((g["frag_clusters"] <= 1).all()))
    coh = pd.Index(ok_pts[ok_pts].index)
    p(f"\n## Nur Designpunkte, in denen alle Schwärme zusammenhängen ({len(coh)} von {n_pts})\n")
    p(md(fmt(auc_table(df, PRIMARY, rng, 500, mask_points=coh))))

    # --- Diagnosen ----------------------------------------------------------------------
    p("\n## Diagnosen\n")
    diag = df.groupby(["N", "class"]).agg(
        Laeufe=("class", "size"), zerfallen=("frag_clusters", lambda s: float((s > 1).mean())),
        stationaer=("stationary_ok", "mean"), Phi=("Phi", "mean"), nn=("nn_dist", "mean"),
        dt_ms=("dt_s", lambda s: 1e3 * float(s.mean())), Sek_je_Lauf=("wall_time_s", "mean"))
    p(md(diag.round(3)))
    p(f"\nRechenzeit gesamt (Summe über Läufe): {df['wall_time_s'].sum() / 60:.1f} CPU-min.")
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    prim.to_csv(args.out.with_suffix(".csv"), index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
