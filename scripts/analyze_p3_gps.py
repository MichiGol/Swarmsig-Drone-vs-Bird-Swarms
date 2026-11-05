"""Auswertung des P3-Nachtrags „Positionierung: RTK gegen Standard-GPS“ (``reports/P3_gps_prereg.md``).

AUC′ = P(T_nn,Drohne > T_nn,Vogel) bei N = 50 mit Wind für die Profile RTK und Standard-GPS, jeweils
mit eigenem GPS-Fehler im Regler („mit“) und ohne („ohne“, Referenz); gepaarte Differenz
Δ_eigen = AUC′(ohne) − AUC′(mit). Vögel aus dem P3-Hauptlauf (K0).

    python scripts/analyze_p3_gps.py
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
from analyze_p3 import BIRDS, DRONES, STAT, boot_aucs, cell, ci, dcell, series  # noqa: E402
from report_figures import COLORS, INK, INK2, MUTED, SURFACE, style  # noqa: E402
from swarmsig.io import read_runs  # noqa: E402

N = 50
PROFILES = {"rtk": "RTK", "gps": "Standard-GPS"}


def verdict(vals: list[float]) -> str:
    if all(v >= 0.8 for v in vals):
        return "robust (alle Regler AUC′ ≥ 0,8)"
    if any(v < 0.7 for v in vals):
        return "hängt wesentlich an der Positionierung (mind. ein Regler AUC′ < 0,7)"
    return "teilweise robust (0,7 ≤ AUC′ < 0,8 für mind. einen Regler)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", type=Path, default=ROOT / "results" / "P3_wind_v1")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "P3_gps_results.md")
    ap.add_argument("--figdir", type=Path, default=ROOT / "reports" / "figures")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    main_df = read_runs(args.main)
    main_df = main_df[main_df["N"] == N]
    birds = series(main_df, BIRDS, "K0", N)
    rng = np.random.default_rng(20261018)
    runs = {}
    for key in PROFILES:
        for own in ("own", "ref"):
            p = args.results / f"P3_pos_{key}_{own}"
            if p.exists() and any(p.glob("*.parquet")):
                runs[(key, own)] = read_runs(p)

    rows = []
    for key, label in PROFILES.items():
        if (key, "own") not in runs or (key, "ref") not in runs:
            continue
        for d in DRONES:
            pairs = [(birds, series(runs[(key, "own")], d, "K0", N)), (birds, series(runs[(key, "ref")], d, "K0", N))]
            est, reps, ids = boot_aucs(pairs, rng, args.n_boot)
            (olo, ohi), (rlo, rhi) = ci(reps[:, 0]), ci(reps[:, 1])
            dlo, dhi = ci(reps[:, 1] - reps[:, 0])
            rows.append({"Profil": label, "Drohne": d, "AUC_mit": est[0], "mit_lo": olo, "mit_hi": ohi,
                         "AUC_ohne": est[1], "ohne_lo": rlo, "ohne_hi": rhi,
                         "d_eigen": est[1] - est[0], "d_lo": dlo, "d_hi": dhi, "n": len(ids)})
    R = pd.DataFrame(rows)
    R.to_csv(args.report.with_name("P3_gps_auc.csv"), index=False)
    # Zum Vergleich: P3-Hauptlauf (breiter Prior, ohne eigenen Fehler)
    base = {}
    for d in DRONES:
        est, reps, _ = boot_aucs([(birds, series(main_df, d, "K0", N))], rng, 500)
        base[d] = (est[0], *ci(reps[:, 0]))

    L = ["# P3-Nachtrag „Positionierung: RTK gegen Standard-GPS“: Ergebnisse (automatisch erzeugt)\n",
         f"Vorregistrierung: `reports/P3_gps_prereg.md`. N = {N}, mit Wind, Designpunkte je Profil: "
         f"{int(R['n'].min()) if len(R) else 0}–{int(R['n'].max()) if len(R) else 0}. "
         "Zellen: AUC′ = P(T_Drohne > T_Vogel) [95-%-Intervall, Designpunkt-Bootstrap].\n"]
    if len(R):
        t = pd.DataFrame({
            "Profil": R["Profil"], "Drohne": R["Drohne"],
            "mit eigenem Fehler": [cell(r["AUC_mit"], r["mit_lo"], r["mit_hi"]) for r in R.to_dict("records")],
            "ohne (Referenz)": [cell(r["AUC_ohne"], r["ohne_lo"], r["ohne_hi"]) for r in R.to_dict("records")],
            "Δ_eigen (ohne − mit)": [dcell(r["d_eigen"], r["d_lo"], r["d_hi"]) for r in R.to_dict("records")],
            "P3-Hauptlauf (breiter Prior)": [cell(*base[d]) for d in R["Drohne"]],
        }).set_index(["Profil", "Drohne"])
        L.append(md(t))
        L.append("")
        for label in R["Profil"].unique():
            g = R[R.Profil == label]
            prefix = "**Primär – " if label == "Standard-GPS" else "**Sekundär – "
            L.append(f"{prefix}{label} mit eigenem Fehler: {verdict(g['AUC_mit'].tolist())}**\n")

    desc = []
    for (key, own), dr in sorted(runs.items()):
        for d in DRONES:
            s = dr[(dr["class"] == d) & (dr["control"] == "K0")]
            desc.append({"Profil": PROFILES[key], "eigener Fehler": "mit" if own == "own" else "ohne", "Drohne": d,
                         "T_nn Median": float(s[STAT].median()),
                         "sd_ds Median [m/s]": float(np.sqrt(s["var_s_A7"].median())),
                         "d_nn Median [m]": float(s["nn_dist"].median()),
                         "Sollabstand Median [m]": float(s["d_spacing"].median()),
                         "Anteil zerfallen": float((s["frag_clusters"] > 1).mean()), "n": len(s)})
    s = main_df[main_df["class"].isin(BIRDS) & (main_df["control"] == "K0")]
    desc.append({"Profil": "–", "eigener Fehler": "–", "Drohne": "Vögel", "T_nn Median": float(s[STAT].median()),
                 "sd_ds Median [m/s]": float(np.sqrt(s["var_s_A7"].median())),
                 "d_nn Median [m]": float(s["nn_dist"].median()), "Sollabstand Median [m]": float("nan"),
                 "Anteil zerfallen": float((s["frag_clusters"] > 1).mean()), "n": len(s)})
    L.append("\n## Beschreibung (Mediane)\n")
    L.append(md(pd.DataFrame(desc).set_index(["Drohne", "Profil", "eigener Fehler"]).round(3)))
    args.report.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    if len(R):
        figure(R, base, args.figdir)


def figure(R: pd.DataFrame, base: dict, figdir: Path) -> None:
    """Abb. 13: AUC′ je Profil, ohne und mit eigenem GPS-Fehler, je Regler."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), sharey=True, facecolor=SURFACE)
    labels = ["RTK\nohne", "RTK\nmit", "GPS\nohne", "GPS\nmit"]
    for ax, d in zip(axes, DRONES):
        style(ax)
        for k, label in enumerate(("RTK", "Standard-GPS")):
            g = R[(R.Drohne == d) & (R.Profil == label)]
            if g.empty:
                continue
            r = g.iloc[0]
            xs = [2 * k, 2 * k + 1]
            ax.plot(xs, [r.AUC_ohne, r.AUC_mit], color=COLORS[d], linewidth=2, zorder=2)
            ax.vlines(xs, [r.ohne_lo, r.mit_lo], [r.ohne_hi, r.mit_hi], color=COLORS[d], linewidth=2, alpha=0.8, zorder=2)
            ax.plot(xs[0], r.AUC_ohne, "o", color=COLORS[d], markersize=7, markerfacecolor=SURFACE, zorder=3)
            ax.plot(xs[1], r.AUC_mit, "o", color=COLORS[d], markersize=7, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, zorder=3)
        ax.axhline(base[d][0], color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        for y in (0.5, 0.7, 0.8):
            ax.axhline(y, color=MUTED, linewidth=1, linestyle=(0, (2, 3)), zorder=1)
        ax.set_xticks(range(4), labels, fontsize=8.5)
        ax.set_xlim(-0.5, 3.5)
        ax.set_title(f"Regler {d}", color=INK, fontsize=10)
    axes[0].set_ylabel("AUC′ = P(T_Drohne > T_Vogel)", color=INK2)
    axes[0].set_ylim(0.4, 1.02)
    fig.suptitle(f"Positionierung RTK gegen Standard-GPS, ohne/mit eigenem GPS-Fehler im Regler (N = {N}, "
                 "mit Wind; gestrichelt: P3-Hauptlauf)", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "fig13_p3_gps_profiles.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
