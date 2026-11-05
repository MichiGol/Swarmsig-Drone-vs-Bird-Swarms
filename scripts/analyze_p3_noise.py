"""Auswertung des P3-Nachtrags „Drohnenrauschen mit Wind“ (``reports/P3_noise_prereg.md``).

AUC′ = P(T_nn,Drohne > T_nn,Vogel) bei N = 50 für σ_d = 0,1 (Hauptlauf P3_wind_v1), 0,3 und 1,0,
jeweils mit Wind (K0) und windstill (K-Calm) auf denselben Designpunkten; gepaarte Differenz
ΔAUC_Wind = AUC′(ohne Wind) − AUC′(mit Wind). Zum Vergleich die P1-Sensitivität (anderer Seed).

    python scripts/analyze_p3_noise.py
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
from matplotlib.lines import Line2D  # noqa: E402

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", type=Path, default=ROOT / "results" / "P3_wind_v1")
    ap.add_argument("--runs", nargs="+", default=["0.3=results/P3_noise_sd03", "1.0=results/P3_noise_sd10"])
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "P3_noise_results.md")
    ap.add_argument("--figdir", type=Path, default=ROOT / "reports" / "figures")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    main_df = read_runs(args.main)
    main_df = main_df[main_df["N"] == N]
    runs: dict[float, pd.DataFrame] = {0.1: main_df[main_df["class"].isin(DRONES)]}
    for spec in args.runs:
        sd, path = spec.split("=")
        p = ROOT / path
        if p.exists() and any(p.glob("*.parquet")):
            runs[float(sd)] = read_runs(p)
    birds_wind = series(main_df, BIRDS, "K0", N)
    birds_calm = series(main_df, BIRDS, "K-Calm", N)
    rng = np.random.default_rng(20261017)

    rows = []
    for sd in sorted(runs):
        dr = runs[sd]
        for d in DRONES:
            pairs = [(birds_wind, series(dr, d, "K0", N)), (birds_calm, series(dr, d, "K-Calm", N))]
            est, reps, ids = boot_aucs(pairs, rng, args.n_boot)
            (wlo, whi), (clo, chi) = ci(reps[:, 0]), ci(reps[:, 1])
            dlo, dhi = ci(reps[:, 1] - reps[:, 0])
            rows.append({"sigma_d": sd, "Drohne": d, "AUC_wind": est[0], "wind_lo": wlo, "wind_hi": whi,
                         "AUC_calm": est[1], "calm_lo": clo, "calm_hi": chi,
                         "dWind": est[1] - est[0], "dWind_lo": dlo, "dWind_hi": dhi, "n": len(ids)})
    R = pd.DataFrame(rows)
    R.to_csv(args.report.with_name("P3_noise_auc.csv"), index=False)
    p1 = pd.read_csv(ROOT / "reports" / "P1_sens_auc.csv")
    p1 = p1[p1["N"] == N].rename(columns={"AUC'": "AUC_p1"})

    L = ["# P3-Nachtrag „Drohnenrauschen mit Wind“: Ergebnisse (automatisch erzeugt)\n",
         f"Vorregistrierung: `reports/P3_noise_prereg.md`. N = {N}, Designpunkte je σ_d: "
         f"{int(R['n'].min())}–{int(R['n'].max())}. Zellen: AUC′ = P(T_Drohne > T_Vogel) "
         "[95-%-Intervall, Designpunkt-Bootstrap]. P1: unabhängiger Seed, ohne Wind.\n"]
    for d in DRONES:
        sub = R[R.Drohne == d].sort_values("sigma_d")
        t = pd.DataFrame({
            "σ_d": sub["sigma_d"],
            "mit Wind (K0)": [cell(r["AUC_wind"], r["wind_lo"], r["wind_hi"]) for r in sub.to_dict("records")],
            "ohne Wind (K-Calm)": [cell(r["AUC_calm"], r["calm_lo"], r["calm_hi"]) for r in sub.to_dict("records")],
            "ΔAUC_Wind (ohne − mit)": [dcell(r["dWind"], r["dWind_lo"], r["dWind_hi"]) for r in sub.to_dict("records")],
            "P1 ohne Wind": [f"{float(p1[(p1.Drohne == d) & np.isclose(p1.sigma_d, s)]['AUC_p1'].iloc[0]):.2f}"
                             if ((p1.Drohne == d) & np.isclose(p1.sigma_d, s)).any() else "–" for s in sub["sigma_d"]],
        }).set_index("σ_d")
        L.append(f"\n## Regler {d}\n")
        L.append(md(t))
    g = R[np.isclose(R.sigma_d, 1.0)]
    if len(g) == 3:
        if (g["AUC_wind"] >= 0.8).all():
            verdict = "robust (AUC′ ≥ 0,8 bei σ_d = 1,0 mit Wind, alle Regler)"
        elif (g["AUC_wind"] < 0.7).any():
            verdict = "hängt wesentlich an der Rauschannahme (AUC′ < 0,7 bei σ_d = 1,0 mit Wind, mind. ein Regler)"
        else:
            verdict = "teilweise robust (0,7 ≤ AUC′ < 0,8 bei σ_d = 1,0 mit Wind für mind. einen Regler)"
        L.append(f"\n**Lesart nach Vorregistrierung: {verdict}**\n")
        big = R[np.abs(R["dWind"]) > 0.05]
        L.append("**Wechselwirkung Wind × Rauschen:** "
                 + ("alle |ΔAUC_Wind| ≤ 0,05 (Erwartung erfüllt)." if big.empty else
                    "|ΔAUC_Wind| > 0,05 bei " + ", ".join(f"{r.Drohne} (σ_d = {r.sigma_d:g}: {r.dWind:+.2f})"
                                                          for r in big.itertuples()) + ".") + "\n")

    desc = []
    for sd, dr in sorted(runs.items()):
        for ctrl in ("K0", "K-Calm"):
            for d in DRONES:
                s = dr[(dr["class"] == d) & (dr["control"] == ctrl)]
                desc.append({"σ_d": sd, "Wind": "mit" if ctrl == "K0" else "ohne", "Drohne": d,
                             "T_nn Median": float(s[STAT].median()),
                             "sd_ds Median [m/s]": float(np.sqrt(s["var_s_A7"].median())), "n": len(s)})
    for ctrl in ("K0", "K-Calm"):
        s = main_df[main_df["class"].isin(BIRDS) & (main_df["control"] == ctrl)]
        desc.append({"σ_d": float("nan"), "Wind": "mit" if ctrl == "K0" else "ohne", "Drohne": "Vögel",
                     "T_nn Median": float(s[STAT].median()),
                     "sd_ds Median [m/s]": float(np.sqrt(s["var_s_A7"].median())), "n": len(s)})
    L.append("\n## Beschreibung (Mediane)\n")
    L.append(md(pd.DataFrame(desc).set_index(["Drohne", "σ_d", "Wind"]).sort_index().round(3)))
    args.report.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    figure(R, p1, args.figdir)


def figure(R: pd.DataFrame, p1: pd.DataFrame, figdir: Path) -> None:
    """Abb. 12: AUC′ gegen σ_d je Regler, mit Wind (durchgezogen) und ohne (gestrichelt)."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), sharey=True, facecolor=SURFACE)
    for ax, d in zip(axes, DRONES):
        style(ax)
        s = R[R.Drohne == d].sort_values("sigma_d")
        off = 1.04
        ax.plot(s.sigma_d / off, s.AUC_wind, color=COLORS[d], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.vlines(s.sigma_d / off, s.wind_lo, s.wind_hi, color=COLORS[d], linewidth=2, alpha=0.8, zorder=3)
        ax.plot(s.sigma_d * off, s.AUC_calm, color=COLORS[d], linewidth=1.5, linestyle=(0, (4, 3)), marker="o",
                markersize=6, markerfacecolor=SURFACE, zorder=2)
        ax.vlines(s.sigma_d * off, s.calm_lo, s.calm_hi, color=COLORS[d], linewidth=1.5, alpha=0.6, zorder=2)
        q = p1[p1.Drohne == d].sort_values("sigma_d")
        ax.plot(q.sigma_d, q.AUC_p1, linestyle="none", marker="x", markersize=7, color=MUTED, zorder=4)
        for y in (0.5, 0.7, 0.8):
            ax.axhline(y, color=MUTED, linewidth=1, linestyle=(0, (2, 3)), zorder=1)
        ax.set_xscale("log")
        xs = sorted(R.sigma_d.unique())
        ax.set_xticks(xs, [f"{x:g}".replace(".", ",") for x in xs])
        ax.minorticks_off()
        ax.set_title(f"Regler {d}", color=INK, fontsize=10)
        ax.set_xlabel("Prozessrauschen σ_d der Drohnen", color=INK2, fontsize=9)
    axes[0].set_ylabel("AUC′ = P(T_Drohne > T_Vogel)", color=INK2)
    axes[0].set_ylim(0.3, 1.02)
    handles = [Line2D([], [], color=INK2, linewidth=2, marker="o", label="mit Herbstwind (P3)"),
               Line2D([], [], color=INK2, linewidth=1.5, linestyle=(0, (4, 3)), marker="o",
                      markerfacecolor=SURFACE, label="ohne Wind (P3, gleiche Designpunkte)"),
               Line2D([], [], color=MUTED, linestyle="none", marker="x", label="P1 ohne Wind (anderer Seed)")]
    axes[-1].legend(handles=handles, frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle(f"Drohnenrauschen mit und ohne Wind, N = {N} (gestrichelt grau: 0,5 / 0,7 / 0,8)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "fig12_p3_noise_wind.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
