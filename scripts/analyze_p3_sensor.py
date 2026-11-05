"""Auswertung des P3-Nachtrags „Messrauschen des Sensors“ (``reports/P3_sensor_prereg.md``).

AUC′ = P(T_nn,Drohne > T_nn,Vogel) bei N = 50 mit Wind für alle Sensorvarianten (Positionsrauschen
σ_pos, Zeitfenster q); gepaarte Differenz zur Grundauswertung (σ_pos = 0, q = 1).

    python scripts/analyze_p3_sensor.py
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
DELTA = 0.1  # s (f_cam = 10 Hz)


def verdict(vals: list[float]) -> str:
    if all(v >= 0.8 for v in vals):
        return "robust"
    if any(v < 0.7 for v in vals):
        return "hängt wesentlich am Messrauschen"
    return "teilweise robust"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "results" / "P3_sensor_v1")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "P3_sensor_results.md")
    ap.add_argument("--figdir", type=Path, default=ROOT / "reports" / "figures")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    df = read_runs(args.runs)
    df = df[df["N"] == N]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261019)
    ref = "K0"
    var = (df[df["control"] != ref].groupby("control")[["sigma_pos_m", "q_frames"]].first()
           .sort_values(["q_frames", "sigma_pos_m"]))
    variants = [(ref, 0.0, 1)] + [(c, float(r.sigma_pos_m), int(r.q_frames)) for c, r in var.iterrows()]

    rows = []
    for ctrl, sp, q in variants:
        for d in DRONES:
            pairs = [(series(df, BIRDS, ref, N), series(df, d, ref, N)),
                     (series(df, BIRDS, ctrl, N), series(df, d, ctrl, N))]
            est, reps, ids = boot_aucs(pairs, rng, args.n_boot)
            lo, hi = ci(reps[:, 1])
            dlo, dhi = ci(reps[:, 0] - reps[:, 1])
            rows.append({"Variante": ctrl, "sigma_pos": sp, "q": q,
                         "sigma_v": np.sqrt(2) * sp / (2 * q * DELTA), "Drohne": d,
                         "AUC": est[1], "lo": lo, "hi": hi, "d": est[0] - est[1], "d_lo": dlo, "d_hi": dhi,
                         "n": len(ids)})
    R = pd.DataFrame(rows)
    R.to_csv(args.report.with_name("P3_sensor_auc.csv"), index=False)

    L = ["# P3-Nachtrag „Messrauschen des Sensors“: Ergebnisse (automatisch erzeugt)\n",
         f"Vorregistrierung: `reports/P3_sensor_prereg.md`. N = {N}, mit Wind, Designpunkte: "
         f"{int(R['n'].min())}. Zellen: AUC′ = P(T_Drohne > T_Vogel) [95-%-Intervall]; "
         "Δ = AUC′(Grundauswertung) − AUC′(Variante), gepaart.\n"]
    t = []
    for (ctrl, sp, q), g in R.groupby(["Variante", "sigma_pos", "q"], sort=False):
        g = g.set_index("Drohne")
        t.append({"σ_pos [m]": f"{sp:g}", "q": q, "σ_v [m/s]": f"{np.sqrt(2) * sp / (2 * q * DELTA):.3f}",
                  **{d: cell(g.loc[d, "AUC"], g.loc[d, "lo"], g.loc[d, "hi"]) for d in DRONES},
                  **{f"Δ {d}": dcell(g.loc[d, "d"], g.loc[d, "d_lo"], g.loc[d, "d_hi"]) for d in DRONES},
                  "Lesart": verdict([g.loc[d, "AUC"] for d in DRONES])})
    L.append(md(pd.DataFrame(t).set_index("σ_pos [m]")))

    desc = []
    for ctrl, sp, q in variants:
        for fam, cls in [("Vögel", BIRDS)] + [(d, (d,)) for d in DRONES]:
            s = df[df["class"].isin(cls) & (df["control"] == ctrl)]
            desc.append({"σ_pos": sp, "q": q, "Klasse": fam, "T_nn Median": float(s[STAT].median()),
                         "sd_ds Median [m/s]": float(np.sqrt(s["var_s_A7"].median())), "n": len(s)})
    L.append("\n## Beschreibung (Mediane)\n")
    L.append(md(pd.DataFrame(desc).set_index(["Klasse", "q", "σ_pos"]).sort_index().round(3)))
    args.report.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    figure(R, args.figdir)


def figure(R: pd.DataFrame, figdir: Path) -> None:
    """Abb. 15: AUC′ gegen Positionsrauschen, je Regler, für q = 1 (durchgezogen) und q = 5 (gestrichelt)."""
    sps = sorted(R["sigma_pos"].unique())
    xs = {sp: k for k, sp in enumerate(sps)}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), sharey=True, facecolor=SURFACE)
    for ax, d in zip(axes, DRONES):
        style(ax)
        for q, ls, off in ((1, "-", -0.06), (5, (0, (4, 3)), 0.06)):
            g = R[(R.Drohne == d) & (R.q == q)].sort_values("sigma_pos")
            x = np.array([xs[s] for s in g.sigma_pos]) + off
            ax.plot(x, g.AUC, color=COLORS[d], linewidth=2 if q == 1 else 1.5, linestyle=ls, marker="o",
                    markersize=7 if q == 1 else 6, markerfacecolor=COLORS[d] if q == 1 else SURFACE,
                    markeredgecolor=SURFACE if q == 1 else COLORS[d], zorder=3)
            ax.vlines(x, g.lo, g.hi, color=COLORS[d], linewidth=2 if q == 1 else 1.5, alpha=0.8, zorder=2)
        for y in (0.5, 0.7, 0.8):
            ax.axhline(y, color=MUTED, linewidth=1, linestyle=(0, (2, 3)), zorder=1)
        ax.set_xticks(range(len(sps)), [f"{100 * s:g} cm" for s in sps])
        ax.set_title(f"Regler {d}", color=INK, fontsize=10)
        ax.set_xlabel("Positionsfehler des Sensors σ_pos", color=INK2, fontsize=9)
    axes[0].set_ylabel("AUC′ = P(T_Drohne > T_Vogel)", color=INK2)
    axes[0].set_ylim(0.0, 1.02)
    handles = [Line2D([], [], color=INK2, linewidth=2, marker="o", label="q = 1 (±0,1 s)"),
               Line2D([], [], color=INK2, linewidth=1.5, linestyle=(0, (4, 3)), marker="o",
                      markerfacecolor=SURFACE, label="q = 5 (±0,5 s)")]
    axes[-1].legend(handles=handles, frameon=False, fontsize=8.5, loc="lower left",
                    title="Zeitfenster der Geschwindigkeit", title_fontsize=8.5)
    fig.suptitle(f"Messrauschen des Sensors, N = {N}, mit Wind (gestrichelt grau: 0,5 / 0,7 / 0,8)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "fig15_p3_sensor_noise.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
