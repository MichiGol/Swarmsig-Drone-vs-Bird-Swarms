"""Auswertung von Bestätigungslauf und Rausch-Sensitivität (``reports/P1_confirm_prereg.md``).

AUC′ = P(T_Drohne > T_Vogel) für T_nn (A7). Konfirmatorisch nur N = 50 (alle drei Regler
≥ 0,8, untere Grenze > 0,7). Sensitivität: Drohnen mit σ_d = 0,3 und 1,0 gegen dieselben Vögel
(gemeinsame Zufallszahlen).

    python scripts/analyze_confirm.py
"""

from __future__ import annotations

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
from report_figures import COLORS, INK, INK2, MUTED, SURFACE, style  # noqa: E402
from swarmsig.experiment.analysis import bootstrap_auc  # noqa: E402
from swarmsig.io import read_runs  # noqa: E402

DRONES = ("D1", "D2", "D3")
STAT = "T_nn_A7"
N_COLORS = {20: "#86b6ef", 30: "#2a78d6", 50: "#104281"}  # Ordinalrampe, mit dataviz-Validator geprüft


def auc_prime(birds: pd.DataFrame, drones: pd.DataFrame, N: int, d: str, rng, n_boot: int,
              ids_mask: pd.Index | None = None) -> dict[str, float]:
    """AUC′ = P(T_Drohne > T_Vogel) mit Designpunkt-Bootstrap."""
    b = birds[birds.N == N].set_index("design_point_id")[STAT]
    dr = drones[(drones.N == N) & (drones["class"] == d)].set_index("design_point_id")[STAT]
    ids = b.index.intersection(dr.index)
    if ids_mask is not None:
        ids = ids.intersection(ids_mask)
    if len(ids) < 10:
        return {"AUC'": np.nan, "lo": np.nan, "hi": np.nan, "n": len(ids)}
    res = bootstrap_auc(dr.loc[ids].to_numpy(float), b.loc[ids].to_numpy(float), rng, n_boot=n_boot)
    return {"AUC'": res.estimate, "lo": res.low, "hi": res.high, "n": len(ids)}


def cell(r: dict[str, float]) -> str:
    a = r["AUC'"]
    return "–" if not np.isfinite(a) else f"{a:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]"


def main() -> None:
    conf = read_runs(ROOT / "results" / "P1_confirm_v1")
    sens = {0.1: conf[conf["class"].isin(DRONES)]}
    for sd, folder in ((0.3, "P1_sens_sd03"), (1.0, "P1_sens_sd10")):
        p = ROOT / "results" / folder
        if p.exists() and any(p.glob("*.parquet")):
            sens[sd] = read_runs(p)
    birds = conf[conf["class"].isin(["B1", "B3"])]
    rng = np.random.default_rng(20261015)
    L = ["# Bestätigungslauf und Rausch-Sensitivität: Ergebnisse\n",
         f"Vorregistrierung: `reports/P1_confirm_prereg.md`. Designpunkte: {conf.design_point_id.nunique()}; "
         f"Sensitivitätsläufe: {', '.join(f'σ_d = {k}' for k in sens)}.\n"]

    # --- A. Konfirmatorisch ---------------------------------------------------------------
    rows = []
    for N in sorted(conf.N.unique()):
        for d in DRONES:
            r = auc_prime(birds, sens[0.1], N, d, rng, 2000)
            rows.append({"N": N, "Drohne": d, **r})
    A = pd.DataFrame(rows)
    A.to_csv(ROOT / "reports" / "P1_confirm_auc.csv", index=False)
    g50 = A[A.N == 50]
    confirmed = bool(len(g50) == 3 and (g50["AUC'"] >= 0.8).all() and (g50["lo"] > 0.7).all())
    L.append("## A. Bestätigungslauf: AUC′ = P(T_Drohne > T_Vogel), T_nn (A7), 95-%-Intervall\n")
    w = A.assign(Zelle=[cell(r) for r in A.to_dict("records")]).pivot(index="N", columns="Drohne", values="Zelle")
    L.append(md(w))
    L.append(f"\n**Konfirmatorischer Test (N = 50): {'BESTÄTIGT' if confirmed else 'NICHT BESTÄTIGT'}**\n")

    ok = conf.groupby("design_point_id").apply(lambda g: bool((g["frag_clusters"] <= 1).all()))
    coh = pd.Index(ok[ok].index)
    rows = [{"N": N, "Drohne": d, **auc_prime(birds, sens[0.1], N, d, rng, 500, coh)}
            for N in sorted(conf.N.unique()) for d in DRONES]
    C = pd.DataFrame(rows)
    L.append(f"### Nur zusammenhängende Schwärme ({len(coh)} Designpunkte)\n")
    L.append(md(C.assign(Zelle=[cell(r) for r in C.to_dict("records")]).pivot(index="N", columns="Drohne", values="Zelle")))

    lab = birds.apply(lambda r: "B1" if r["class"] == "B1" else f"B3-{'steif' if r['b_speed_control'] == 'stiff' else 'marginal'}", axis=1)
    rows = []
    for name in ("B1", "B3-steif", "B3-marginal"):
        ids = pd.Index(birds.loc[lab == name, "design_point_id"])
        for N in sorted(conf.N.unique()):
            vals = [auc_prime(birds, sens[0.1], N, d, rng, 1, ids)["AUC'"] for d in DRONES]
            rows.append({"N": N, "Vogelmodell": name, "AUC′ (Mittel D1–D3)": float(np.nanmean(vals))})
    L.append("\n### Nach Vogelmodell (Punktschätzer)\n")
    L.append(md(pd.DataFrame(rows).pivot(index="N", columns="Vogelmodell", values="AUC′ (Mittel D1–D3)").round(2)))

    # --- B. Sensitivität ------------------------------------------------------------------
    rows, meds = [], []
    for sd, dr in sens.items():
        for N in sorted(conf.N.unique()):
            for d in DRONES:
                rows.append({"sigma_d": sd, "N": N, "Drohne": d, **auc_prime(birds, dr, N, d, rng, 1000)})
                sub = dr[(dr.N == N) & (dr["class"] == d)]
                meds.append({"sigma_d": sd, "N": N, "Drohne": d, "T_nn Median": float(sub[STAT].median()),
                             "sd_ds Median [m/s]": float(np.sqrt(sub["var_s_A7"].median())),
                             "Phi": float(sub["Phi"].median())})
    B = pd.DataFrame(rows)
    B.to_csv(ROOT / "reports" / "P1_sens_auc.csv", index=False)
    L.append("\n## B. Sensitivität: AUC′ je σ_d (gemeinsame Zufallszahlen)\n")
    for N in sorted(conf.N.unique()):
        sub = B[B.N == N]
        L.append(f"\n**N = {N}**\n")
        L.append(md(sub.assign(Zelle=[cell(r) for r in sub.to_dict("records")])
                    .pivot(index="sigma_d", columns="Drohne", values="Zelle")))
    bird_med = birds.groupby("N").agg(**{"T_nn Median Vögel": (STAT, "median"),
                                         "sd_ds Median Vögel": ("var_s_A7", lambda s: float(np.sqrt(s.median())))})
    L.append("\n### Drohnen: T_nn und Schwankungsgröße je σ_d (Mediane); Vögel zum Vergleich\n")
    L.append(md(pd.DataFrame(meds).set_index(["sigma_d", "N", "Drohne"]).round(3)))
    L.append("")
    L.append(md(bird_med.round(3)))
    if 1.0 in sens:
        g = B[(B.sigma_d == 1.0) & (B.N == 50)]
        if (g["AUC'"] >= 0.8).all():
            verdict = "robust (AUC′ ≥ 0,8 bei σ_d = 1,0, N = 50, alle Regler)"
        elif (g["AUC'"] < 0.7).any():
            verdict = "hängt wesentlich an der Rauschannahme (AUC′ < 0,7 bei σ_d = 1,0, N = 50, mind. ein Regler)"
        else:
            verdict = "teilweise robust (0,7 ≤ AUC′ < 0,8 bei σ_d = 1,0, N = 50 für mind. einen Regler)"
        L.append(f"\n**Lesart nach Vorregistrierung: {verdict}**\n")
    (ROOT / "reports" / "P1_confirm_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    # --- Abbildungen ----------------------------------------------------------------------
    figdir = ROOT / "reports" / "figures"
    explo = pd.read_csv(figdir / "auc_T_nn_A7.csv")  # P1_small_v1: AUC = P(T_Vogel > T_Drohne)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    style(ax)
    for d in DRONES:
        e = explo[(explo.Drohne == d) & explo.N.isin([20, 30, 50])].sort_values("N")
        c = A[A.Drohne == d].sort_values("N")
        ax.plot(e.N, 1 - e.AUC, color=COLORS[d], linewidth=1.5, linestyle=(0, (4, 3)), marker="o",
                markersize=5, markerfacecolor=SURFACE, zorder=2)
        ax.plot(c.N, c["AUC'"], color=COLORS[d], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=d, zorder=3)
        ax.vlines(c.N, c.lo, c.hi, color=COLORS[d], linewidth=2, alpha=0.8, zorder=3)
    for y, t in ((0.5, "Zufall (0,5)"), (0.7, "0,7"), (0.8, "Kriterium 0,8")):
        ax.axhline(y, color=MUTED, linewidth=1, linestyle=(0, (2, 3)), zorder=1)
        ax.text(53.5, y + 0.012, t, fontsize=8, color=INK2, ha="right")
    ax.set_xscale("log")
    ax.set_xticks([20, 30, 50], ["20", "30", "50"])
    ax.minorticks_off()
    ax.set_xlim(17, 56)
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("Gruppengröße N", color=INK2)
    ax.set_ylabel("AUC′ = P(T_Drohne > T_Vogel)", color=INK2)
    ax.set_title("Bestätigung: neue Zufallszahlen (durchgezogen) vs. explorativer Lauf (gestrichelt)",
                 color=INK, fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right", ncol=3, title="Drohnenregler", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "fig6_confirm_auc.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    if len(sens) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), sharey=True, facecolor=SURFACE)
        for ax, d in zip(axes, DRONES):
            style(ax)
            for N in sorted(conf.N.unique()):
                s = B[(B.Drohne == d) & (B.N == N)].sort_values("sigma_d")
                ax.plot(s.sigma_d, s["AUC'"], color=N_COLORS[N], linewidth=2, marker="o", markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"N = {N}")
                ax.vlines(s.sigma_d, s.lo, s.hi, color=N_COLORS[N], linewidth=2, alpha=0.8)
            for y in (0.5, 0.7, 0.8):
                ax.axhline(y, color=MUTED, linewidth=1, linestyle=(0, (2, 3)), zorder=1)
            ax.set_xscale("log")
            xs = sorted(sens)
            ax.set_xticks(xs, [str(x).replace(".", ",") for x in xs])
            ax.minorticks_off()
            ax.set_title(f"Regler {d}", color=INK, fontsize=10)
            ax.set_xlabel("Prozessrauschen σ_d der Drohnen", color=INK2, fontsize=9)
        axes[0].set_ylabel("AUC′ = P(T_Drohne > T_Vogel)", color=INK2)
        axes[0].set_ylim(0, 1.02)
        axes[-1].legend(frameon=False, fontsize=9, loc="lower left", title="Gruppengröße", title_fontsize=9)
        fig.suptitle("Sensitivität: Unterscheidbarkeit bei mehr Drohnenrauschen (gestrichelt: 0,5 / 0,7 / 0,8)",
                     x=0.01, ha="left", fontsize=10.5, color=INK)
        fig.tight_layout()
        fig.savefig(figdir / "fig7_sensitivity_sigma_d.png", dpi=160, facecolor=SURFACE)
        plt.close(fig)


if __name__ == "__main__":
    main()
