"""Auswertung von P3 Wind (Vorregistrierung: ``reports/P3_prereg.md``).

AUC′ = P(T_Drohne > T_Vogel) für T_nn (A7), U-Statistik ohne Diagonale, Designpunkt-Bootstrap.

* Primär (konfirmatorisch): K0 mit realem Herbstwind, N = 50, je Regler D1, D2, D3.
* Sekundär: ΔAUC_Wind = AUC′(K-Calm) − AUC′(K0) und ΔAUC_K-Wind = AUC′(K0) − AUC′(K-Wind)
  (gepaarter Bootstrap über dieselben Designpunkte), AUC′ nach Windklasse, Abzugsvariante A12,
  Variante „schnelle Böenregelung“ (``results/P3_gustfast_v1``, falls vorhanden).

    python scripts/analyze_p3.py [--runs results/P3_wind_v1] [--variant results/P3_gustfast_v1]
                                 [--report reports/P3_results.md] [--figdir reports/figures]
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
from report_figures import COLORS, INK, INK2, MUTED, SURFACE, style  # noqa: E402
from swarmsig.experiment.analysis import auc_u  # noqa: E402
from swarmsig.io import read_runs  # noqa: E402

DRONES = ("D1", "D2", "D3")
BIRDS = ("B1", "B3")
STAT = "T_nn_A7"
WIND_CLASSES = [("schwach (< 3 m/s)", 0.0, 3.0), ("mäßig (3–6 m/s)", 3.0, 6.0), ("frisch (≥ 6 m/s)", 6.0, np.inf)]


# ---------------------------------------------------------------------------
# AUC′ mit (gepaartem) Designpunkt-Bootstrap
# ---------------------------------------------------------------------------


def series(df: pd.DataFrame, cls: tuple[str, ...] | str, control: str, N: int, stat: str = STAT) -> pd.Series:
    cls = (cls,) if isinstance(cls, str) else cls
    sub = df[df["class"].isin(cls) & (df["control"] == control) & (df["N"] == N)]
    return sub.set_index("design_point_id")[stat].astype(float)


def boot_aucs(pairs: list[tuple[pd.Series, pd.Series]], rng: np.random.Generator, n_boot: int,
              ids: pd.Index | None = None) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """AUC′ = P(T_d > T_b) für mehrere (Vögel, Drohnen)-Paare auf **denselben** Designpunkten.

    Rückgabe: Schätzer (K,), Bootstrap-Replikate (n_boot, K) mit gemeinsamen Gewichten, Indizes.
    """
    common = ids if ids is not None else pairs[0][0].index
    for b, d in pairs:
        common = common.intersection(b.index).intersection(d.index)
    arrs = [(b.loc[common].to_numpy(float), d.loc[common].to_numpy(float)) for b, d in pairs]
    m = len(common)
    est = np.array([auc_u(d, b) if m >= 2 else np.nan for b, d in arrs])  # auc_u(x, y) = P(x > y)
    reps = np.full((n_boot, len(pairs)), np.nan)
    if m >= 10:
        for k in range(n_boot):
            w = np.bincount(rng.integers(0, m, size=m), minlength=m).astype(float)
            reps[k] = [auc_u(d, b, weights=w) for b, d in arrs]
    return est, reps, common


def ci(reps: np.ndarray) -> tuple[float, float]:
    if not np.isfinite(reps).any():
        return float("nan"), float("nan")
    lo, hi = np.nanquantile(reps, [0.025, 0.975])
    return float(lo), float(hi)


def cell(a: float, lo: float, hi: float, digits: int = 2) -> str:
    if not np.isfinite(a):
        return "–"
    f = f"{{:.{digits}f}}"
    return f"{f.format(a)} [{f.format(lo)}, {f.format(hi)}]"


def dcell(a: float, lo: float, hi: float) -> str:
    if not np.isfinite(a):
        return "–"
    return f"{a:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def verdict(a: float, lo: float) -> str:
    if not np.isfinite(a):
        return "–"
    if a >= 0.8 and lo > 0.7:
        return "robust"
    return "abgeschwächt" if a >= 0.7 else "eingebrochen"


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "results" / "P3_wind_v1")
    ap.add_argument("--variant", type=Path, default=ROOT / "results" / "P3_gustfast_v1")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "P3_results.md")
    ap.add_argument("--figdir", type=Path, default=ROOT / "reports" / "figures")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    df = read_runs(args.runs)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261016)
    Ns = sorted(df["N"].unique())
    controls = [c for c in ("K0", "K-Wind", "K-Calm") if c in set(df["control"])]
    # Wetter je Designpunkt (Kovariate, aus der Vogelzeile von K0)
    wx = (df[df["class"].isin(BIRDS) & (df["control"] == "K0")]
          .set_index("design_point_id")[["U10", "U_z0", "sigma_u_z0", "I_u10", "alpha", "z0_m", "weather_station"]])
    L = ["# P3 Wind: Ergebnisse (automatisch erzeugt)\n",
         f"Vorregistrierung: `reports/P3_prereg.md`. Läufe: {len(df)}, Designpunkte: "
         f"{df.design_point_id.nunique()}, Kontrollen: {', '.join(controls)}. Statistik: {STAT}. "
         "Zellen: AUC′ = P(T_Drohne > T_Vogel) [95-%-Intervall, Designpunkt-Bootstrap].\n"]

    # --- A. Primär: K0, alle N, je Regler (konfirmatorisch nur N = 50) ----------------------
    rows = []
    for N in Ns:
        for d in DRONES:
            pairs = [(series(df, BIRDS, "K0", N), series(df, d, "K0", N))]
            est, reps, ids = boot_aucs(pairs, rng, args.n_boot)
            lo, hi = ci(reps[:, 0])
            rows.append({"N": N, "Drohne": d, "AUC'": est[0], "lo": lo, "hi": hi, "n": len(ids),
                         "Urteil": verdict(est[0], lo) if N == 50 else ""})
    A = pd.DataFrame(rows)
    A.to_csv(args.report.with_name("P3_auc_K0.csv"), index=False)
    L.append("## A. Primär: K0 mit realem Herbstwind\n")
    L.append(md(A.assign(Zelle=[cell(r["AUC'"], r["lo"], r["hi"]) for r in A.to_dict("records")])
                .pivot(index="N", columns="Drohne", values="Zelle")))
    g50 = A[A.N == 50]
    if len(g50) == 3:
        v = dict(zip(g50.Drohne, g50.Urteil))
        overall = "ROBUST" if all(x == "robust" for x in v.values()) else (
            "EINGEBROCHEN (mind. ein Regler)" if any(x == "eingebrochen" for x in v.values()) else "ABGESCHWÄCHT")
        L.append(f"\n**Konfirmatorischer Test (N = 50): {overall}** – je Regler: "
                 + ", ".join(f"{k}: {x}" for k, x in v.items()) + "\n")

    # --- B. Gepaarte Kontrollen ---------------------------------------------------------------
    rowsB = []
    for N in Ns:
        for d in DRONES:
            pairs, names = [(series(df, BIRDS, "K0", N), series(df, d, "K0", N))], ["K0"]
            if "K-Calm" in controls:
                pairs.append((series(df, BIRDS, "K-Calm", N), series(df, d, "K-Calm", N)))
                names.append("K-Calm")
            if "K-Wind" in controls:
                pairs.append((series(df, BIRDS, "K0", N), series(df, d, "K-Wind", N)))
                names.append("K-Wind")
            est, reps, ids = boot_aucs(pairs, rng, args.n_boot)
            r = {"N": N, "Drohne": d, "n": len(ids)}
            for k, nm in enumerate(names):
                r[nm], (r[nm + "_lo"], r[nm + "_hi"]) = est[k], ci(reps[:, k])
            if "K-Calm" in names:
                k = names.index("K-Calm")
                r["dWind"] = est[k] - est[0]
                r["dWind_lo"], r["dWind_hi"] = ci(reps[:, k] - reps[:, 0])
            if "K-Wind" in names:
                k = names.index("K-Wind")
                r["dKWind"] = est[0] - est[k]
                r["dKWind_lo"], r["dKWind_hi"] = ci(reps[:, 0] - reps[:, k])
            rowsB.append(r)
    B = pd.DataFrame(rowsB)
    B.to_csv(args.report.with_name("P3_auc_controls.csv"), index=False)
    L.append("\n## B. Kontrollen (gepaart, dieselben Designpunkte und Zufallszahlen)\n")
    for nm, title in (("K-Calm", "ohne Wind (K-Calm)"), ("K-Wind", "Drohnen mit Vogel-Windreaktion (K-Wind)")):
        if nm in B:
            L.append(f"\n**AUC′ {title}**\n")
            L.append(md(B.assign(Zelle=[cell(r[nm], r[nm + "_lo"], r[nm + "_hi"]) for r in B.to_dict("records")])
                        .pivot(index="N", columns="Drohne", values="Zelle")))
    if "dWind" in B:
        L.append("\n**ΔAUC_Wind = AUC′(K-Calm) − AUC′(K0)** (positiv = Wind kostet Unterscheidbarkeit)\n")
        L.append(md(B.assign(Zelle=[dcell(r["dWind"], r["dWind_lo"], r["dWind_hi"]) for r in B.to_dict("records")])
                    .pivot(index="N", columns="Drohne", values="Zelle")))
    if "dKWind" in B:
        L.append("\n**ΔAUC_K-Wind = AUC′(K0) − AUC′(K-Wind)** (positiv = Anteil aus unterschiedlicher Windreaktion)\n")
        L.append(md(B.assign(Zelle=[dcell(r["dKWind"], r["dKWind_lo"], r["dKWind_hi"]) for r in B.to_dict("records")])
                    .pivot(index="N", columns="Drohne", values="Zelle")))

    # --- C. Nach Windklasse (U10 am Designpunkt) ------------------------------------------------
    rowsC = []
    for name, lo_u, hi_u in WIND_CLASSES:
        ids_w = pd.Index(wx.index[(wx["U10"] >= lo_u) & (wx["U10"] < hi_u)])
        for N in Ns:
            for d in DRONES:
                est, reps, ids = boot_aucs([(series(df, BIRDS, "K0", N), series(df, d, "K0", N))], rng,
                                           max(200, args.n_boot // 4), ids_w)
                lo, hi = ci(reps[:, 0])
                rowsC.append({"Windklasse": name, "N": N, "Drohne": d, "AUC'": est[0], "lo": lo, "hi": hi,
                              "n": len(ids)})
    C = pd.DataFrame(rowsC)
    C.to_csv(args.report.with_name("P3_auc_windclass.csv"), index=False)
    L.append("\n## C. AUC′ (K0) nach Windklasse (U10 des Designpunkts)\n")
    for N in Ns:
        sub = C[C.N == N]
        L.append(f"\n**N = {N}** (Designpunkte je Klasse: "
                 + ", ".join(f"{k}: {int(v)}" for k, v in sub.groupby('Windklasse', sort=False)['n'].first().items()) + ")\n")
        L.append(md(sub.assign(Zelle=[cell(r["AUC'"], r["lo"], r["hi"]) for r in sub.to_dict("records")])
                    .pivot(index="Windklasse", columns="Drohne", values="Zelle")
                    .reindex([w[0] for w in WIND_CLASSES])))
    # Terzile der Turbulenz in Flughöhe
    q = wx["sigma_u_z0"].quantile([1 / 3, 2 / 3]).to_numpy()
    rowsT = []
    for k, (a, b) in enumerate([(-np.inf, q[0]), (q[0], q[1]), (q[1], np.inf)]):
        ids_t = pd.Index(wx.index[(wx["sigma_u_z0"] > a) & (wx["sigma_u_z0"] <= b)])
        for d in DRONES:
            est, reps, ids = boot_aucs([(series(df, BIRDS, "K0", 50), series(df, d, "K0", 50))], rng,
                                       max(200, args.n_boot // 4), ids_t)
            lo, hi = ci(reps[:, 0])
            rowsT.append({"σ_u-Terzil": f"T{k + 1} ({a:.2f}–{b:.2f} m/s)".replace("-inf", "0").replace("inf", "max"),
                          "Drohne": d, "Zelle": cell(est[0], lo, hi), "n": len(ids)})
    L.append("\n**N = 50 nach Turbulenz σ_u in Flughöhe (Terzile)**\n")
    L.append(md(pd.DataFrame(rowsT).pivot(index="σ_u-Terzil", columns="Drohne", values="Zelle")))

    # --- D. Abzugsvariante A12 (entfernt zusätzlich lineare Scherung) ----------------------------
    if "T_nn_A12" in df:
        rowsD = []
        for N in Ns:
            for d in DRONES:
                est, reps, ids = boot_aucs([(series(df, BIRDS, "K0", N, "T_nn_A12"),
                                             series(df, d, "K0", N, "T_nn_A12"))], rng, max(200, args.n_boot // 4))
                lo, hi = ci(reps[:, 0])
                rowsD.append({"N": N, "Drohne": d, "Zelle": cell(est[0], lo, hi)})
        L.append("\n## D. Robustheit: Abzugsvariante A12 (K0)\n")
        L.append(md(pd.DataFrame(rowsD).pivot(index="N", columns="Drohne", values="Zelle")))

    # --- H. Robustheit (vorregistriert): nur zusammenhängende Schwärme, ohne Sturm ---------------
    k0 = df[df["control"] == "K0"]
    coh_ok = k0.groupby("design_point_id")["frag_clusters"].apply(lambda s: bool((s <= 1).all()))
    coh = pd.Index(coh_ok[coh_ok].index)
    calm_ok = pd.Index(wx.index[wx["U10"] <= 10.0])
    n_storm = int((wx["U10"] > 10.0).sum())
    rowsH = []
    for label, ids_h in (("nur zusammenhängend", coh), ("ohne Sturm (U10 ≤ 10 m/s)", calm_ok)):
        for N in Ns:
            for d in DRONES:
                est, reps, ids = boot_aucs([(series(df, BIRDS, "K0", N), series(df, d, "K0", N))], rng,
                                           max(500, args.n_boot // 2), ids_h)
                lo, hi = ci(reps[:, 0])
                rowsH.append({"Teilmenge": label, "N": N, "Drohne": d, "Zelle": cell(est[0], lo, hi), "n": len(ids)})
    H = pd.DataFrame(rowsH)
    L.append(f"\n## H. Robustheit (vorregistriert): Teilmengen von K0\n")
    L.append(f"Zusammenhängend = alle vier Schwärme des Designpunkts ohne Teilgruppen am Ende "
             f"({len(coh)} von {df.design_point_id.nunique()} Designpunkten). Sturm = U10 > 10 m/s "
             f"({n_storm} Designpunkte).\n")
    for label in H["Teilmenge"].unique():
        sub_h = H[H.Teilmenge == label]
        L.append(f"\n**{label}** (Designpunkte je N: "
                 + ", ".join(f"{N}: {int(sub_h[sub_h.N == N]['n'].iloc[0])}" for N in Ns) + ")\n")
        L.append(md(sub_h.pivot(index="N", columns="Drohne", values="Zelle")))
    if "K-Calm" in controls and "T_nn_A12" in df:
        rowsA = []
        for N in Ns:
            for d in DRONES:
                est, reps, ids = boot_aucs([(series(df, BIRDS, "K-Calm", N, "T_nn_A12"),
                                             series(df, d, "K-Calm", N, "T_nn_A12"))], rng, max(200, args.n_boot // 4))
                lo, hi = ci(reps[:, 0])
                rowsA.append({"N": N, "Drohne": d, "Zelle": cell(est[0], lo, hi)})
        L.append("\n**Zur Einordnung von D: Abzugsvariante A12 ohne Wind (K-Calm)**\n")
        L.append(md(pd.DataFrame(rowsA).pivot(index="N", columns="Drohne", values="Zelle")))

    # --- E. Variante schnelle Böenregelung ------------------------------------------------------
    var = None
    if args.variant.exists() and any(args.variant.glob("*.parquet")):
        var = read_runs(args.variant)
        rowsE = []
        for N in sorted(var["N"].unique()):
            for d in DRONES:
                pairs = [(series(df, BIRDS, "K0", N), series(df, d, "K0", N)),
                         (series(df, BIRDS, "K0", N), series(var, d, "K0", N))]
                est, reps, ids = boot_aucs(pairs, rng, args.n_boot)
                lo, hi = ci(reps[:, 1])
                dlo, dhi = ci(reps[:, 1] - reps[:, 0])
                rowsE.append({"N": N, "Drohne": d, "AUC′ Variante": cell(est[1], lo, hi),
                              "Δ zu K0": dcell(est[1] - est[0], dlo, dhi), "n": len(ids)})
        L.append("\n## E. Variante „schnelle Böenregelung“ (τ_est,D ~ U[0,1; 0,5] s), gegen dieselben Vögel\n")
        L.append(md(pd.DataFrame(rowsE).set_index(["N", "Drohne"])))

    # --- F. Beschreibung ------------------------------------------------------------------------
    df["fam"] = np.where(df["class"].isin(BIRDS), "Vögel", df["class"])
    desc = (df.groupby(["control", "N", "fam"])
            .agg(T_nn_Median=(STAT, "median"), sd_ds=("var_s_A7", lambda s: float(np.sqrt(s.median()))),
                 Phi=("Phi", "median"), Fragment=("frag_clusters", lambda s: float((s > 1).mean())),
                 n=(STAT, "size")).round(3))
    L.append("\n## F. Beschreibung (Mediane; Fragment = Anteil mit > 1 Teilgruppe)\n")
    L.append(md(desc))
    L.append("\n**Wetter der Designpunkte (K0)**\n")
    L.append(md(wx[["U10", "U_z0", "sigma_u_z0", "I_u10", "alpha"]].describe().loc[["mean", "25%", "50%", "75%", "max"]].round(3)))

    # --- G. Explorativ: gepaarte Windwirkung je Klasse ------------------------------------------
    G = None
    if "K-Calm" in controls:
        G = paired_wind_effect(df, wx)
        L.append("\n## G. Explorativ: Wie verändert der Wind T_nn und die Schwankungsgröße je Klasse?\n")
        L.append("Gepaarte Differenz je Designpunkt: Wert mit Wind (K0) minus Wert ohne Wind (K-Calm), "
                 "gleiche Zufallszahlen. Spearman-ρ: Zusammenhang der Differenz mit σ_u in Flughöhe.\n")
        tab = (G.groupby(["N", "fam"])
               .agg(dT_nn_Median=("dT", "median"), Anteil_dT_pos=("dT", lambda x: float((x > 0).mean())),
                    sd_ds_ohne=("sd0", "median"), sd_ds_mit=("sd1", "median"),
                    rho_dT_sigma_u=("dT", lambda x: float(pd.Series(x.to_numpy()).corr(
                        pd.Series(G.loc[x.index, "sigma_u_z0"].to_numpy()), method="spearman"))),
                    n=("dT", "size")).round(3))
        L.append(md(tab))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    figures(A, B, C, args.figdir)
    if G is not None:
        fig_wind_effect(G, args.figdir)


def paired_wind_effect(df: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    """Je Designpunkt und Klasse: T_nn und Schwankungsgröße mit (K0) und ohne Wind (K-Calm)."""
    rows = []
    fam = lambda c: "Vögel" if c in BIRDS else c  # noqa: E731
    k0 = df[df["control"] == "K0"].set_index(["design_point_id", "class"])
    kc = df[df["control"] == "K-Calm"].set_index(["design_point_id", "class"])
    for key in k0.index.intersection(kc.index):
        a, b = k0.loc[key], kc.loc[key]
        rows.append({"design_point_id": key[0], "fam": fam(key[1]), "N": int(a["N"]),
                     "dT": float(a[STAT] - b[STAT]), "sd1": float(np.sqrt(a["var_s_A7"])),
                     "sd0": float(np.sqrt(b["var_s_A7"])),
                     "sigma_u_z0": float(wx.loc[key[0], "sigma_u_z0"]) if key[0] in wx.index else np.nan})
    return pd.DataFrame(rows)


def fig_wind_effect(G: pd.DataFrame, figdir: Path) -> None:
    """Abb. 11: gepaarte Änderung von T_nn durch Wind gegen σ_u in Flughöhe (N = 50, Quintile).

    Punkte: Median je Quintil; Balken: 95-%-Bootstrap-Intervall des Medians.
    """
    N = 50 if 50 in set(G.N) else int(G.N.max())
    sub = G[(G.N == N) & np.isfinite(G.sigma_u_z0)]
    if len(sub) < 20:
        return
    edges = np.quantile(sub.sigma_u_z0, np.linspace(0, 1, 6))
    sub = sub.assign(bin=np.clip(np.searchsorted(edges, sub.sigma_u_z0, side="right") - 1, 0, 4))
    rng = np.random.default_rng(11)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    style(ax)
    ax.axhline(0.0, color=MUTED, linewidth=1, zorder=1)
    offs = {"Vögel": -0.012, "D1": -0.004, "D2": 0.004, "D3": 0.012}
    ends = []
    for f in ("Vögel", *DRONES):
        rows = []
        for b, g in sub[sub.fam == f].groupby("bin"):
            v = g["dT"].to_numpy()
            boots = np.median(rng.choice(v, size=(1000, v.size), replace=True), axis=1)
            rows.append((float(g.sigma_u_z0.median()), float(np.median(v)), *np.quantile(boots, [0.025, 0.975])))
        if not rows:
            continue
        x, y, lo, hi = (np.array(c) for c in zip(*rows))
        x = x + offs[f]
        ax.plot(x, y, color=COLORS[f], linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=f, zorder=3)
        ax.vlines(x, lo, hi, color=COLORS[f], linewidth=2, alpha=0.8, zorder=2)
        ends.append([float(y[-1]), f, float(x[-1])])
    ends.sort(key=lambda e: e[0])
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.012)
    x_lab = max(e[2] for e in ends)
    for y, f, _ in ends:
        ax.annotate(f, (x_lab, y), xytext=(12, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Turbulenz in Flughöhe σ_u [m/s] (Median je Quintil der Designpunkte)", color=INK2)
    ax.set_ylabel("ΔT_nn = T_nn mit Wind − ohne Wind", color=INK2)
    ax.set_title(f"Explorativ: Wie der Wind die Nachbarkorrelation verändert (N = {N})",
                 color=INK, fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left", ncol=4)
    fig.tight_layout()
    fig.savefig(figdir / "fig11_p3_wind_effect.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def figures(A: pd.DataFrame, B: pd.DataFrame, C: pd.DataFrame, figdir: Path) -> None:
    figdir.mkdir(parents=True, exist_ok=True)
    Ns = sorted(A.N.unique())
    # Abb. 8: AUC′ mit Wind (durchgezogen) und ohne Wind (gestrichelt)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    style(ax)
    offs = {"D1": -0.02, "D2": 0.0, "D3": 0.02}
    for d in DRONES:
        g = A[A.Drohne == d].sort_values("N")
        x = g.N.to_numpy(float) * 10 ** offs[d]
        if "K-Calm" in B:
            c = B[B.Drohne == d].sort_values("N")
            ax.plot(x, c["K-Calm"], color=COLORS[d], linewidth=1.5, linestyle=(0, (4, 3)), marker="o",
                    markersize=5, markerfacecolor=SURFACE, zorder=2)
        ax.plot(x, g["AUC'"], color=COLORS[d], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=d, zorder=3)
        ax.vlines(x, g.lo, g.hi, color=COLORS[d], linewidth=2, alpha=0.8, zorder=3)
    for y, t in ((0.5, "Zufall (0,5)"), (0.7, "0,7"), (0.8, "Kriterium 0,8")):
        ax.axhline(y, color=MUTED, linewidth=1, linestyle=(0, (2, 3)), zorder=1)
        ax.text(max(Ns) * 1.07, y + 0.012, t, fontsize=8, color=INK2, ha="right")
    ax.set_xscale("log")
    ax.set_xticks(Ns, [str(n) for n in Ns])
    ax.minorticks_off()
    ax.set_xlim(min(Ns) * 0.85, max(Ns) * 1.12)
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("Gruppengröße N", color=INK2)
    ax.set_ylabel("AUC′ = P(T_Drohne > T_Vogel)", color=INK2)
    ax.set_title("P3: mit realem Herbstwind (durchgezogen) vs. ohne Wind (gestrichelt)",
                 color=INK, fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right", ncol=3, title="Drohnenregler", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "fig8_p3_auc_wind.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # Abb. 9: AUC′ nach Windklasse, N = 50
    sub = C[C.N == 50] if 50 in set(C.N) else C[C.N == C.N.max()]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor=SURFACE)
    style(ax)
    xs = np.arange(len(WIND_CLASSES))
    for k, d in enumerate(DRONES):
        g = sub[sub.Drohne == d].set_index("Windklasse").reindex([w[0] for w in WIND_CLASSES])
        x = xs + (k - 1) * 0.09
        ax.plot(x, g["AUC'"], color=COLORS[d], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=d, zorder=3)
        ax.vlines(x, g["lo"], g["hi"], color=COLORS[d], linewidth=2, alpha=0.8, zorder=2)
    for y in (0.5, 0.7, 0.8):
        ax.axhline(y, color=MUTED, linewidth=1, linestyle=(0, (2, 3)), zorder=1)
    ax.set_xticks(xs, [w[0] for w in WIND_CLASSES])
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("AUC′ = P(T_Drohne > T_Vogel)", color=INK2)
    ax.set_xlabel("Windklasse (U10 am Designpunkt, DWD-Herbstzustände)", color=INK2)
    ax.set_title(f"P3: Unterscheidbarkeit nach Windstärke, N = {int(sub.N.iloc[0])} (gestrichelt: 0,5 / 0,7 / 0,8)",
                 color=INK, fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left", ncol=3, title="Drohnenregler", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "fig9_p3_windclass.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
