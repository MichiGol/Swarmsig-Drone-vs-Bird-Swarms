"""T_nn an echten Vogelschwärmen gegen echte Drohnen (O3a, O3b; vorregistriert in
``reports/O3a_echte_Voegel_prereg.md`` und ``reports/O3b_prereg.md``).

Daten: ``scripts/bird_data.py`` (Dohlen, Tauben) und ``scripts/vasarhelyi_data.py`` (Drohnen).

Je Bild:

- Hauptschwarm (Einzelverknüpfung mit 5 × medianem Nachbarabstand), mindestens 6 Individuen;
- c_nn je Abzug (``--variants``, der erste ist primär) mit den Funktionen des Simulationscodes
  (``modes.remove``, ``smalln.c_nn``);
- Bias-Korrektur je Bild (20 Wiederholungen unabhängiger Schwankungen auf denselben Positionen). Alle Abzüge
  nutzen dieselben Zufallszahlen, daher reproduzieren A7 und M bei anderer Variantenliste exakt.

Einheiten: 10-s-Fenster (Tauben, Drohnen) bzw. ganze Schwärme und 2-s-Fenster (Dohlen).
T = Mittel(c_nn) − Mittel(Bias).

Abweichung von der Vorregistrierung O3a: Dohlen-Schwarm 3 ist in ``DataS1.txt`` ein Duplikat von Schwarm 2
(≈ 50 000 identische Zeilen) und wird nicht als eigener Schwarm gezählt.

O3a (≈ 9 min) und Abbildung aus der gespeicherten Einheiten-CSV:

    python scripts/analyze_real_birds.py
    python scripts/analyze_real_birds.py --figure-only

O3b (≈ 15 min):

    python scripts/analyze_real_birds.py --variants Q,A7,M,L,RL --report reports/O3b_echte_Voegel_results.md
        --fig fig18_robuste_kenngroesse.png --fig-variants Q,A7,M --no-sim-compare
"""

from __future__ import annotations

import argparse
import sys
import time
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from analyze_p1 import md  # noqa: E402
from analyze_p3 import cell, ci  # noqa: E402
from analyze_vasarhelyi import GRID  # noqa: E402
from analyze_vasarhelyi import prepare as prepare_drones  # noqa: E402
from report_figures import COLORS, INK, INK2, MUTED, SURFACE, style  # noqa: E402
from run_phase import keep_awake  # noqa: E402
from swarmsig import nullmodels as nm  # noqa: E402
from swarmsig.agents.neighbors import nearest_neighbor  # noqa: E402
from swarmsig.config import load_config  # noqa: E402
from swarmsig.io import read_runs  # noqa: E402
from swarmsig.observables import size  # noqa: E402
from swarmsig.observables.empirical import main_cluster, null_bias  # noqa: E402
from swarmsig.observables.modes import remove  # noqa: E402
from swarmsig.observables.smalln import NNBiasTable, c_nn, t_nn  # noqa: E402

VARIANTS = ("A7", "M", "A12")
MIN_N = 6
REPS = 20
PIGEON_DT = 0.2
JD_FPS = 60.0
DUPLICATE_FLOCKS = (3,)  # Duplikat von Schwarm 2


def rng_for(label: str) -> np.random.Generator:
    return np.random.default_rng([20261012, zlib.crc32(label.encode())])


def pad3(a: np.ndarray) -> np.ndarray:
    if a.shape[1] == 3:
        return a
    out = np.zeros((a.shape[0], 3))
    out[:, : a.shape[1]] = a
    return out


def frame_rec(x: np.ndarray, u: np.ndarray, dim: int, rng: np.random.Generator) -> dict | None:
    """Kenngrößen eines Bildes nach Hauptschwarm-Filter; None, wenn weniger als MIN_N Individuen."""
    m = main_cluster(x)
    x, u = x[m], u[m]
    n = x.shape[0]
    if n < MIN_N:
        return None
    nn, d = nearest_neighbor(x)
    rec = {"N": n, "nn_dist": float(np.mean(d)), "speed": size.mean_speed(u), "Phi": size.polarization(u)}
    for v in VARIANTS:
        ds = remove(v, x, u).ds
        rec[f"c_nn_{v}"] = c_nn(x, ds, nn=nn)
        rec[f"var_s_{v}"] = float(np.mean(ds**2))
    b = null_bias(x, u.mean(axis=0), VARIANTS, rng, reps=REPS, dim=dim)
    rec.update({f"bias_{v}": b[v] for v in VARIANTS})
    return rec


def unit_row(recs: list[dict], dim: int, table: NNBiasTable) -> dict:
    R = pd.DataFrame(recs)
    row = {"frames": len(R), "N": float(R["N"].mean()), "nn_dist": float(R["nn_dist"].mean()),
           "speed": float(R["speed"].mean()), "Phi": float(R["Phi"].mean())}
    n_mean = int(round(row["N"]))
    for v in VARIANTS:
        c, b = float(R[f"c_nn_{v}"].mean()), float(R[f"bias_{v}"].mean())
        row[f"c_nn_{v}"], row[f"bias_{v}"], row[f"T_{v}"] = c, b, c - b
        row[f"sd_s_{v}"] = float(np.sqrt(R[f"var_s_{v}"].mean()))
        row[f"Ttab_{v}"] = t_nn(c, n_mean, v, table) if dim == 3 else float("nan")
    return row


# ---------------------------------------------------------------- Tauben
def pigeon_units(P: pd.DataFrame, condition: str, q: int, table: NNBiasTable, min_dpred: float | None) -> list[dict]:
    rows = []
    n_win = int(round(10.0 / PIGEON_DT))
    for flight, g in P[P["condition"] == condition].groupby("flight"):
        rng = rng_for(f"pigeon|{flight}|q{q}|{min_dpred}")
        birds = np.sort(g["pigeon"].unique())
        idx = {b: j for j, b in enumerate(birds)}
        K = int(g["frame"].max()) + 1 + q
        X = np.full((K, len(birds), 2), np.nan)
        DP = np.full((K, len(birds)), np.nan)
        j, k = g["pigeon"].map(idx).to_numpy(), g["frame"].to_numpy()
        X[k, j, 0], X[k, j, 1], DP[k, j] = g["x"].to_numpy(), g["y"].to_numpy(), g["dist2pred"].to_numpy()
        V = np.full_like(X, np.nan)
        V[q:K - q] = (X[2 * q:] - X[:K - 2 * q]) / (2 * q * PIGEON_DT)
        recs: dict[int, dict] = {}
        for kk in range(K):
            ok = np.isfinite(X[kk]).all(axis=1) & np.isfinite(V[kk]).all(axis=1)
            if ok.sum() < MIN_N or np.linalg.norm(V[kk, ok], axis=1).mean() < 10.0:
                continue
            if min_dpred is not None and not np.nanmedian(DP[kk, ok]) >= min_dpred:
                continue
            r = frame_rec(pad3(X[kk, ok]), pad3(V[kk, ok]), 2, rng)
            if r is not None:
                recs[kk] = r
        k_first = int(g["frame"].min())
        for k0 in range(k_first, K - n_win, n_win):
            win = [recs[kk] for kk in range(k0, k0 + n_win + 1) if kk in recs]
            if len(win) >= 0.8 * (n_win + 1):
                rows.append({"dataset": "Tauben", "flight": flight, "size": g["size"].iloc[0], "q": q,
                             "condition": condition, "t0": k0 * PIGEON_DT, **unit_row(win, 2, table)})
    return rows


# ---------------------------------------------------------------- Drohnen
def drone_units(flights: dict, q: int, dim: int, table: NNBiasTable) -> list[dict]:
    rows = []
    n_win = int(round(10.0 / GRID))
    for flight, (grid, X, F) in flights.items():
        rng = rng_for(f"drone|{flight}|q{q}|d{dim}")
        k0 = q
        while k0 + n_win + q < grid.size:
            sl = slice(k0 - q, k0 + n_win + q + 1)
            ok = F[sl].all(axis=0) & np.isfinite(X[sl]).all(axis=(0, 2))
            if ok.sum() >= 20:
                Xw = X[sl][:, ok][:, :, :dim]
                u = (Xw[2 * q:] - Xw[:-2 * q]) / (2 * q * GRID)
                xo = Xw[q:-q]
                win = [r for r in (frame_rec(pad3(xo[i]), pad3(u[i]), dim, rng) for i in range(xo.shape[0]))
                       if r is not None]
                if len(win) >= 0.8 * xo.shape[0]:
                    rows.append({"dataset": "Drohnen", "flight": flight, "q": q, "dim": dim,
                                 "t0": float(grid[k0]), **unit_row(win, dim, table)})
            k0 += n_win
    return rows


# ---------------------------------------------------------------- Dohlen
def jackdaw_units(J: pd.DataFrame, vel: str, table: NNBiasTable) -> tuple[list[dict], list[dict]]:
    """Einheiten: ganzer Schwarm und 2-s-Fenster. vel: 'auth' oder Halbfenster in Bildern ('6', '15')."""
    flocks, wins = [], []
    n_win = int(round(2.0 * JD_FPS))
    for flock, g in J.groupby("flock"):
        if flock in DUPLICATE_FLOCKS:
            continue
        rng = rng_for(f"jackdaw|{flock}|{vel}")
        ids = np.sort(g["id"].unique())
        idx = {b: j for j, b in enumerate(ids)}
        K = int(g["frame"].max()) + 1 + 16
        X = np.full((K, len(ids), 3), np.nan)
        j, k = g["id"].map(idx).to_numpy(), g["frame"].to_numpy()
        X[k, j] = g[["x", "y", "z"]].to_numpy()
        if vel == "auth":
            V = np.full_like(X, np.nan)
            V[k, j] = g[["u", "v", "w"]].to_numpy()
        else:
            q = int(vel)
            V = np.full_like(X, np.nan)
            V[q:K - q] = (X[2 * q:] - X[:K - 2 * q]) / (2 * q / JD_FPS)
        recs: dict[int, dict] = {}
        for kk in range(K):
            ok = np.isfinite(X[kk]).all(axis=1) & np.isfinite(V[kk]).all(axis=1)
            if ok.sum() < MIN_N:
                continue
            r = frame_rec(X[kk, ok], V[kk, ok], 3, rng)
            if r is not None:
                recs[kk] = r
        if recs:
            flocks.append({"dataset": "Dohlen", "flock": flock, "vel": vel, **unit_row(list(recs.values()), 3, table)})
        for k0 in range(int(g["frame"].min()), K - n_win, n_win):
            win = [recs[kk] for kk in range(k0, k0 + n_win) if kk in recs]
            if len(win) >= 0.8 * n_win:
                wins.append({"dataset": "Dohlen", "flock": flock, "vel": vel, "t0": k0 / JD_FPS,
                             **unit_row(win, 3, table)})
    return flocks, wins


# ---------------------------------------------------------------- Auswertung
def auc(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) + ½ P(a = b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not a.size or not b.size:
        return float("nan")
    r = rankdata(np.concatenate([a, b]))
    return float((r[: a.size].sum() - a.size * (a.size + 1) / 2) / (a.size * b.size))


def cluster_auc(A: dict, B: dict, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    ka, kb = list(A), list(B)
    est = auc(np.concatenate([A[k] for k in ka]), np.concatenate([B[k] for k in kb]))
    reps = np.array([auc(np.concatenate([A[ka[i]] for i in rng.integers(0, len(ka), len(ka))]),
                         np.concatenate([B[kb[i]] for i in rng.integers(0, len(kb), len(kb))]))
                     for _ in range(n_boot)])
    lo, hi = ci(reps)
    return est, lo, hi


def reading_a(est: float, lo: float, hi: float) -> str:
    if lo > 0.5:
        return "Drohnen stärker gekoppelt" if est >= 0.8 else "schwache Tendenz zu Drohnen"
    if hi < 0.5:
        return "Tauben stärker gekoppelt (H1-Richtung)"
    return "kein belastbarer Unterschied"


def reading_b(med: float, q10: float, q90: float) -> str:
    if med > q90:
        return "Vogelmodelle zu schwach gekoppelt"
    if med < q10:
        return "Vogelmodelle zu stark gekoppelt"
    return "Vogelmodelle realistisch"


def by_cluster(df: pd.DataFrame, key: str, stat: str) -> dict:
    return {k: g[stat].to_numpy() for k, g in df.groupby(key)}


TABLE_VARIANTS = ("A7", "M", "A12")
TITLES = {"A7": "Abzug A7", "M": "nur Mittelwert M (Starenstudien)", "A12": "Abzug A12",
          "L": "Speed − lineares Gefälle (L)", "Q": "Speed − quadratisches Gefälle (Q)",
          "RL": "Speed relativ zur starren Bewegung (RL)"}


def summarize(df: pd.DataFrame, label: str) -> dict:
    def q(s: str, p: float) -> float:
        return float(df[s].quantile(p))
    row = {"Einheiten": label, "n": len(df), "N (Median)": round(float(df["N"].median()), 1)}
    for v in VARIANTS:
        row[f"T {v}"] = f"{q(f'T_{v}', .5):.2f} [{q(f'T_{v}', .25):.2f}; {q(f'T_{v}', .75):.2f}]"
    row.update({f"sd_s {VARIANTS[0]} [m/s]": f"{q(f'sd_s_{VARIANTS[0]}', .5):.2f}",
                "Speed [m/s]": f"{q('speed', .5):.1f}", "d_nn [m]": f"{q('nn_dist', .5):.2f}",
                "Φ": f"{q('Phi', .5):.3f}"})
    return row


def validate_bias(table: NNBiasTable, aspect: tuple[float, float, float]) -> pd.DataFrame:
    """Prüfung: Bias je Bild auf zufälligen Positionen (wie beim Tabellenbau) gegen die Tabelle."""
    tv = [v for v in VARIANTS if v in TABLE_VARIANTS]
    rng = rng_for("validate")
    U = np.array([0.0, 10.0, 0.0])
    rows = []
    for n in (10, 20, 30, 50, 100):
        acc = {v: [] for v in tv}
        for _ in range(200):
            b = null_bias(nm.positions(n, 1.0, rng, aspect=aspect), U, tv, rng, reps=REPS, dim=3)
            for v in tv:
                acc[v].append(b[v])
        for v in tv:
            m = float(np.mean(acc[v]))
            tab = table.bias(n, v)
            rows.append({"N": n, "Abzug": v, "je Bild": round(m, 4), "Tabelle": round(tab, 4),
                         "Differenz": round(m - tab, 4), "ok (±0,01)": abs(m - tab) <= 0.01})
    return pd.DataFrame(rows)


def views(pig: pd.DataFrame, dro: pd.DataFrame, jf: pd.DataFrame, jw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"pc5": pig[(pig.condition == "c") & (pig.q == 5)], "pc1": pig[(pig.condition == "c") & (pig.q == 1)],
            "pp5": pig[pig.condition == "p"],
            "d25": dro[(dro.q == 5) & (dro.dim == 2)], "d21": dro[(dro.q == 1) & (dro.dim == 2)],
            "d35": dro[(dro.q == 5) & (dro.dim == 3)], "d31": dro[(dro.q == 1) & (dro.dim == 3)],
            "jfa": jf[jf.vel == "auth"], "jwa": jw[jw.vel == "auth"]}


def main() -> None:
    global VARIANTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "p3_wind.yaml")
    ap.add_argument("--sim", type=Path, default=ROOT / "results" / "P3_wind_v1")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "O3a_echte_Voegel_results.md")
    ap.add_argument("--figdir", type=Path, default=ROOT / "reports" / "figures")
    ap.add_argument("--fig", default="fig17_echte_voegel_tnn.png")
    ap.add_argument("--variants", default="A7,M,A12", help="Abzüge, kommagetrennt; der erste ist primär")
    ap.add_argument("--fig-variants", default="A7,M", help="Panels der Abbildung")
    ap.add_argument("--no-sim-compare", action="store_true", help="ohne Vergleich (b) mit den Vogelmodellen")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--figure-only", action="store_true", help="nur die Abbildung aus der gespeicherten CSV")
    args = ap.parse_args()
    VARIANTS = tuple(args.variants.split(","))
    prefix = args.report.name.split("_")[0]
    units_csv = args.report.with_name(f"{prefix}_echte_Voegel_units.csv")
    panels = tuple(args.fig_variants.split(","))
    sim = read_runs(args.sim)
    sim = sim[sim["control"] == "K0"]
    sbird, sd3 = sim[sim["class"].isin(["B1", "B3"])], sim[sim["class"] == "D3"]
    if args.figure_only:
        U = pd.read_csv(units_csv, dtype={"vel": str})
        jac = U[U.dataset == "Dohlen"]
        W = views(U[U.dataset == "Tauben"], U[U.dataset == "Drohnen"], jac[jac.unit == "Schwarm"],
                  jac[jac.unit == "2-s-Fenster"])
        figure(W, sbird, sd3, panels, args.figdir / args.fig)
        return
    cfg = load_config(args.config)
    table = NNBiasTable.load(cfg.observables.nn_bias_table)
    proc = ROOT / "data" / "processed"
    t_start = time.time()
    keep_awake(True)  # Rechnung ≈ 10–20 min; Windows-Standby verhindern

    val = validate_bias(table, tuple(cfg.init.aspect))
    print(f"Prüfung Bias-Korrektur fertig ({time.time() - t_start:.0f} s)")

    P = pd.read_parquet(proc / "papadopoulou2022_pigeons.parquet")
    pig = []
    for q in (5, 1):
        pig += pigeon_units(P, "c", q, table, None)
    pig += pigeon_units(P, "p", 5, table, 50.0)
    pig = pd.DataFrame(pig)
    print(f"Tauben fertig ({time.time() - t_start:.0f} s): {len(pig)} Fenster")

    flights = prepare_drones(pd.read_parquet(proc / "vasarhelyi2018_raw.parquet"))
    dro = pd.DataFrame([r for q in (5, 1) for dim in (2, 3) for r in drone_units(flights, q, dim, table)])
    print(f"Drohnen fertig ({time.time() - t_start:.0f} s): {len(dro)} Fenster")

    J = pd.read_parquet(proc / "ling2019_jackdaws.parquet")
    jf, jw = [], []
    for vel in ("auth", "6", "15"):
        f, w = jackdaw_units(J, vel, table)
        jf += f
        jw += w
    jf, jw = pd.DataFrame(jf), pd.DataFrame(jw)
    print(f"Dohlen fertig ({time.time() - t_start:.0f} s): {len(jf)} Schwarm-Einheiten, {len(jw)} Fenster")

    units = pd.concat([pig.assign(unit="10-s-Fenster"), dro.assign(unit="10-s-Fenster"),
                       jf.assign(unit="Schwarm"), jw.assign(unit="2-s-Fenster")], ignore_index=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    units.to_csv(units_csv, index=False)
    W = views(pig, dro, jf, jw)
    pc5, pc1, pp5, d25, d21, d35, d31, jfa, jwa = (W[k] for k in
                                                    ("pc5", "pc1", "pp5", "d25", "d21", "d35", "d31", "jfa", "jwa"))

    L = [f"# {prefix}: echte Vogelschwärme gegen echte Drohnen – Ergebnisse (automatisch erzeugt)\n",
         f"Abzüge: {', '.join(VARIANTS)} (primär: **{VARIANTS[0]}**). T = Mittel(c_nn) − Mittel(Bias je Bild); "
         "Zellen: Median [Quartile] über Einheiten. Dohlen-Schwarm 3 = Duplikat von Schwarm 2, nicht gezählt.\n",
         "## 1. Prüfung der Bias-Korrektur je Bild (zufällige Positionen, 3D, wie Tabellenbau)\n",
         md(val.set_index(["N", "Abzug"])),
         f"\nAlle innerhalb ±0,01: **{bool(val['ok (±0,01)'].all())}**\n",
         "## 2. Einheiten je Datensatz\n"]
    desc = [summarize(pc5[pc5["size"] == "s"], "Tauben klein, Kontrolle, 2D, q = 5"),
            summarize(pc5[pc5["size"] == "b"], "Tauben groß, Kontrolle, 2D, q = 5"),
            summarize(pc1, "Tauben alle, Kontrolle, 2D, q = 1"),
            summarize(pp5, "Tauben Falkenflüge (≥ 50 m), 2D, q = 5"),
            summarize(d25, "Drohnen 2D, q = 5"), summarize(d21, "Drohnen 2D, q = 1"),
            summarize(d35, "Drohnen 3D, q = 5"), summarize(d31, "Drohnen 3D, q = 1"),
            summarize(jfa, "Dohlen, Schwärme, Geschw. der Autoren"),
            summarize(jwa, "Dohlen, 2-s-Fenster, Geschw. der Autoren")]
    for vel, lab in (("6", "±0,1 s"), ("15", "±0,25 s")):
        desc.append(summarize(jf[jf.vel == vel], f"Dohlen, Schwärme, zentrale Differenz {lab}"))
    L.append(md(pd.DataFrame(desc).set_index("Einheiten")))
    tcols = [f"T_{v}" for v in VARIANTS]
    L.append("\n**Dohlen je Schwarm (Geschwindigkeiten der Autoren):**\n")
    L.append(md(jfa[["flock", "frames", "N", *tcols, "speed", "nn_dist", "Phi"]].round(3).set_index("flock")))
    L.append("\n**Drohnen je Flug (2D, q = 5; Median über Fenster):**\n")
    L.append(md(d25.groupby("flight")[["N", *tcols, "Phi"]].median().round(3)))

    # (a) Tauben gegen Drohnen
    rng = np.random.default_rng(20261013)
    comps = [("primär: alle Kontrollflüge, 2D, q = 5", d25, pc5),
             ("nur kleine Taubenschwärme (7–10)", d25, pc5[pc5["size"] == "s"]),
             ("nur große Taubenschwärme (30–34)", d25, pc5[pc5["size"] == "b"]),
             ("Sensitivität q = 1", d21, pc1),
             ("Sensitivität Drohnen 3D (q = 5)", d35, pc5),
             ("Sensitivität Falkenflüge (≥ 50 m), q = 5", d25, pp5)]
    rows = []
    for v in VARIANTS:
        for name, D, Pg in comps:
            est, lo, hi = cluster_auc(by_cluster(D, "flight", f"T_{v}"), by_cluster(Pg, "flight", f"T_{v}"), rng,
                                      args.n_boot)
            rows.append({"Abzug": v, "Vergleich": name, "Fenster Drohnen/Tauben": f"{len(D)}/{len(Pg)}",
                         "Flüge Drohnen/Tauben": f"{D.flight.nunique()}/{Pg.flight.nunique()}",
                         "AUC′ = P(T_Drohne > T_Taube)": cell(est, lo, hi), "Lesart": reading_a(est, lo, hi)})
    A = pd.DataFrame(rows)
    A.to_csv(args.report.with_name(f"{prefix}_auc.csv"), index=False)
    L.append("\n## 3. (a) Tauben gegen echte Drohnen (Cluster-Bootstrap über Flüge)\n")
    L.append(md(A.set_index(["Abzug", "Vergleich"])))

    # (b) Realismus der Vogelmodelle (nur Abzüge, die die Simulation kennt)
    sec = 4
    if not args.no_sim_compare:
        rows = []
        for v in [v for v in VARIANTS if f"T_nn_{v}" in sbird.columns]:
            for name, real, n_sim in (("Tauben klein", pc5[pc5["size"] == "s"], 20),
                                      ("Tauben groß", pc5[pc5["size"] == "b"], 30),
                                      ("Dohlen (Schwärme)", jfa, 50)):
                s = sbird[sbird["N"] == n_sim][f"T_nn_{v}"]
                q10, q50, q90 = (float(s.quantile(p)) for p in (0.1, 0.5, 0.9))
                med = float(real[f"T_{v}"].median())
                rows.append({"Abzug": v, "echt": name, "echter Median": round(med, 3), "Sim. N": n_sim,
                             "Sim. Vögel 10 %": round(q10, 3), "Median": round(q50, 3), "90 %": round(q90, 3),
                             "Sim. D3 Median": round(float(sd3[sd3["N"] == n_sim][f"T_nn_{v}"].median()), 3),
                             "Lesart": reading_b(med, q10, q90)})
        L.append(f"\n## {sec}. (b) Realismus der Vogelmodelle (P3-Hauptlauf mit Wind, K0)\n")
        L.append(md(pd.DataFrame(rows).set_index(["Abzug", "echt"])))
        sec += 1

    # Tabellenkorrektur gegen Korrektur je Bild (3D-Daten)
    rows = []
    for name, df in (("Drohnen 3D, q = 5", d35), ("Drohnen 3D, q = 1", d31), ("Dohlen, Schwärme", jfa)):
        for v in [v for v in VARIANTS if v in TABLE_VARIANTS]:
            rows.append({"Daten": name, "Abzug": v, "T je Bild (Median)": round(float(df[f"T_{v}"].median()), 3),
                         "T Tabelle (Median)": round(float(df[f"Ttab_{v}"].median()), 3),
                         "Einheiten mit Tabellenwert": int(df[f"Ttab_{v}"].notna().sum())})
    if rows:
        L.append(f"\n## {sec}. Tabellenkorrektur gegen Korrektur je Bild (nur 3D, Tabelle gilt für N = 10–100)\n")
        L.append(md(pd.DataFrame(rows).set_index(["Daten", "Abzug"])))
    L.append(f"\nRechenzeit: {time.time() - t_start:.0f} s.\n")
    args.report.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    figure(W, sbird, sd3, panels, args.figdir / args.fig)
    keep_awake(False)


def figure(W: dict[str, pd.DataFrame], sbird: pd.DataFrame, sd3: pd.DataFrame, panels: tuple[str, ...],
           path: Path) -> None:
    """T je Abzug (ein Panel je Abzug) für echte Daten und, wo vorhanden, Simulation."""
    rng = np.random.default_rng(0)

    def cols(df: pd.DataFrame, fmt: str) -> dict:
        return {v: df[fmt.format(v)] if fmt.format(v) in df.columns else None for v in panels}

    rows = []
    for n in (20, 30, 50):
        rows.append((f"Vögel, simuliert, N = {n}", cols(sbird[sbird["N"] == n], "T_nn_{}"), COLORS["Vögel"], "o"))
    rows.append(("D3, simuliert, N = 30", cols(sd3[sd3["N"] == 30], "T_nn_{}"), COLORS["D3"], "o"))
    rows.append(("Drohnen, echt, 2D (q = 5)", cols(W["d25"], "T_{}"), INK, "s"))
    rows.append(("Drohnen, echt, 3D (q = 5)", cols(W["d35"], "T_{}"), INK, "s"))
    for lab, sz in (("klein (7–10)", "s"), ("groß (30–34)", "b")):
        rows.append((f"Tauben, echt, {lab}", cols(W["pc5"][W["pc5"]["size"] == sz], "T_{}"), INK2, "D"))
    rows.append(("Dohlen, echt, 2-s-Fenster", cols(W["jwa"], "T_{}"), INK2, "^"))
    rows.append(("Dohlen, echt, ganze Schwärme", cols(W["jfa"], "T_{}"), INK2, "^"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels) + 2.9, 6.2), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, v in zip(axes, panels):
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
        for k, (_, vals, col, mk) in enumerate(rows):
            if vals[v] is None:
                continue
            a = np.asarray(vals[v], float)
            a = a[np.isfinite(a)]
            y = k + rng.uniform(-0.18, 0.18, a.size)
            ax.plot(a, y, mk, color=col, markersize=3.5 if a.size > 20 else 6, alpha=0.4 if a.size > 20 else 0.8,
                    zorder=2)
            if a.size:
                q1, med, q3 = np.quantile(a, [0.25, 0.5, 0.75])
                halo = [pe.withStroke(linewidth=6, foreground=SURFACE)]
                ax.hlines(k, q1, q3, color=INK, linewidth=3, zorder=3, path_effects=halo)
                ax.plot(med, k, "|", color=INK, markersize=14, markeredgewidth=2.5, zorder=4, path_effects=halo)
        ax.axvline(0.0, color=MUTED, linewidth=1, linestyle=(0, (2, 3)))
        ax.set_title(TITLES.get(v, v) + (" · primär" if v == VARIANTS[0] else ""), color=INK, fontsize=10, loc="left")
        ax.set_xlabel("T_nn (biaskorrigiert)", color=INK2, fontsize=9)
    axes[0].set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5, color=INK)
    axes[0].invert_yaxis()
    fig.suptitle("Echte Vögel und echte Drohnen (grau/schwarz) gegen Simulation (farbig, wo vorhanden) · "
                 "Balken: Quartile, Strich: Median", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
