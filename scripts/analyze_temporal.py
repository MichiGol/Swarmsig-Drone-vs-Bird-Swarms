"""O20 (vorregistriert, ``reports/O20_Zeitsignatur_prereg.md``): zeitliche Signatur der Nachbarkopplung.

Asymmetrie A der zeitversetzten Korrelation zwischen Vorder- und Hintermann (nächste Nachbarn):
A > 0 heißt, der Hintere folgt dem Vorderen. Echte Drohnen, Tauben, Dohlen und Segler bei gemeinsam 5 Hz.

    python scripts/analyze_temporal.py
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

from analyze_p1 import md  # noqa: E402
from analyze_p3 import cell, ci  # noqa: E402
from analyze_real_birds import auc  # noqa: E402
from analyze_vasarhelyi import prepare as prepare_drones  # noqa: E402
from swarmsig.agents.neighbors import nearest_neighbor  # noqa: E402
from swarmsig.observables.empirical import main_cluster  # noqa: E402
from swarmsig.observables.modes import remove  # noqa: E402

DT, K, MIN_N, MIN_PAIRS = 0.2, 5, 6, 200
VARIANTS = ("L", "Q", "M")


def pad3(a: np.ndarray) -> np.ndarray:
    if a.shape[-1] == 3:
        return a
    out = np.zeros(a.shape[:-1] + (3,))
    out[..., : a.shape[-1]] = a
    return out


def signature(X: np.ndarray, V: np.ndarray, variant: str) -> dict:
    """X, V: (T, M, 3) mit NaN für fehlende Individuen. Liefert A, C(k), k*, τ_ac und die Paarzahl."""
    T = X.shape[0]
    S = np.full(X.shape[:2], np.nan)
    pairs = []
    for t in range(T):
        sp = np.linalg.norm(V[t], axis=1)
        ids = np.where(np.isfinite(X[t]).all(axis=1) & np.isfinite(V[t]).all(axis=1) & (sp > 0))[0]
        if ids.size < MIN_N:
            continue
        m = main_cluster(X[t, ids])
        ids = ids[m]
        if ids.size < MIN_N:
            continue
        x, u = X[t, ids], V[t, ids]
        if variant == "Q" and ids.size < 30:
            continue
        S[t, ids] = remove(variant, x, u).ds
        nn, _ = nearest_neighbor(x)
        head = u / np.linalg.norm(u, axis=1)[:, None]
        j_ahead = np.einsum("ij,ij->i", x[nn] - x, head) > 0  # Nachbar liegt in Flugrichtung von i
        lead = np.where(j_ahead, ids[nn], ids)
        follow = np.where(j_ahead, ids, ids[nn])
        pairs.append((t, lead, follow))
    var = np.nanmean(S**2)
    C, n0 = {}, 0
    for k in range(-K, K + 1):
        num = cnt = 0.0
        for t, lead, follow in pairs:
            tk = t + k
            if not 0 <= tk < T:
                continue
            a, b = S[t, lead], S[tk, follow]
            ok = np.isfinite(a) & np.isfinite(b)
            num += float(np.sum(a[ok] * b[ok]))
            cnt += int(ok.sum())
        C[k] = num / cnt / var if cnt else np.nan
        if k == 0:
            n0 = cnt
    pos = sum(C[k] for k in range(1, K + 1))
    neg = sum(C[-k] for k in range(1, K + 1))
    tot = sum(abs(C[k]) for k in range(-K, K + 1))
    ac = [np.nanmean(S[:-k] * S[k:]) / var for k in range(1, K + 1)]
    tau = np.nan
    prev = 1.0
    for k, a in enumerate(ac, start=1):
        if a < np.exp(-1):
            tau = DT * (k - 1 + (prev - np.exp(-1)) / (prev - a))
            break
        prev = a
    return {"A": (pos - neg) / tot if tot > 0 else np.nan, "D": pos - neg, "k_peak": int(max(C, key=lambda q: C[q])) * DT,
            "C0": C[0], "tau_ac": tau, "pairs0": n0, **{f"C{k:+d}": C[k] for k in range(-K, K + 1)}}


# ---------------------------------------------------------------- Datensätze
def drone_units(variant: str) -> list[dict]:
    rows = []
    for flight, (grid, X, F) in prepare_drones(pd.read_parquet(ROOT / "data/processed/vasarhelyi2018_raw.parquet")).items():
        V = np.full_like(X, np.nan)
        V[1:-1] = (X[2:] - X[:-2]) / (2 * DT)
        ok = np.zeros(F.shape, bool)
        ok[1:-1] = F[:-2] & F[1:-1] & F[2:]
        Xm, Vm = X.copy(), V.copy()
        Xm[~ok], Vm[~ok] = np.nan, np.nan
        n = int(round(10 / DT))
        for k0 in range(0, len(grid) - n, n):
            r = signature(Xm[k0:k0 + n + 1], Vm[k0:k0 + n + 1], variant)
            rows.append({"dataset": "echte Drohnen", "cluster": flight, **r})
    return rows


def pigeon_units(variant: str) -> list[dict]:
    P = pd.read_parquet(ROOT / "data/processed/papadopoulou2022_pigeons.parquet")
    rows = []
    for flight, g in P[P.condition == "c"].groupby("flight"):
        birds = np.sort(g.pigeon.unique())
        idx = {b: j for j, b in enumerate(birds)}
        T = int(g.frame.max()) + 2
        X = np.full((T, len(birds), 2), np.nan)
        X[g.frame.to_numpy(), g.pigeon.map(idx).to_numpy()] = g[["x", "y"]].to_numpy()
        V = np.full_like(X, np.nan)
        V[1:-1] = (X[2:] - X[:-2]) / (2 * DT)
        slow = np.nanmean(np.linalg.norm(V, axis=2), axis=1) < 10.0  # Flugphase
        X3, V3 = pad3(X), pad3(V)
        X3[slow], V3[slow] = np.nan, np.nan
        n = int(round(10 / DT))
        for k0 in range(int(g.frame.min()), T - n, n):
            r = signature(X3[k0:k0 + n + 1], V3[k0:k0 + n + 1], variant)
            rows.append({"dataset": f"Tauben ({'klein' if g['size'].iloc[0] == 's' else 'groß'})", "cluster": flight, **r})
    return rows


def jackdaw_units(variant: str) -> list[dict]:
    J = pd.read_parquet(ROOT / "data/processed/ling2019_jackdaws.parquet")
    J = J[(J.flock != 3) & (J.frame % 12 == 0)]
    rows = []
    for flock, g in J.groupby("flock"):
        ids = np.sort(g.id.unique())
        idx = {b: j for j, b in enumerate(ids)}
        t = (g.frame // 12).to_numpy()
        t = t - t.min()
        X = np.full((t.max() + 1, len(ids), 3), np.nan)
        V = np.full_like(X, np.nan)
        j = g.id.map(idx).to_numpy()
        X[t, j] = g[["x", "y", "z"]].to_numpy()
        V[t, j] = g[["u", "v", "w"]].to_numpy()
        rows.append({"dataset": "Dohlen", "cluster": f"Schwarm {flock}", **signature(X, V, variant)})
    return rows


def swift_units(variant: str, path: Path) -> list[dict]:
    rows = []
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys(), key=int)[::6]  # ≈ 5 Hz
        n = int(round(10 / DT))
        for w0 in range(0, len(keys) - n, n):
            frames = []
            for k in keys[w0:w0 + n + 1]:
                g = f[k]
                frames.append((g["tid"][()], np.column_stack([g["x"][()], g["y"][()], g["z"][()]]),
                               np.column_stack([g["vx"][()], g["vy"][()], g["vz"][()]])))
            tids = np.unique(np.concatenate([fr[0] for fr in frames]))
            idx = {tid: j for j, tid in enumerate(tids)}
            X = np.full((len(frames), len(tids), 3), np.nan)
            V = np.full_like(X, np.nan)
            for t, (tid, x, v) in enumerate(frames):
                jj = np.array([idx[q] for q in tid], int)
                X[t, jj], V[t, jj] = x, v
            rows.append({"dataset": "Segler", "cluster": f"Fenster {w0 // n}", **signature(X, V, variant)})
    return rows


def boot_median(df: pd.DataFrame, rng: np.random.Generator, n_boot: int = 2000) -> tuple[float, float, float]:
    groups = {k: g["A"].to_numpy() for k, g in df.groupby("cluster")}
    keys = list(groups)
    est = float(np.median(df["A"]))
    reps = [np.median(np.concatenate([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))]))
            for _ in range(n_boot)]
    lo, hi = ci(np.array(reps))
    return est, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swifts", type=Path, default=ROOT / "data/raw/evangelista2017_swifts/A_nodes.hdf5")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "O20_Zeitsignatur_results.md")
    args = ap.parse_args()
    t0 = time.time()
    rows = []
    for v in VARIANTS:
        for fn in (drone_units, pigeon_units, jackdaw_units):
            rows += [{"Abzug": v, **r} for r in fn(v)]
        rows += [{"Abzug": v, **r} for r in swift_units(v, args.swifts)]
        print(f"Abzug {v} fertig ({time.time() - t0:.0f} s)", flush=True)
    U = pd.DataFrame(rows)
    U = U[U["pairs0"] >= MIN_PAIRS]
    U.to_csv(args.report.with_name("O20_Zeitsignatur_units.csv"), index=False)
    rng = np.random.default_rng(20261016)
    L = ["# O20: zeitliche Signatur der Nachbarkopplung – Ergebnisse (automatisch erzeugt)\n",
         "Vorregistrierung: `reports/O20_Zeitsignatur_prereg.md`. A > 0: der Hintermann folgt dem Vordermann. "
         "Median [95-%-Intervall, Cluster-Bootstrap].\n"]
    for v in VARIANTS:
        d = U[U["Abzug"] == v]
        t = []
        for name, g in d.groupby("dataset"):
            est, lo, hi = boot_median(g, rng)
            t.append({"Datensatz": name, "Einheiten": len(g), "Cluster": g.cluster.nunique(), "A": cell(est, lo, hi),
                      "D (Median)": round(float(g.D.median()), 4),
                      "C(0)": round(float(g.C0.median()), 3), "Spitze k* [s]": round(float(g.k_peak.median()), 2),
                      "τ_ac [s]": round(float(g.tau_ac.median()), 2)})
        L.append(f"\n## Abzug {v}{' (primär)' if v == 'L' else ''}\n")
        L.append(md(pd.DataFrame(t).set_index("Datensatz")))
        dr = d[d.dataset == "echte Drohnen"]["A"].to_numpy()
        L.append("\nAUC′ = P(A_Vogel > A_Drohne): " + ", ".join(
            f"{name} {auc(g['A'].to_numpy(), dr):.2f}" for name, g in d.groupby("dataset") if name != "echte Drohnen") + "\n")
        prof = d.groupby("dataset")[[f"C{k:+d}" for k in range(-K, K + 1)]].median().round(3)
        L.append("\n**C(k) je Datensatz (Median; Spalten k = −5 … +5, Δ = 0,2 s):**\n")
        L.append(md(prof))
    L.append(f"\nRechenzeit: {time.time() - t0:.0f} s.\n")
    args.report.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
