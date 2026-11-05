"""Kalibrierung „light“ der Vogelmodelle (4.8) als ABC-Verwerfungsverfahren, Version 2.

Je Modell (B1, B3) werden Parametersätze aus breiten Prioren gezogen, je ein Schwarm
(N = 50, 40 s, Messung bei t = 20, 22, …, 40 s) simuliert und akzeptiert, wenn alle
Kriterien erfüllt sind. Ergebnis: Parquet-Datei mit allen akzeptierten Sätzen (= Posterior),
aus der die Läufe je eine Zeile ziehen (``birds.<Modell>.calibration_posterior``).

Kriterien und Herkunft:
  B1  Polarisation Φ ∈ [0,84; 0,995]  (L: Starendaten der Cavagna-Gruppe, 4.8)
      Nächster-Nachbar-Abstand d_nn ∈ [0,7; 1,5] m  (ANNAHME bis offener Punkt 3 geklärt)
  B3  Φ ∈ [0,84; 0,995]  (ANNAHME, offener Punkt 15)
      d_nn ∈ [1,0; 3,0] m  (ANNAHME, offener Punkt 15)
  beide: stationär und zusammenhängend (D, 4.9 und 14.23): Schwarmgröße L_g ändert sich
      zwischen Anfang und Ende des Messfensters um höchstens 15 %, und am Ende bildet der
      Schwarm einen einzigen Verbund (Kanten bis 4·d_nn).
Vereinfachungen gegenüber 4.8: keine Skalenfreiheits-Ziele (bei N = 50 nicht prüfbar), keine
Ziele für Speed-Schwankungen (Literaturwerte fehlen); κ_s und u_* bleiben breit verteilt.

Zeitschritt je akzeptiertem Satz (4.9): ρ_max = Maximum aus Jacobi-Spektralradius (an drei
Momentaufnahmen), Speed-Steifigkeit 3 κ_s (u_q/u_*)² mit dem 99,9-%-Quantil u_q von |s − s0|
und Harter-Kern-Steifigkeit am kleinsten beobachteten Abstand; Δt = größter zulässiger Wert
mit Δt ≤ 0,1/ρ_max.

    python scripts/calibrate_bird_model.py --model B1 --samples 400 --jobs 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows-Konsole (cp1252) kann z. B. ρ, Δ nicht ausgeben

from swarmsig.agents.bird import BirdModel, BirdParams  # noqa: E402
from swarmsig.agents.terms import Navigation  # noqa: E402
from swarmsig.config import Choice, LogUniform, Uniform, load_config, sample  # noqa: E402
from swarmsig.dynamics.fast import make_fast, simulate_fast  # noqa: E402
from swarmsig.dynamics.init import ellipsoid_positions, initial_velocities  # noqa: E402
from swarmsig.dynamics.integrator import dt_from_rule, kappa_eff, spectral_radius  # noqa: E402
from swarmsig.observables import size  # noqa: E402
from swarmsig.observables.modes import remove  # noqa: E402
from swarmsig.observables.smalln import NNBiasTable, c_nn  # noqa: E402
from swarmsig.rng import experiment_rng  # noqa: E402
from swarmsig.state import SwarmState  # noqa: E402

PRIORS = {  # ANNAHME: breite Prioren für die Kalibrierung
    # Version 3 (nach Prior-Prüfung): stabile Schwärme waren fast immer zu geordnet (Φ > 0,995)
    # -> mehr Rauschen relativ zur Ausrichtung; Zusammenhalt über n_coh Nachbarn staucht auf
    # d_nn ≈ 0,45·r0 -> r0-Bereich nach oben verschoben.
    "J_per_s": LogUniform(dist="loguniform", low=0.5, high=5.0),
    "sigma": LogUniform(dist="loguniform", low=1.0, high=6.0),
    "beta_mps2": LogUniform(dist="loguniform", low=0.3, high=5.0),
    "kappa_s_per_s": LogUniform(dist="loguniform", low=0.5, high=5.0),
    "u_star_mps": LogUniform(dist="loguniform", low=0.3, high=3.0),
    # Zusammenhalt über die n_coh nächsten Nachbarn (Entscheidung 2026-10-11, gegen Zerfall)
    "n_coh": Choice(dist="choice", values=list(range(10, 31))),
}
# B1s (O3c-2, vorregistriert in reports/O3c2_prereg.md): stark gekoppelte Vögel mit Zielen aus den echten Dohlen
# (Ling et al. 2019; Werte aus O3b). Struktur wie B1 (topologisch, marginale Speed-Regelung), Prioren in Richtung
# stärkerer Ausrichtung, weniger Rauschen und schwächerer Speed-Regelung.
PRIORS_STRONG = {
    "J_per_s": LogUniform(dist="loguniform", low=1.0, high=10.0),
    "sigma": LogUniform(dist="loguniform", low=0.2, high=3.0),
    "beta_mps2": LogUniform(dist="loguniform", low=0.3, high=5.0),
    "kappa_s_per_s": LogUniform(dist="loguniform", low=0.05, high=2.0),
    "u_star_mps": LogUniform(dist="loguniform", low=0.5, high=5.0),
    "n_coh": Choice(dist="choice", values=list(range(10, 31))),
}
PRIORS_WIDE = {**PRIORS_STRONG,  # Erkundung O3c-2: noch stärkere Ausrichtung, weniger Rauschen, schwächere Speed-Regelung
               "J_per_s": LogUniform(dist="loguniform", low=1.0, high=30.0),
               "sigma": LogUniform(dist="loguniform", low=0.05, high=3.0),
               "kappa_s_per_s": LogUniform(dist="loguniform", low=0.01, high=2.0)}
PRIORS_BY = {"B1": PRIORS, "B3": PRIORS, "B1s": PRIORS_STRONG}
R0_PRIOR = {"B1": Uniform(dist="uniform", low=1.0, high=3.5), "B3": Uniform(dist="uniform", low=2.0, high=6.0),
            "B1s": Uniform(dist="uniform", low=2.0, high=7.0)}
TARGETS = {"B1": {"Phi": (0.84, 0.995), "nn_dist": (0.7, 1.5)},
           "B3": {"Phi": (0.84, 0.995), "nn_dist": (1.0, 3.0)},
           # Dohlen (O3b, 5 Schwärme): Φ 0,944–0,991, d_nn 2,1–2,8 m, T_L 0,40–0,61, T_Q 0,06–0,35
           "B1s": {"Phi": (0.93, 0.995), "nn_dist": (1.6, 3.0), "T_L": (0.35, 0.65), "T_Q": (0.15, 0.45)}}
BIAS_TABLE = ROOT / "data" / "processed" / "nn_bias_table_robust.parquet"
MAX_L_CHANGE = 0.15
S0 = {"B1": 11.0, "B3": 14.0, "B1s": 12.0}
N_CAL, T_BURN, T_END, DT = 50, 20.0, 40.0, 0.0025


def n_clusters(x: np.ndarray, link: float) -> int:
    pairs = cKDTree(x).query_pairs(link, output_type="ndarray")
    n = x.shape[0]
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    return int(connected_components(g, directed=False)[0])


def set_rho(m: BirdModel, snaps: list[SwarmState], p: BirdParams) -> tuple[float, float, float]:
    """ρ_max je Parametersatz: Jacobi (3 Aufnahmen), Speed-Steifigkeit (u_q), harter Kern (r_min)."""
    n = snaps[0].n
    jac = []
    for st in snaps[-3:]:
        def rhs(z: np.ndarray, st=st) -> np.ndarray:
            x, v = z[: 3 * n].reshape(n, 3), z[3 * n:].reshape(n, 3)
            return m.rhs(st.replace(x=x, v=v, u=v))
        jac.append(spectral_radius(rhs, np.concatenate([st.x.ravel(), st.v.ravel()])))
    dev = np.concatenate([np.abs(np.linalg.norm(s.v, axis=1) - p.s0) for s in snaps])
    u_q = float(np.quantile(dev, 0.999))
    r_min = min(float(cKDTree(s.x).query(s.x, k=2)[0][:, 1].min()) for s in snaps)
    r_eff = max(r_min, 0.5 * p.r_hc)
    k_hc = 6.0 * p.beta * p.r_hc**6 / r_eff**7 if r_eff < 2 * p.r_hc else 0.0
    rho = max(max(jac), kappa_eff(p.kappa_s, p.speed_control, u_q, p.u_star), np.sqrt(k_hc), 2 * p.J)
    return float(rho), u_q, r_min


def one(model: str, k: int, cfg_path: str, seed: int, wide: bool = False) -> dict[str, object]:
    cfg = load_config(cfg_path)
    bc = cfg.birds["B1" if model == "B1s" else model]
    rng = experiment_rng(seed, f"calibration_{model}_{k}")
    priors = PRIORS_WIDE if wide and model == "B1s" else PRIORS_BY[model]
    pr = {name: float(sample(spec, rng)) for name, spec in priors.items()}
    pr["r0_m"] = float(sample(R0_PRIOR[model], rng))
    p = BirdParams(
        model=model, neighborhood=str(sample(bc.neighborhood, rng)), n_c=int(sample(bc.n_c, rng)),
        r_c=float(sample(bc.r_c_m, rng)), speed_control=str(sample(bc.speed_control, rng)),
        s0=S0[model], J=pr["J_per_s"], sigma=pr["sigma"], kappa_s=pr["kappa_s_per_s"],
        u_star=pr["u_star_mps"], beta=pr["beta_mps2"], r0=pr["r0_m"], r_w=float(sample(bc.r_w_m, rng)),
        r_hc=float(sample(bc.r_hc_m, rng)), eta_h=float(sample(bc.eta_h, rng)),
        tau0=float(sample(bc.tau0_s, rng)), c_comp=0.0, tau_est=5.0, n_coh=int(pr["n_coh"]))
    e_T = np.array([0.0, 1.0, 0.0])
    nav = Navigation("transit", z0=100.0, kappa_z=0.05, e_T=e_T, kappa_n=0.5)
    m = BirdModel(p, nav, DT, N_CAL, rank_rng=rng)
    x0 = ellipsoid_positions(N_CAL, np.array([0.0, 0.0, 100.0]), p.r0, cfg.init.aspect, e_T, rng)
    v0 = initial_velocities(N_CAL, p.s0, e_T, cfg.init.speed_noise_rel, rng)
    s0 = SwarmState.initial(0.0, x0, v0)
    m.start(s0)
    snaps: list[SwarmState] = []
    times = np.arange(T_BURN, T_END + 1e-9, 0.5 if model == "B1s" else 2.0)
    with np.errstate(all="ignore"):
        simulate_fast(make_fast(m, DT), s0, dt=DT, n_steps=int(round(T_END / DT)), noise_rng=rng,
                      event_times=times, callbacks=(lambda a, b, s: snaps.append(b),))
    row: dict[str, object] = {**pr, "neighborhood": p.neighborhood, "speed_control": p.speed_control,
                              "n_c": p.n_c, "r_c_m": p.r_c}
    if len(snaps) < len(times) or not all(np.isfinite(s.x).all() for s in snaps):
        row.update(accepted=False, reason="nicht endlich")
        return row
    phi = np.array([size.polarization(s.u) for s in snaps])
    nn = np.array([size.nn_distance_mean(s.x) for s in snaps])
    lg = np.array([size.L_g(s.x) for s in snaps])
    l_change = float(lg[-3:].mean() / lg[:3].mean() - 1.0)
    clusters = n_clusters(snaps[-1].x, 4.0 * float(np.median(nn)))
    row.update(Phi=float(phi.mean()), nn_dist=float(nn.mean()), L_g=float(lg.mean()),
               L_change=l_change, clusters=clusters)
    if model == "B1s":  # Korrelationsziele: T = Mittel(c_nn) − Tabellen-Bias (N = N_CAL)
        table = NNBiasTable.load(BIAS_TABLE)
        for v in ("L", "Q", "M", "A7"):  # M, A7 nur zur Diagnose
            c = np.nanmean([c_nn(st.x, remove(v, st.x, st.u).ds) for st in snaps])
            row[f"T_{v}"] = float(c - table.bias(N_CAL, v))
        row["sd_s"] = float(np.mean([np.std(np.linalg.norm(st.u, axis=1)) for st in snaps]))
    t = TARGETS[model]
    ok_targets = all(lo <= row[key] <= hi for key, (lo, hi) in t.items())  # type: ignore[operator]
    ok_stable = abs(l_change) <= MAX_L_CHANGE and clusters == 1
    row["accepted"] = bool(ok_targets and ok_stable)
    row["reason"] = "" if row["accepted"] else ("Ziele" if not ok_targets else "instabil")
    if row["accepted"]:
        rho, u_q, r_min = set_rho(m, snaps, p)
        dt, _ = dt_from_rule(rho, safety=cfg.dynamics.dt_safety, allowed=cfg.dynamics.dt_allowed_s)
        row.update(rho_set=rho, u_q=u_q, r_min=r_min, dt_s=dt)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["B1", "B3", "B1s"], required=True)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "base.yaml")
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--wide", action="store_true", help="B1s: breitere Prioren (Erkundung O3c-2)")
    args = ap.parse_args()
    out = args.out or ROOT / "data" / "processed" / f"{args.model.lower()}_posterior.parquet"
    t0 = time.perf_counter()
    rows = Parallel(n_jobs=args.jobs)(
        delayed(one)(args.model, k, str(args.config), args.seed, args.wide) for k in range(args.samples))
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out.with_name(out.stem + "_all.parquet"), index=False)
    acc = df[df["accepted"]].reset_index(drop=True)
    acc.to_parquet(out, index=False)
    print(f"{args.model}: {len(acc)}/{len(df)} akzeptiert ({len(acc) / len(df):.0%}); "
          f"abgelehnt: {df['reason'].value_counts().to_dict()}; {time.perf_counter() - t0:.0f} s")
    if len(acc):
        cols = list(PRIORS_BY[args.model]) + ["r0_m", "Phi", "nn_dist", "L_change", "rho_set", "dt_s"]
        cols += [c for c in ("T_L", "T_Q", "sd_s") if c in acc.columns]
        print(acc[cols].describe().loc[["min", "50%", "max"]].round(3).to_string())
        print("Δt-Verteilung:", acc["dt_s"].value_counts().sort_index().to_dict())
        print("Varianten:", acc.groupby(["neighborhood", "speed_control"]).size().to_dict())


if __name__ == "__main__":
    main()
