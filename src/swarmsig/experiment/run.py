"""Ein Monte-Carlo-Lauf (3.1): Designpunkt -> Schwarm je Klasse -> Sensor -> Kenngrößen -> Zeile.

Innerhalb eines Designpunkts teilen die Klassen nur die Kovariaten θ (unabhängige
Realisierungen, 8.5). Kontrollvarianten verwenden dieselben Klassen-Seeds (CRN, 8.6).
Umgesetzt sind die Kontrollen K0, K-Dichte (8.6), K-Wind (8.6, ab P3) und K-Calm (P3: derselbe
Designpunkt ohne Wind, CRN). Radar (P2) fehlt noch.

K-Wind (Festlegung 2026-10-11, in 8.6 offen gelassen): Die **Drohnen** übernehmen die
Windreaktion (c_B, τ_est,B) des Vogels desselben Designpunkts, analog zu K-Dichte.
"""

from __future__ import annotations

import math
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..agents.bird import BirdModel, bird_rho_max, draw_bird_params
from ..agents.drone import DRONE_KINDS, DroneModel, draw_drone_params, drone_rho_max
from ..agents.realism import draw_realism_params
from ..config import Config, config_hash, sample
from ..dynamics.init import ellipsoid_positions, initial_velocities, stationarity_ok
from ..dynamics.fast import make_fast, simulate_fast
from ..dynamics.integrator import dt_from_rule, simulate
from ..dynamics.wind_transfer import WindTransfer
from ..environment.scenario import cos_los, los_class, plan_flight
from ..environment.weather import WindSetup, build_wind
from ..observables import size
from ..observables.smalln import NNBiasTable
from ..observables.statistics import IdealStatistics
from ..rng import ClassSeeds, PointSeeds
from ..sensors.ideal import IdealSensor
from ..sensors.interpolate import FrameRecorder
from ..state import SwarmState, Trajectory
from .design import Covariates

BIRD_KINDS = ("B1", "B3")


@dataclass(frozen=True)
class Provenance:
    experiment_id: str
    config_hash: str
    git_commit: str

    @staticmethod
    def of(cfg: Config, repo: str | Path | None = None) -> "Provenance":
        return Provenance(cfg.experiment.id, config_hash(cfg), git_commit(repo))


def git_commit(repo: str | Path | None = None) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                             text=True, timeout=5, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "nogit"


# ---------------------------------------------------------------------------
# Zeitschritt
# ---------------------------------------------------------------------------


def choose_dt(cfg: Config, rho: float, family: str | None = None) -> tuple[float, bool]:
    """Zeitschritt: Pilotwert je Modellfamilie (falls gesetzt), sonst Zeitschrittregel (4.9).

    Der Pilotwert stammt aus der primären Definition (Jacobi-Spektralradius im Pilotlauf);
    ``ok`` vergleicht trotzdem mit der analytischen Schätzung und wird protokolliert.
    """
    dyn = cfg.dynamics
    pilot = {"bird": dyn.dt_bird_s, "drone": dyn.dt_drone_s}.get(family or "")
    if pilot is not None:
        return pilot, pilot <= dyn.dt_safety / rho if rho > 0 else True
    if dyn.auto_reduce_dt:
        return dt_from_rule(rho, safety=dyn.dt_safety, dt_max=dyn.dt_s, allowed=dyn.dt_allowed_s)
    return dyn.dt_s, dyn.dt_s <= dyn.dt_safety / rho if rho > 0 else True


# ---------------------------------------------------------------------------
# Stationaritäts-Stichprobe während der Einschwingzeit
# ---------------------------------------------------------------------------


class _BurnSampler:
    def __init__(self, t_burn: float, interval: float) -> None:
        self.t_burn = t_burn
        self.interval = interval
        self.next_t = interval
        self.t: list[float] = []
        self.series: dict[str, list[float]] = {"Phi": [], "mean_speed": [], "L_g": [], "nn_dist": []}

    def __call__(self, prev: SwarmState, new: SwarmState, step: int) -> None:  # noqa: ARG002
        if new.t > self.t_burn + 1e-9 or new.t + 1e-9 < self.next_t:
            return
        self.next_t += self.interval
        self.t.append(new.t)
        self.series["Phi"].append(size.polarization(new.u))
        self.series["mean_speed"].append(size.mean_speed(new.u))
        self.series["L_g"].append(size.L_g(new.x))
        self.series["nn_dist"].append(size.nn_distance_mean(new.x))


# ---------------------------------------------------------------------------
# Ein Lauf einer Klasse
# ---------------------------------------------------------------------------


def _rng_for(cs: ClassSeeds, stream: str) -> Callable[[str], np.random.Generator]:
    return lambda name: cs.rng(stream, name)


@dataclass
class ClassSetup:
    """Fertig aufgebauter Lauf einer Klasse: Modell, Startzustand und Zeitschritt."""

    model: BirdModel | DroneModel
    state: SwarmState
    dt: float
    dt_ok: bool
    rho: float
    params_row: dict[str, object]
    c_comp: float
    tau_est: float
    latency_rounding: float
    wind: WindSetup | None = None


def setup_class(cls: str, theta: Covariates, ps: PointSeeds, cfg: Config, *,
                spacing_override: float | None = None,
                wind_reaction: tuple[float, float] | None = None,
                wind_off: bool = False) -> ClassSetup:
    """Zieht die Parameter, wählt Δt und erzeugt Modell und Startzustand (ohne zu simulieren).

    ``wind_reaction = (c, τ_est)`` überschreibt die Windreaktion (Kontrolle K-Wind);
    ``wind_off`` rechnet ohne Wind (Kontrolle K-Calm). Beides ändert keine Zufallsziehung.
    """
    cs = ps.for_class(cls)
    par, init = _rng_for(cs, "par"), _rng_for(cs, "init")
    N = theta.N
    plan = plan_flight(
        mode=theta.flight_mode, R=theta.R_m, azimuth=theta.azimuth_rad, z0=theta.z0_m,
        h_r=cfg.site.h_r_m, heading=theta.heading_rad, circle_radius=theta.circle_radius_m,
        s0=theta.s0_mps, T_burn=theta.T_burn_s,
        kappa_n=float(sample(cfg.nav.kappa_n_per_s, par("kappa_n"))),
        kappa_z=float(sample(cfg.nav.kappa_z_per_s2, par("kappa_z"))),
        start_angle=float(init("start_angle").uniform(0.0, 2 * np.pi)),
    )

    row: dict[str, object] = {}
    latency_rounding = 0.0
    if cls in BIRD_KINDS:
        bp = draw_bird_params(cls, cfg.birds[cls], theta.s0_mps, par)
        rho = bird_rho_max(bp, cfg.dynamics.u_q_marginal_mps)
        if bp.dt_hint > 0:  # Zeitschritt je Parametersatz aus der Kalibrierung (Regel 4.9)
            dt, dt_ok = bp.dt_hint, True
        else:
            dt, dt_ok = choose_dt(cfg, rho, "bird")
        model: BirdModel | DroneModel = BirdModel(bp, plan.nav, dt, N, rank_rng=par("rank"))
        spacing0 = bp.r0
        row.update(bp.as_row("b_"))
        c_comp, tau_est = bp.c_comp, bp.tau_est
    elif cls in DRONE_KINDS:
        dp = draw_drone_params(cls, cfg.drones, theta.s0_mps, par, spacing_override)
        rp = draw_realism_params(cfg.realism, par)
        rho = drone_rho_max(dp, rp)
        dt, dt_ok = choose_dt(cfg, rho, "drone")
        model = DroneModel(dp, rp, plan.nav, dt, N, _rng_for(cs, "agent"),
                           cfg.realism.max_packet_age_s, plan.reference)
        latency_rounding = model.latency_rounding_s
        spacing0 = dp.spacing
        row.update(dp.as_row("d_"))
        row.update(rp.as_row("d_"))
        c_comp, tau_est = 1.0, dp.tau_est
    else:
        raise ValueError(f"Klasse {cls!r} ist in P1 nicht verfügbar")
    if wind_reaction is not None:
        c_comp, tau_est = float(wind_reaction[0]), float(wind_reaction[1])
        model.transfer = WindTransfer(c_comp, tau_est)

    wind = None
    if cfg.environment.wind != "none" and not wind_off:
        wind = build_wind(cfg.environment, theta.weather, theta.z0_m, dt, cs.rng("env", "turbulence"))
    x0 = ellipsoid_positions(N, plan.start_centroid, spacing0, cfg.init.aspect, plan.start_direction,
                             init("positions"))
    v0 = initial_velocities(N, theta.s0_mps, plan.start_direction, cfg.init.speed_noise_rel,
                            init("velocities"))
    w0 = wind.wind.sample(x0, 0.0) if wind is not None else None
    state = SwarmState.initial(0.0, x0, v0, w0, c_comp)
    model.start(state)
    return ClassSetup(model, state, dt, dt_ok, rho, row, c_comp, tau_est, latency_rounding, wind)


def _trim(traj: Trajectory, d: int) -> Trajectory:
    """Entfernt d Randbilder an beiden Enden (Aufzeichnung für das größte q, Auswertung für kleineres q)."""
    if d <= 0:
        return traj
    return Trajectory(traj.t[d:-d].copy(), traj.x[d:-d], traj.u[d:-d])


def run_class(cls: str, theta: Covariates, ps: PointSeeds, cfg: Config, prov: Provenance, *,
              control: str = "K0", spacing_override: float | None = None,
              wind_reaction: tuple[float, float] | None = None, wind_off: bool = False,
              nn_table: NNBiasTable | None = None, keep_curves: bool = False,
              ) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    t_wall = time.perf_counter()
    cs = ps.for_class(cls)
    sens = _rng_for(cs, "sensor")
    N = theta.N
    su = setup_class(cls, theta, ps, cfg, spacing_override=spacing_override,
                     wind_reaction=wind_reaction, wind_off=wind_off)
    model, state, dt, dt_ok, rho = su.model, su.state, su.dt, su.dt_ok, su.rho
    row, c_comp, tau_est, latency_rounding = su.params_row, su.c_comp, su.tau_est, su.latency_rounding

    sensor = IdealSensor(
        sigma_pos=float(sample(cfg.sensor.sigma_pos_m, sens("sigma_pos"))),
        f_cam=float(sample(cfg.sensor.f_cam_hz, sens("f_cam"))),
        q=cfg.sensor.q_frames,
        p_miss=float(sample(cfg.sensor.p_miss, sens("p_miss"))),
    )
    # Aufgezeichnet wird für das größte q aller Sensorvarianten; die Grundauswertung nutzt den
    # gekürzten Ausschnitt und bleibt damit unverändert (deterministische Interpolation).
    q_max = max([sensor.q] + [v.q_frames for v in cfg.sensor.variants])
    frames = IdealSensor(sensor.sigma_pos, sensor.f_cam, q_max, sensor.p_miss).frame_times(
        theta.T_burn_s, theta.T_obs_s)
    n_steps = int(math.ceil(frames[-1] / dt - 1e-9))
    recorder = FrameRecorder(frames, N)
    interval = cfg.dynamics.stationarity.sample_interval_s
    burn = _BurnSampler(theta.T_burn_s, interval)
    noise = cs.rng("agent", "process_noise")
    use_fast = cfg.dynamics.backend == "numba" and N <= cfg.dynamics.fast_max_n
    wind = su.wind.wind if su.wind is not None else None
    hold = su.wind.hold_steps if su.wind is not None else 1
    if use_fast:
        burn_times = np.arange(interval, theta.T_burn_s + 1e-9, interval)
        simulate_fast(make_fast(model, dt, wind, hold), state, dt=dt, n_steps=n_steps, noise_rng=noise,
                      event_times=np.concatenate([burn_times, frames]), callbacks=(burn, recorder))
    else:
        simulate(model, state, dt=dt, n_steps=n_steps, noise_rng=noise, wind=wind,
                 wind_hold_steps=hold, callbacks=(burn, recorder))
    traj_full = recorder.trajectory()
    traj = _trim(traj_full, q_max - sensor.q)
    obs = sensor.observe(traj, sens("ideal_noise"))

    stats = IdealStatistics(
        cfg.observables.mode_removal, cfg.observables.dr_m,
        max_n_all_pairs=cfg.observables.max_n_all_pairs, n_pair_samples=cfg.observables.n_pair_samples,
        min_pairs_per_bin=cfg.observables.min_pairs_per_bin, rng=sens("pair_sampling"),
        nn_table=nn_table)
    stats.add_observation(obs, cfg.observables.frame_stride)
    srow, curves = stats.finalize(keep_curves)

    cos_k = np.array([cos_los(traj.x[k].mean(axis=0), traj.u[k].mean(axis=0), cfg.site.h_r_m)
                      for k in range(traj.n_frames)])
    # Fragmentierung (14.23): Teilgruppen am Ende (Kanten bis 4·d_nn), Größenwachstum im Fenster
    nn_end = size.nn_distance_mean(traj.x[-1])
    frag_clusters = size.n_clusters(traj.x[-1], 4.0 * nn_end) if np.isfinite(nn_end) else -1
    L_growth = size.L_g(traj.x[-1]) / size.L_g(traj.x[0]) - 1.0
    stat_ok, pvals = stationarity_ok(np.asarray(burn.t), {k: np.asarray(v) for k, v in burn.series.items()},
                                     cfg.dynamics.stationarity.window_fraction,
                                     cfg.dynamics.stationarity.alpha)
    flags = [] if dt_ok else ["dt_rule_violated"]
    for v in cfg.observables.mode_removal:
        for key in (f"xi_s_flag_{v}", f"xi_perp_flag_{v}"):
            if srow.get(key) not in (None, "ok"):
                flags.append(f"{key}={srow[key]}")

    out: dict[str, object] = {
        # Identität
        "experiment_id": prov.experiment_id, "design_point_id": theta.index, "class": cls,
        "control": control,
        "sweep_value": float("nan") if cfg.experiment.sweep_value is None else float(cfg.experiment.sweep_value),
        "git_commit": prov.git_commit,
        "seed_entropy": ps.entropy_tag, "config_hash": prov.config_hash,
        "hypothetical": bool(theta.hypothetical and cls in DRONE_KINDS),
        # Szenario
        "N": N, "N_band": theta.N_band, "bird_model": theta.bird_model, "s0_mps": theta.s0_mps,
        "R_m": theta.R_m, "z0_m": theta.z0_m, "azimuth_rad": theta.azimuth_rad,
        "flight_mode": theta.flight_mode, "heading_rad": theta.heading_rad,
        "circle_radius_m": theta.circle_radius_m, "cos_los_mean": float(np.mean(cos_k)),
        "abs_cos_los_mean": float(np.mean(np.abs(cos_k))),
        "los_class": los_class(float(np.mean(np.abs(cos_k))), cfg.observables.los_class_edges),
        "T_obs_s": theta.T_obs_s, "T_burn_s": theta.T_burn_s,
        # Wind (P3; ohne Wind NaN)
        "wind_on": su.wind is not None,
        "U10": theta.weather.U10 if su.wind is not None else float("nan"),
        "alpha": theta.weather.alpha if su.wind is not None else float("nan"),
        "wind_dir": theta.weather.wind_dir_rad if su.wind is not None else float("nan"),
        "I_u10": theta.weather.I_u10 if su.wind is not None else float("nan"),
        "weather_source": theta.weather.source, "weather_station": theta.weather.station,
        "weather_time": theta.weather.time_utc,
        "c_comp": c_comp, "tau_est": tau_est,
        # Sensor
        "sensor_type": "ideal", "detected": True, "sigma_pos_m": sensor.sigma_pos,
        "f_cam_hz": sensor.f_cam, "q_frames": sensor.q, "p_miss": sensor.p_miss,
        # Kontext
        "mean_ground_speed": float(np.mean(np.linalg.norm(traj.u.mean(axis=1), axis=1))),
        # Diagnose
        "backend": "numba" if use_fast else "numpy",
        "frag_clusters": frag_clusters, "L_growth_obs": float(L_growth),
        "stationary_ok": bool(stat_ok), "stationarity_pmin": float(min(pvals.values())) if pvals else float("nan"),
        "validity_flags": ";".join(flags), "dt_s": dt, "rho_max_est": rho, "dt_rule_ok": bool(dt_ok),
        "latency_rounding_s": latency_rounding,
    }
    wrow = su.wind.row(dt) if su.wind is not None else {
        k: float("nan") for k in ("U_z0", "sigma_u_z0", "sigma_w_z0", "L_u_z0", "wind_modes",
                                  "turb_resolved_frac", "wind_hold_s")}
    out.update(wrow)
    out.update(row)
    out.update(srow)
    out["wall_time_s"] = time.perf_counter() - t_wall
    variant_rows = []
    for v in cfg.sensor.variants:  # O2: dieselbe Flugbahn, anderer Sensor (eigene Zufallsströme)
        sv = IdealSensor(float(v.sigma_pos_m), sensor.f_cam, int(v.q_frames), sensor.p_miss)
        obs_v = sv.observe(_trim(traj_full, q_max - sv.q), sens(f"ideal_noise_{v.label}"))
        st = IdealStatistics(
            cfg.observables.mode_removal, cfg.observables.dr_m,
            max_n_all_pairs=cfg.observables.max_n_all_pairs, n_pair_samples=cfg.observables.n_pair_samples,
            min_pairs_per_bin=cfg.observables.min_pairs_per_bin, rng=sens(f"pair_sampling_{v.label}"),
            nn_table=nn_table)
        st.add_observation(obs_v, cfg.observables.frame_stride)
        srow_v, _ = st.finalize(False)
        r = dict(out)
        r.update(srow_v)
        r.update({"control": f"{control}|S:{v.label}", "sigma_pos_m": sv.sigma_pos, "q_frames": sv.q})
        variant_rows.append(r)
    if variant_rows:
        out["_sensor_variants"] = variant_rows
    return out, curves


def _expand(r: dict[str, object]) -> list[dict[str, object]]:
    """Zeile eines Laufs plus die Zeilen seiner Sensorvarianten."""
    return [r] + list(r.pop("_sensor_variants", []))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Ein Designpunkt
# ---------------------------------------------------------------------------


def run_design_point(theta: Covariates, ps: PointSeeds, cfg: Config, prov: Provenance, *,
                     drone_classes: list[str] | None = None, controls: list[str] | None = None,
                     nn_table: NNBiasTable | None = None, keep_curves: bool = False
                     ) -> list[dict[str, object]]:
    """Alle Klassen eines Designpunkts (unabhängige Realisierungen) und die Kontrollen (CRN)."""
    drone_classes = list(cfg.experiment.drone_classes if drone_classes is None else drone_classes)
    controls = list(cfg.experiment.controls if controls is None else controls)
    rows: list[dict[str, object]] = []
    bird_row: dict[str, object] = {}
    if cfg.experiment.skip_birds:  # Vögel kommen aus dem Basislauf (gleiche Seeds, CRN)
        if {"K-Dichte", "K-Wind"} & set(controls):
            raise ValueError("K-Dichte und K-Wind brauchen den Vogellauf (skip_birds = false)")
    else:
        bird_row, _ = run_class(theta.bird_model, theta, ps, cfg, prov, nn_table=nn_table,
                                keep_curves=keep_curves)
        rows.extend(_expand(bird_row))
    for ctrl in controls:
        if ctrl == "K-Calm" and not cfg.experiment.skip_birds:  # ohne Wind (CRN), auch für den Vogel
            r, _ = run_class(theta.bird_model, theta, ps, cfg, prov, control=ctrl, wind_off=True,
                             nn_table=nn_table, keep_curves=keep_curves)
            rows.extend(_expand(r))
        for cls in drone_classes:
            if ctrl == "K0":
                r, _ = run_class(cls, theta, ps, cfg, prov, nn_table=nn_table, keep_curves=keep_curves)
            elif ctrl == "K-Dichte":
                cs = ps.for_class(cls)
                factor = float(sample(cfg.drones.dense_factor, cs.rng("par", "dense_factor")))
                spacing = float(bird_row["nn_dist"]) * factor  # type: ignore[arg-type]
                r, _ = run_class(cls, theta, ps, cfg, prov, control=ctrl, spacing_override=spacing,
                                 nn_table=nn_table, keep_curves=keep_curves)
            elif ctrl == "K-Wind":
                reaction = (float(bird_row["b_c_comp"]), float(bird_row["b_tau_est"]))  # type: ignore[arg-type]
                r, _ = run_class(cls, theta, ps, cfg, prov, control=ctrl, wind_reaction=reaction,
                                 nn_table=nn_table, keep_curves=keep_curves)
            elif ctrl == "K-Calm":
                r, _ = run_class(cls, theta, ps, cfg, prov, control=ctrl, wind_off=True,
                                 nn_table=nn_table, keep_curves=keep_curves)
            else:
                raise NotImplementedError(f"Kontrolle {ctrl!r} gehört zu P2 (Radar)")
            rows.extend(_expand(r))
    return rows
