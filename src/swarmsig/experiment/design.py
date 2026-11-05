"""Designpunkte (8.4, 8.5, 9.1).

Diskrete Größen (N-Band, Vogelmodell im Band, Flugmodus) werden **stratifiziert**
zugeteilt: Anteile nach Gewicht (Methode des größten Rests), Reihenfolge mit einem
experimentweiten Strom gemischt. Stetige Kovariaten werden je Designpunkt aus dessen
Kovariaten-Strom gezogen (jede Kovariate mit eigenem benannten Unterstrom).
Latin-Hypercube-Stichproben sind noch nicht umgesetzt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..config import Choice, Config, sample
from ..environment.weather import WeatherState, draw_weather
from ..rng import PointSeeds, design_seeds, experiment_rng


@dataclass(frozen=True)
class Covariates:
    index: int
    N: int
    N_band: str
    bird_model: str
    hypothetical: bool
    s0_mps: float
    R_m: float
    z0_m: float
    azimuth_rad: float
    flight_mode: str
    heading_rad: float
    circle_radius_m: float
    T_obs_s: float
    T_burn_s: float
    # Wetterzustand (P3); für alle Klassen des Designpunkts gleich
    weather: WeatherState = field(default_factory=WeatherState.none)

    def as_row(self) -> dict[str, object]:
        d = asdict(self)
        d["design_point_id"] = d.pop("index")
        d.update({f"weather_{k}": v for k, v in d.pop("weather").items()})
        return d


def allocate(labels: list[str], weights: list[float], m: int, rng: np.random.Generator) -> list[str]:
    """Stratifizierte Zuteilung von ``m`` Punkten auf ``labels`` (größter Rest, dann gemischt)."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    exact = w * m
    counts = np.floor(exact).astype(int)
    rest = m - counts.sum()
    order = np.argsort(-(exact - counts), kind="stable")
    counts[order[:rest]] += 1
    out = np.repeat(np.asarray(labels, dtype=object), counts)
    rng.shuffle(out)
    return [str(v) for v in out]


def band_of(cfg: Config, N: int) -> str:
    """N-Band eines Werts: Band aus der Config, sonst „XS“ (kleiner als Band S) bzw. „S“."""
    for b, spec in cfg.scenario.N_bands.items():
        if N in spec.values:
            return b
    smallest = min(min(spec.values) for spec in cfg.scenario.N_bands.values())
    return "XS" if N < smallest else "S"


def _covariates(cfg: Config, k: int, ps: PointSeeds, N: int, band: str, bird: str,
                mode: str) -> Covariates:
    sc = cfg.scenario
    c = lambda name, spec: sample(spec, ps.cov_rng(name))  # noqa: E731
    return Covariates(
        index=k, N=N, N_band=band, bird_model=bird, hypothetical=N > sc.hypothetical_N_above,
        s0_mps=float(c("s0", sc.s0_mps[bird])),
        R_m=float(c("R", sc.R_m)), z0_m=float(c("z0", sc.z0_m)),
        azimuth_rad=float(c("azimuth", sc.azimuth_rad)), flight_mode=mode,
        heading_rad=float(c("heading", sc.heading_rad)),
        circle_radius_m=float(c("circle_radius", sc.circle_radius_m)),
        T_obs_s=float(c("T_obs", sc.T_obs_s)), T_burn_s=float(c("T_burn", sc.T_burn_s)),
        weather=(draw_weather(cfg.environment, ps.cov_rng) if cfg.environment.wind != "none"
                 else WeatherState.none()),
    )


def _modes(cfg: Config, m: int, name: str) -> list[str]:
    fm = cfg.scenario.flight_mode
    if isinstance(fm, Choice):
        return allocate([str(v) for v in fm.values], fm.weights or [1.0] * len(fm.values), m,
                        experiment_rng(cfg.experiment.seed, name))
    return [str(fm)] * m


def draw_design_plan(cfg: Config) -> list[tuple[Covariates, PointSeeds]]:
    """Fester Plan je Gruppengröße (``scenario.N_plan``); Indizes aufsteigend nach N, damit
    ein Lauf mit Zeitlimit die kleinen Gruppen zuerst vollständig rechnet."""
    plan = cfg.scenario.N_plan or {}
    seed = cfg.experiment.seed
    total = sum(plan.values())
    seeds = design_seeds(seed, total)
    out: list[tuple[Covariates, PointSeeds]] = []
    k = 0
    for N in sorted(plan):
        count = int(plan[N])
        band = band_of(cfg, N)
        models = cfg.experiment.bird_classes[band]
        birds = allocate(models, [1.0] * len(models), count, experiment_rng(seed, f"bird_model_N{N}"))
        modes = _modes(cfg, count, f"flight_mode_N{N}")
        for q in range(count):
            out.append((_covariates(cfg, k, seeds[k], N, band, birds[q], modes[q]), seeds[k]))
            k += 1
    return out


def draw_design(cfg: Config, n_points: int | None = None) -> list[tuple[Covariates, PointSeeds]]:
    ex, sc = cfg.experiment, cfg.scenario
    if sc.N_plan:
        design = draw_design_plan(cfg)
        return design if n_points is None else design[: int(n_points)]
    m = ex.n_design_points if n_points is None else int(n_points)
    seeds = design_seeds(ex.seed, m)
    bands = list(sc.N_bands)
    band_alloc = allocate(bands, [sc.N_bands[b].weight for b in bands], m, experiment_rng(ex.seed, "N_band"))
    # Vogelmodell je Band stratifiziert (gleiche Anteile der erlaubten Modelle, 9.1).
    bird_of: list[str] = [""] * m
    for b in bands:
        idx = [k for k in range(m) if band_alloc[k] == b]
        models = ex.bird_classes[b]
        alloc = allocate(models, [1.0] * len(models), len(idx), experiment_rng(ex.seed, f"bird_model_{b}"))
        for k, mdl in zip(idx, alloc):
            bird_of[k] = mdl
    modes = _modes(cfg, m, "flight_mode")
    out = []
    for k, ps in enumerate(seeds):
        band = band_alloc[k]
        N = int(sample(Choice(dist="choice", values=list(sc.N_bands[band].values)), ps.cov_rng("N")))
        out.append((_covariates(cfg, k, ps, N, band, bird_of[k], modes[k]), ps))
    return out
