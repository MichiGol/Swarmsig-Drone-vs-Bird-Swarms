"""Config-System (Abschnitt 10.1/10.4): YAML-Dateien, validiert mit pydantic.

* Jede Zahl in den YAML-Dateien trägt einen Kommentar mit Herkunft
  (``# Quelle: …``, ``# ANNAHME``, ``# D``: Design-Entscheidung, ``# K``: Kalibrierung), Regel 2.
* Parameter können fest (Zahl/Text) oder als Verteilung angegeben werden
  (``{dist: uniform|loguniform|choice|weibull|const, …}``). Annahmen werden so in der
  Monte-Carlo-Schleife variiert statt fest verdrahtet.
* Eine Phasen-Config kann mit ``base: base.yaml`` eine Basis-Config erweitern (rekursives
  Überschreiben). Die effektive, validierte Config wird mit den Ergebnissen gespeichert
  (:func:`config_hash`).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Verteilungen
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Const(_Strict):
    dist: Literal["const"]
    value: float | int | str


class Uniform(_Strict):
    dist: Literal["uniform"]
    low: float
    high: float

    @model_validator(mode="after")
    def _check(self) -> "Uniform":
        if not self.low <= self.high:
            raise ValueError(f"uniform: low={self.low} > high={self.high}")
        return self


class LogUniform(_Strict):
    dist: Literal["loguniform"]
    low: float
    high: float

    @model_validator(mode="after")
    def _check(self) -> "LogUniform":
        if not 0 < self.low <= self.high:
            raise ValueError(f"loguniform: braucht 0 < low <= high, erhalten {self.low}, {self.high}")
        return self


class Choice(_Strict):
    dist: Literal["choice"]
    values: list[float | int | str]
    weights: list[float] | None = None

    @model_validator(mode="after")
    def _check(self) -> "Choice":
        if not self.values:
            raise ValueError("choice: leere Werteliste")
        if self.weights is not None:
            if len(self.weights) != len(self.values) or min(self.weights) < 0 or sum(self.weights) <= 0:
                raise ValueError("choice: Gewichte passen nicht zu den Werten")
        return self


class TruncWeibull(_Strict):
    """Weibull(Form k, Skala λ), abgeschnitten auf [low, high] (z. B. U10-Prior, 5.2)."""

    dist: Literal["weibull"]
    shape: float
    scale: float
    low: float = 0.0
    high: float = math.inf


DistSpec = Annotated[
    Union[Const, Uniform, LogUniform, Choice, TruncWeibull], Field(discriminator="dist")
]
#: Ein Parameter ist entweder ein fester Wert oder eine Verteilung.
Param = Union[int, float, str, bool, DistSpec]


def _py(v: Any) -> Any:
    """NumPy-Skalare in Python-Typen umwandeln (für saubere Ergebniszeilen)."""
    return v.item() if isinstance(v, np.generic) else v


def sample(spec: Param, rng: np.random.Generator) -> Any:
    """Zieht einen Wert aus ``spec`` (feste Werte werden unverändert zurückgegeben)."""
    if isinstance(spec, (int, float, str, bool)):
        return spec
    if isinstance(spec, Const):
        return spec.value
    if isinstance(spec, Uniform):
        return float(rng.uniform(spec.low, spec.high))
    if isinstance(spec, LogUniform):
        return float(np.exp(rng.uniform(np.log(spec.low), np.log(spec.high))))
    if isinstance(spec, Choice):
        p = None
        if spec.weights is not None:
            w = np.asarray(spec.weights, dtype=float)
            p = w / w.sum()
        return _py(spec.values[int(rng.choice(len(spec.values), p=p))])
    if isinstance(spec, TruncWeibull):
        k, lam = spec.shape, spec.scale
        cdf = lambda x: 1.0 - math.exp(-((x / lam) ** k)) if x < math.inf else 1.0  # noqa: E731
        q = rng.uniform(cdf(spec.low), cdf(spec.high))
        return float(lam * (-math.log1p(-q)) ** (1.0 / k))
    raise TypeError(f"unbekannte Parameterangabe {spec!r}")


def is_random(spec: Param) -> bool:
    return not isinstance(spec, (int, float, str, bool, Const))


# ---------------------------------------------------------------------------
# Config-Abschnitte
# ---------------------------------------------------------------------------


class ExperimentCfg(_Strict):
    id: str
    seed: int
    n_design_points: int = Field(gt=0)
    bird_classes: dict[str, list[str]]
    drone_classes: list[str]
    controls: list[str] = ["K0"]
    record_curves: bool = False
    # Sweeps mit gemeinsamen Zufallszahlen (8.5): nur Drohnen neu rechnen, Vögel aus dem Basislauf
    skip_birds: bool = False
    sweep_value: float | None = None  # wird in die Spalte sweep_value geschrieben
    # Nur Designpunkte mit diesen Gruppengrößen rechnen (Indizes und Seeds bleiben wie im vollen Plan)
    only_N: list[int] | None = None


class NBand(_Strict):
    values: list[int]
    weight: float = Field(ge=0)


class ScenarioCfg(_Strict):
    N_bands: dict[str, NBand]
    hypothetical_N_above: int
    R_m: Param
    z0_m: Param
    azimuth_rad: Param
    flight_mode: Param
    heading_rad: Param
    circle_radius_m: Param
    T_obs_s: Param
    T_burn_s: Param
    s0_mps: dict[str, Param]  # Soll-Speed je Vogelmodell; Drohnen übernehmen sie (9.3)
    # Fester Plan „Designpunkte je Gruppengröße“ (ersetzt N_bands/n_design_points, falls gesetzt)
    N_plan: dict[int, int] | None = None


class SiteCfg(_Strict):
    h_r_m: float


class StationarityCfg(_Strict):
    sample_interval_s: float
    window_fraction: float
    alpha: float


class DynamicsCfg(_Strict):
    dt_s: float
    dt_allowed_s: list[float]
    dt_safety: float
    auto_reduce_dt: bool = True
    u_q_marginal_mps: float
    stationarity: StationarityCfg
    # "numba": kompilierte Kerne (nur ohne Wind, N ≤ fast_max_n); "numpy": Referenzimplementierung
    backend: Literal["numba", "numpy"] = "numba"
    fast_max_n: int = 1500
    # Zeitschritte je Modellfamilie aus dem Pilotlauf (überschreiben dt_s/auto_reduce_dt, 4.9)
    dt_bird_s: float | None = None
    dt_drone_s: float | None = None


class InitCfg(_Strict):
    aspect: tuple[float, float, float]
    speed_noise_rel: float


class NavCfg(_Strict):
    kappa_n_per_s: Param
    kappa_z_per_s2: Param


class BirdCfg(_Strict):
    neighborhood: Param  # "topological" | "metric"
    n_c: Param
    r_c_m: Param = 0.0  # nur für metrische Nachbarschaft
    speed_control: Param  # "stiff" | "marginal"
    J_per_s: Param
    sigma: Param
    kappa_s_per_s: Param
    u_star_mps: Param
    beta_mps2: Param
    r0_m: Param
    r_w_m: Param
    r_hc_m: Param
    eta_h: Param = 0.0
    tau0_s: Param = 0.0
    # Zusammenhalt über die n_coh nächsten Nachbarn; 0 = dieselben Nachbarn wie die Ausrichtung (4.2)
    n_coh: Param = 0
    c_comp: Param
    tau_est_s: Param
    # Posterior der Kalibrierung (4.8): Parquet mit akzeptierten Parametersätzen; je Lauf wird
    # eine Zeile gezogen und überschreibt die gleichnamigen Parameter (Spaltennamen wie hier).
    calibration_posterior: str | None = None


class D1Cfg(_Strict):
    c_a_per_s: Param
    c_c_per_s2: Param
    c_s_factor: Param


class D2Cfg(_Strict):
    a: Param
    b: Param
    h: Param
    eps: Param
    c1_per_s2: Param
    c2_per_s: Param
    # "literal": r = r_c (wörtlich 4.5); "local": r = κ·d (Olfati-Saber); "knn": r = r_c, nur die k nächsten
    range_mode: Param = "literal"
    kappa_range: float = 1.2
    knn: int = 10


class D3Cfg(_Strict):
    p_rep_per_s: Param
    r0_fric_m: Param
    C_fric: Param
    v_fric_mps: Param
    p_fric_per_s: Param
    a_fric_mps2: Param
    attraction: bool = False


class DFCfg(_Strict):
    k_form_per_s: Param


class DroneCfg(_Strict):
    r_c_m: Param
    # Höchstens so viele nächste Nachbarn innerhalb r_c (SwarmLab max_neig); None = alle (wörtlich 4.5)
    max_neighbors: int | None = None
    spacing_m: Param
    dense_factor: Param
    kappa_s_per_s: Param
    sigma_d: Param
    c_comp: Param
    tau_est_s: Param
    D1: D1Cfg
    D2: D2Cfg
    D3: D3Cfg
    DF: DFCfg | None = None  # O5: Formation ohne Nachbarkopplung


class RealismCfg(_Strict):
    f_ctrl_hz: Param
    tau_d_s: Param
    sigma_g_m: Param
    sigma_gv_mps: Param
    tau_g_s: Param
    p_loss: Param
    tau_v_s: Param
    a_max_mps2: Param
    v_max_mps: Param
    max_packet_age_s: float
    # Anteil der GPS-Fehlervarianz, den alle Drohnen eines Schwarms teilen (atmosphärische und
    # Bahnfehler); 0 = unabhängige Fehler wie in 4.7
    gps_shared_frac: Param = 0.0
    # Regler nutzt für den eigenen Zustand die eigene GPS-Schätzung (wie gesendet); False = exakt (4.7)
    own_state_noise: bool = False


class EnvironmentCfg(_Strict):
    # "none": kein Wind (P1, P2); "mean": nur Potenzprofil; "turbulent": Profil + von-Kármán-Turbulenz (P3)
    wind: Literal["none", "mean", "turbulent"] = "none"
    # Reale Wetterzustände (scripts/prepare_dwd_wind.py); None = parametrische Prioren (5.2)
    weather_table: str | None = None
    U10_mps: Param = TruncWeibull(dist="weibull", shape=2.0, scale=4.5, low=0.0, high=15.0)
    wind_dir_rad: Param = Uniform(dist="uniform", low=0.0, high=2.0 * math.pi)
    alpha: Param = Uniform(dist="uniform", low=0.1, high=0.4)
    # Turbulenzintensität in 10 m, falls kein Böenfaktor vorliegt (Median der DWD-Herbstdaten)
    I_u10_default: float = 0.141
    # "gust": MIL-F-8785C mit Niveau aus dem Böenfaktor; "mil": MIL-F-8785C aus W20 (5.3.1)
    turbulence_level: Literal["gust", "mil"] = "gust"
    l_min_m: float = Field(default=1.0, gt=0)  # kleinste Wirbelgröße (Körpergrößen-Filter)
    k_lo_L: float = Field(default=0.05, gt=0)  # kleinste Wellenzahl in Einheiten 1/L
    shells_per_decade: int = Field(default=8, ge=1)
    dirs_per_shell: int = Field(default=2, ge=1)
    hold_s: float = Field(default=0.005, gt=0)  # Windauswertung höchstens alle hold_s (gehalten)


class SensorVariantCfg(_Strict):
    """Zusätzliche Auswertung derselben Flugbahn mit anderem Positionsrauschen bzw. Zeitfenster."""

    label: str
    sigma_pos_m: float = Field(ge=0)
    q_frames: int = Field(ge=1)


class SensorCfg(_Strict):
    type: Literal["ideal"]
    sigma_pos_m: Param
    f_cam_hz: Param
    q_frames: int = Field(ge=1)
    p_miss: Param
    # Sensorvarianten (O2): je Variante eine weitere Ergebniszeile mit control = "<Kontrolle>|S:<label>"
    variants: list[SensorVariantCfg] = []


class ObservablesCfg(_Strict):
    mode_removal: list[Literal["A7", "M", "A12", "L", "Q", "RL"]]
    primary: dict[str, str]
    dr_m: float = Field(gt=0)
    max_n_all_pairs: int
    n_pair_samples: int
    min_pairs_per_bin: int = Field(ge=1)
    frame_stride: int = Field(ge=1)
    nn_bias_table: str | None = None
    los_class_edges: list[float]


class NullModelCfg(_Strict):
    matern_nu: float
    c_s: float
    c_perp: float
    cholesky_max_n: int
    n_fourier_features: int


class Config(_Strict):
    experiment: ExperimentCfg
    scenario: ScenarioCfg
    site: SiteCfg
    dynamics: DynamicsCfg
    init: InitCfg
    nav: NavCfg
    birds: dict[str, BirdCfg]
    drones: DroneCfg
    realism: RealismCfg
    environment: EnvironmentCfg
    sensor: SensorCfg
    observables: ObservablesCfg
    nullmodels: NullModelCfg

    def with_overrides(self, overrides: dict[str, Any]) -> "Config":
        """Neue, validierte Config mit rekursiv überschriebenen Einträgen."""
        return Config.model_validate(deep_merge(self.model_dump(mode="python"), overrides))


# ---------------------------------------------------------------------------
# Laden, Mischen, Hash
# ---------------------------------------------------------------------------


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and "dist" not in v:
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_raw(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    base = raw.pop("base", None)
    if base is not None:
        raw = deep_merge(_load_raw((path.parent / base).resolve()), raw)
    return raw


def load_config(path: str | Path) -> Config:
    return Config.model_validate(_load_raw(Path(path)))


def config_hash(cfg: Config) -> str:
    """SHA-256 der kanonischen JSON-Form der effektiven Config (Spalte ``config_hash``)."""
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dump_config(cfg: Config, path: str | Path) -> None:
    """Effektive Config als YAML speichern (wird mit den Ergebnissen abgelegt)."""
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.model_dump(mode="json"), fh, sort_keys=False, allow_unicode=True)
