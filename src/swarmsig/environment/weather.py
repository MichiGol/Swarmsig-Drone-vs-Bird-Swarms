"""Wetterzustand eines Designpunkts und das daraus folgende Windfeld (P3; 5.2, 5.3, 5.4).

Zwei Quellen für (U10, Windrichtung, Turbulenzintensität):

* **DWD-Bootstrap** (Entscheidung Projektleitung 2026-10-11, leichte Form von 5.4 mit
  Stationsdaten statt ERA5): je Designpunkt ein realer 10-Minuten-Zustand aus
  ``scripts/prepare_dwd_wind.py`` (Flughäfen im Norddeutschen Tiefland, September–November,
  Tageslicht). Die Turbulenzintensität in 10 m stammt aus dem gemessenen Böenfaktor.
* **Prioren des Kernpfads** (5.2): U10 ~ Weibull(2; 4,5 m/s) auf [0; 15], Richtung gleichverteilt.

Der Profilexponent α kommt in beiden Fällen aus dem Prior (5.2), weil Stationsdaten nur eine
Höhe haben. Turbulenzniveau: ``gust`` = MIL-F-8785C mit dem Niveau aus I_u10 (``spectra.gust_scaled``),
``mil`` = MIL-F-8785C aus W20 = U(6,1 m) wie in 5.3.1.

Zuständigkeiten der Zufallsströme (8.5): Wetterzustand und α sind **Kovariaten** (für alle
Klassen eines Designpunkts gleich); die Turbulenzrealisierung (Moden) kommt aus dem
``env``-Strom der Klasse (unabhängig je Klasse, gemeinsam über Kontrollvarianten, CRN).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from ..config import EnvironmentCfg, sample
from .spectra import TurbulenceParams, gust_scaled, mil_low_altitude
from .wind_kraichnan import TurbulentWind, draw_modes
from .wind_profile import PowerLawWind

RngFor = Callable[[str], np.random.Generator]


@lru_cache(maxsize=4)
def load_weather_table(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["station", "time_utc", "U10", "dir_from_deg", "I_u10"])
    if df.empty:
        raise ValueError(f"Wettertabelle {path} ist leer")
    return df.reset_index(drop=True)


@dataclass(frozen=True)
class WeatherState:
    U10: float  # m/s in 10 m
    wind_dir_rad: float  # Richtung, *wohin* der Wind weht (Azimut von Nord im Uhrzeigersinn, 2.1)
    I_u10: float  # Turbulenzintensität σ_u/U in 10 m
    alpha: float  # Profilexponent
    source: str  # "dwd" | "prior"
    station: str = ""
    time_utc: str = ""

    def as_row(self) -> dict[str, object]:
        return asdict(self)

    @staticmethod
    def none() -> "WeatherState":
        """Kein Wind; immer dasselbe Objekt, damit Gleichheitsvergleiche trotz NaN gelingen."""
        return _NO_WEATHER


_NAN = float("nan")
_NO_WEATHER = WeatherState(_NAN, _NAN, _NAN, _NAN, "none")


def draw_weather(env: EnvironmentCfg, rng_for: RngFor) -> WeatherState:
    """Zieht den Wetterzustand eines Designpunkts (benannte Unterströme des Kovariaten-Stroms)."""
    alpha = float(sample(env.alpha, rng_for("alpha")))
    if env.weather_table:
        tab = load_weather_table(env.weather_table)
        row = tab.iloc[int(rng_for("weather_row").integers(len(tab)))]
        U10 = float(row["U10"])
        d_from = float(row["dir_from_deg"])
        if not math.isfinite(d_from):  # Windstille: Richtung undefiniert
            d_from = float(rng_for("wind_dir").uniform(0.0, 360.0))
        I_u = float(row["I_u10"]) if math.isfinite(float(row["I_u10"])) else env.I_u10_default
        return WeatherState(U10, math.radians((d_from + 180.0) % 360.0), I_u, alpha, "dwd",
                            str(row["station"]), str(row["time_utc"]))
    U10 = float(sample(env.U10_mps, rng_for("U10")))
    direction = float(sample(env.wind_dir_rad, rng_for("wind_dir")))
    return WeatherState(U10, direction, env.I_u10_default, alpha, "prior")


def mean_speed(ws: WeatherState, z: float) -> float:
    """U(z) = U10 (z/10 m)^α (5.2)."""
    return float(ws.U10 * (max(z, 0.0) / 10.0) ** ws.alpha)


def turbulence_params(env: EnvironmentCfg, ws: WeatherState, h: float) -> TurbulenceParams:
    if env.turbulence_level == "gust":
        return gust_scaled(h, ws.U10, ws.I_u10)
    return mil_low_altitude(h, mean_speed(ws, 6.1))


@dataclass(frozen=True)
class WindSetup:
    """Windfeld einer Klasse in einem Designpunkt plus Kennzahlen für die Ergebniszeile."""

    wind: TurbulentWind
    tp: TurbulenceParams
    U_z0: float
    hold_steps: int

    def row(self, dt: float) -> dict[str, object]:
        return {"U_z0": self.U_z0, "sigma_u_z0": self.tp.sigma_u, "sigma_w_z0": self.tp.sigma_w,
                "L_u_z0": self.tp.L_u, "wind_modes": self.wind.modes.n_modes,
                "turb_resolved_frac": self.wind.modes.resolved_fraction,
                "wind_hold_s": self.hold_steps * dt}


def build_wind(env: EnvironmentCfg, ws: WeatherState, z0: float, dt: float,
               rng: np.random.Generator) -> WindSetup:
    """Mittleres Profil + Turbulenzrealisierung (``env.wind == "turbulent"``) bzw. nur Profil (``"mean"``)."""
    tp = turbulence_params(env, ws, z0)
    U_z0 = mean_speed(ws, z0)
    if env.wind == "turbulent":
        modes = draw_modes(tp, ws.wind_dir_rad, rng, l_min=env.l_min_m, k_lo_L=env.k_lo_L,
                           shells_per_decade=env.shells_per_decade, dirs_per_shell=env.dirs_per_shell)
    else:
        from .wind_kraichnan import FourierModes

        modes = FourierModes.empty()
    wind = TurbulentWind(PowerLawWind(ws.U10, ws.alpha, ws.wind_dir_rad), modes, U_z0)
    hold = max(1, int(env.hold_s / dt + 1e-9))
    return WindSetup(wind, tp, U_z0, hold)
