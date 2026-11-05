"""Speicherformat (10.5): Parquet-Tabelle ``runs`` mit Schema-Prüfung.

Pflichtspalten werden beim Schreiben geprüft; zusätzliche Spalten (Modellparameter mit
Präfix ``b_``/``d_``, Kenngrößen je Abzugsvariante) sind erlaubt. Kurven (C(r), χ(r))
werden optional als ``.npz`` je Lauf abgelegt; Zarr folgt mit P3 (Windgitter).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

#: Pflichtspalten nach Gruppen (Auswahl aus 10.5, soweit in P1 definiert).
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "identity": ("experiment_id", "design_point_id", "class", "control", "sweep_value",
                 "git_commit", "seed_entropy", "config_hash", "hypothetical"),
    "scenario": ("N", "N_band", "R_m", "z0_m", "azimuth_rad", "flight_mode", "heading_rad",
                 "cos_los_mean", "los_class", "T_obs_s"),
    "wind": ("U10", "alpha", "wind_dir", "c_comp", "tau_est"),
    "sensor": ("sensor_type", "detected"),
    "observables": ("L_g", "xi_s_A7", "xi_perp_A7", "chi_s_max_A7", "T_s_A7", "T_nn_A7",
                    "T_perp_A7", "T_chi_A7", "c_nn_A7"),
    "context": ("Phi", "mean_speed", "nn_dist", "density", "Omega_abs", "dilatation_D"),
    "diagnostics": ("stationary_ok", "validity_flags", "dt_s", "latency_rounding_s", "wall_time_s"),
}


def required_columns() -> list[str]:
    return [c for group in REQUIRED_COLUMNS.values() for c in group]


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in required_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"Ausgabeschema verletzt, fehlende Spalten: {missing}")
    dup = df.duplicated(subset=["experiment_id", "design_point_id", "class", "control", "sweep_value"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} doppelte Laufkennungen")


def rows_to_frame(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    # Leere Diagnosespalten einheitlich als Text speichern (Parquet-Typstabilität).
    if "validity_flags" in df:
        df["validity_flags"] = df["validity_flags"].fillna("").astype(str)
    return df


def write_runs(rows: Iterable[dict[str, object]], path: str | Path) -> pd.DataFrame:
    df = rows_to_frame(rows)
    validate(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def read_runs(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir():
        parts = sorted(p.glob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"keine Parquet-Dateien in {p}")
        return pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)
    return pd.read_parquet(p)


def write_curves(curves: dict[str, np.ndarray], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **curves)
