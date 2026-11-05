"""Wetterdaten für die Systemstudie (O21 Stufe 3) und P3: DWD-Archive lesen, Sonnenhöhe berechnen.

Die Archive des DWD Climate Data Center werden direkt aus den ZIP-Dateien gelesen, ohne sie zu entpacken, und nur als
Text/CSV behandelt.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

MISSING = -999  # Fehlwert-Kennung des DWD


def parse_station_list(path: Path) -> pd.DataFrame:
    """Stationsliste (``*_Beschreibung_Stationen.txt``) mit Höhe, Lage, Name und Bundesland, Index ``station_id``."""
    pat = re.compile(r"^(\d+)\s+(\d{8})\s+(\d{8})\s+(-?\d+)\s+([\d.]+)\s+([\d.]+)\s+(.+?)\s{2,}(.+?)\s{2,}")
    rows = []
    for line in Path(path).read_text(encoding="latin1").splitlines()[2:]:
        m = pat.match(line)
        if m:
            rows.append({"station_id": int(m[1]), "height_m": float(m[4]), "lat": float(m[5]),
                         "lon": float(m[6]), "station": m[7].strip(), "state": m[8].strip()})
    return pd.DataFrame(rows).set_index("station_id")


def read_dwd_product(zpath: str | Path) -> pd.DataFrame:
    """Produktdatei (``produkt_*.txt``) eines DWD-Archivs als Tabelle.

    ``MESS_DATUM`` wird zu ``time`` (UTC; bei 10-Minuten-Werten das Ende des Intervalls, bei Stundenwerten der
    Beobachtungstermin), ``STATIONS_ID`` zu ``station_id``. Fehlwerte (−999) werden NaN, Textspalten werden getrimmt.
    """
    with zipfile.ZipFile(zpath) as zf:
        member = [n for n in zf.namelist() if n.startswith("produkt_")][0]
        df = pd.read_csv(io.BytesIO(zf.read(member)), sep=";", skipinitialspace=True, dtype={"MESS_DATUM": str})
    df.columns = [c.strip() for c in df.columns]
    stamp = df["MESS_DATUM"].str.strip()
    df["time"] = pd.to_datetime(stamp, format="%Y%m%d%H%M" if stamp.str.len().iloc[0] == 12 else "%Y%m%d%H")
    df = df.drop(columns=["MESS_DATUM", "eor"], errors="ignore").rename(columns={"STATIONS_ID": "station_id"})
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].str.strip()
    return df.replace(MISSING, np.nan)


def nearest_log_index(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Index des im logarithmischen Abstand nächsten Gitterpunkts (Gitter aufsteigend, > 0; Werte außerhalb → Rand)."""
    lg = np.log(np.asarray(grid, dtype=float))
    return np.searchsorted(0.5 * (lg[1:] + lg[:-1]), np.log(np.asarray(values, dtype=float)))


def solar_elevation_deg(t_utc: pd.Series | pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """Sonnenhöhe in Grad ohne Refraktion.

    Niedrigpräzisionsformeln des Astronomical Almanac (Fehler ≈ 0,01° für 1950–2050). Die in P3 verwendete
    NOAA-Näherung ignoriert Schaltjahre und weicht um die Tagundnachtgleichen bis ≈ 0,5° ab.
    """
    n = np.asarray(pd.DatetimeIndex(t_utc).to_julian_date()) - 2451545.0  # Tage seit J2000.0
    mean_lon = np.radians((280.460 + 0.9856474 * n) % 360.0)
    anomaly = np.radians((357.528 + 0.9856003 * n) % 360.0)
    ecl_lon = mean_lon + np.radians(1.915) * np.sin(anomaly) + np.radians(0.020) * np.sin(2 * anomaly)
    obliq = np.radians(23.439 - 4e-7 * n)
    dec = np.arcsin(np.sin(obliq) * np.sin(ecl_lon))
    ra = np.arctan2(np.cos(obliq) * np.sin(ecl_lon), np.cos(ecl_lon))
    gmst_h = (18.697374558 + 24.06570982441908 * n) % 24.0
    ha = np.radians(gmst_h * 15.0 + lon) - ra
    phi = np.radians(lat)
    sin_e = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(ha)
    return np.degrees(np.arcsin(np.clip(sin_e, -1.0, 1.0)))
