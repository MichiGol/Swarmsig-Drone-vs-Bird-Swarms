"""Lädt die DWD-Rohdaten für O21 Stufe 3 (10-Minuten-Niederschlag und stündliche Sichtweite) von opendata.dwd.de.

Holt für die Flughafenstationen Berlin Brandenburg (00427), Bremen (00691), Hamburg-Fuhlsbüttel (01975) und
Hannover (02014):

* Stationslisten und die Parameterbeschreibung des Niederschlags (``meta/``),
* 10-Minuten-Niederschlag 2010–2019 und 2020–2025 (``zip_rr/``),
* stündliche Sichtweite, gesamter Zeitraum je Station (``zip_vv/``),
* 10-Minuten-Lufttemperatur 2010–2019 und 2020–2025 (``zip_tu/``), um möglicherweise gefrorenen Niederschlag zu kennzeichnen.

2010–2019 wird für O21 Stufe 4 gebraucht (Vogelprofile 2018).

Der Wind (``zip_ff/``) kommt aus ``download_dwd_wind.py``. Vorhandene Dateien werden nicht erneut geladen.
Danach: ``python scripts/prepare_dwd_o21.py``.

    python scripts/download_dwd_o21.py [--raw data/raw/dwd] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate"
RR = f"{BASE}/10_minutes/precipitation/historical"
VV = f"{BASE}/hourly/visibility/historical"
TU = f"{BASE}/10_minutes/air_temperature/historical"
STATIONS = ("00427", "00691", "01975", "02014")
VV_START = {"00427": "19750701", "00691": "19490101", "01975": "19490101", "02014": "19490101"}
PERIODS = ("20100101_20191231", "20200101_20251231")


def targets() -> list[tuple[str, str]]:
    """(URL, relativer Zielpfad) aller benötigten Dateien."""
    out = [(f"{RR}/zehn_min_rr_Beschreibung_Stationen.txt", "meta/rr_stations.txt"),
           (f"{VV}/VV_Stundenwerte_Beschreibung_Stationen.txt", "meta/vv_stations.txt")]
    for st in STATIONS:
        vv = f"stundenwerte_VV_{st}_{VV_START[st]}_20251231_hist.zip"
        out.append((f"{VV}/{vv}", f"zip_vv/{vv}"))
        for per in PERIODS:
            rr = f"10minutenwerte_nieder_{st}_{per}_hist.zip"
            tu = f"10minutenwerte_TU_{st}_{per}_hist.zip"
            out.append((f"{RR}/{rr}", f"zip_rr/{rr}"))
            out.append((f"{TU}/{tu}", f"zip_tu/{tu}"))
    # Parameterbeschreibung (u. a. Bedeutung der Niederschlagskennung RWS_IND_10)
    out.append((f"{BASE}/10_minutes/precipitation/meta_data/Meta_Daten_zehn_min_rr_00691.zip",
                "meta/Meta_Daten_zehn_min_rr_00691.zip"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw/dwd"))
    ap.add_argument("--dry-run", action="store_true", help="nur anzeigen, was geladen würde")
    args = ap.parse_args()
    for url, rel in targets():
        dest = args.raw / rel
        if dest.exists() and dest.stat().st_size > 0:
            print(f"vorhanden  {rel}")
            continue
        if args.dry_run:
            print(f"würde laden {url}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:  # noqa: S310 (feste DWD-URL)
            f.write(r.read())
        tmp.replace(dest)
        print(f"geladen    {rel} ({dest.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
