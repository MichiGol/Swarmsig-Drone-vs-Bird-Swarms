"""Lädt die DWD-Rohdaten für P3 (10-Minuten-Wind und -Böen) von opendata.dwd.de.

Holt für die Flughafenstationen Bremen (00691), Hannover (02014), Hamburg-Fuhlsbüttel (01975)
und Berlin Brandenburg (00427):

* Stationslisten und Verzeichnisse (``meta/``),
* Metadaten-Archive der Windmessung (Geberhöhe, Gerätetyp; ``meta/``),
* Produktarchive 2010–2019 und 2020–2025 für Wind (``zip_ff/``) und Böen (``zip_fx/``).

Vorhandene Dateien werden nicht erneut geladen. Danach: ``python scripts/prepare_dwd_wind.py``.

    python scripts/download_dwd_wind.py [--raw data/raw/dwd] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes"
STATIONS = ("00427", "00691", "01975", "02014")
PERIODS = ("20100101_20191231", "20200101_20251231")


def targets() -> list[tuple[str, str]]:
    """(URL, relativer Zielpfad) aller benötigten Dateien."""
    out = [(f"{BASE}/wind/historical/zehn_min_ff_Beschreibung_Stationen.txt", "meta/ff_stations.txt"),
           (f"{BASE}/extreme_wind/historical/zehn_min_fx_Beschreibung_Stationen.txt", "meta/fx_stations.txt")]
    for st in STATIONS:
        out.append((f"{BASE}/wind/meta_data/Meta_Daten_zehn_min_ff_{st}.zip", f"meta/Meta_Daten_zehn_min_ff_{st}.zip"))
        for per in PERIODS:
            ff = f"10minutenwerte_wind_{st}_{per}_hist.zip"
            fx = f"10minutenwerte_extrema_wind_{st}_{per}_hist.zip"
            out.append((f"{BASE}/wind/historical/{ff}", f"zip_ff/{ff}"))
            out.append((f"{BASE}/extreme_wind/historical/{fx}", f"zip_fx/{fx}"))
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
