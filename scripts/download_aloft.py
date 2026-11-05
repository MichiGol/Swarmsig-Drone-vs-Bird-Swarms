"""Lädt Vogelprofile (VPTS) aus dem Aloft-Datenbestand für O21 Stufe 4.

Quelle: Aloft (https://aloftdata.eu), öffentlicher S3-Bucket, Ordner ``baltrad/monthly/<radar>/<jahr>/``, eine
gzip-komprimierte CSV-Datei je Radar und Monat. Geladen werden die Wetterradare nahe den Flughäfen aus Stufe 3:
Boostedt (``deboo``, Hamburg), Hannover (``dehnr``) und Prötzel (``depro``, Berlin Brandenburg).
Standard: 2018 (Hauptanalyse, letztes vollständiges Jahr mit plausiblen Profilen) und 2023–2025 (vorregistrierte Variante).

Jede Datei wird gegen den ETag des Servers geprüft (MD5 bzw. Teil-MD5), sonst über Größe und gzip-CRC32 (siehe
``verify``). Vorhandene, geprüfte Dateien werden nicht erneut geladen. Danach: ``python scripts/prepare_aloft.py``.

    python scripts/download_aloft.py [--raw data/raw/aloft] [--dry-run]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import math
import re
import sys
import urllib.request
import zlib
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BUCKET = "https://aloftdata.s3-eu-west-1.amazonaws.com"
RADARS = ("deboo", "dehnr", "depro")
YEARS = (2018, 2023, 2024, 2025)


def listing(prefix: str) -> list[tuple[str, str, int]]:
    """(Schlüssel, ETag, Größe) aller Objekte unter ``prefix``; die S3-Liste wird Eintrag für Eintrag per Muster gelesen."""
    with urllib.request.urlopen(f"{BUCKET}/?list-type=2&prefix={prefix}", timeout=60) as r:  # noqa: S310 (feste URL)
        xml = r.read().decode("utf-8")
    out = []
    for block in re.findall(r"<Contents>(.*?)</Contents>", xml, re.S):
        key = re.search(r"<Key>([^<]+)</Key>", block)[1]
        etag = re.search(r"<ETag>&quot;([0-9a-f]{32}(?:-\d+)?)&quot;</ETag>", block)[1]
        out.append((key, etag, int(re.search(r"<Size>(\d+)</Size>", block)[1])))
    return out


def verify(path: Path, etag: str, size: int) -> str | None:
    """Prüft eine geladene Datei; gibt die verwendete Prüfung zurück oder None, wenn sie scheitert.

    1. Einfacher S3-ETag: MD5 der Datei.
    2. Teil-Upload (ETag ``md5-N``): MD5 der Teil-MD5s für übliche Teilgrößen. Manche Aloft-Dateien wurden mit
       unbekannter Teilgröße hochgeladen; dann gilt die Datei als intakt, wenn ihre Größe stimmt und das gzip-Archiv
       vollständig mit gültiger CRC32 entpackt.

    Nur Prüfsummen gegen den Server, keine Sicherheitsfunktion.
    """
    if path.stat().st_size != size:
        return None
    if "-" not in etag:
        h = hashlib.md5()  # noqa: S324
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return "MD5" if h.hexdigest() == etag else None
    n = int(etag.split("-")[1])
    for mib in (8, 16, 5, 10, 15, 32, 64):
        part = mib << 20
        if math.ceil(size / part) != n:
            continue
        with open(path, "rb") as f:
            digests = [hashlib.md5(chunk).digest() for chunk in iter(lambda: f.read(part), b"")]  # noqa: S324
        if f"{hashlib.md5(b''.join(digests)).hexdigest()}-{n}" == etag:  # noqa: S324
            return f"Teil-MD5 ({mib} MiB)"
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 20):
                pass
    except (OSError, EOFError, zlib.error):
        return None
    return "Größe und gzip-CRC32"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw/aloft"))
    ap.add_argument("--dry-run", action="store_true", help="nur anzeigen, was geladen würde")
    ap.add_argument("--years", type=int, nargs="+", default=list(YEARS))
    args = ap.parse_args()
    total = 0
    for radar in RADARS:
        for year in args.years:
            for key, etag, size in listing(f"baltrad/monthly/{radar}/{year}/"):
                if not key.endswith(".csv.gz"):
                    continue
                dest = args.raw / "baltrad" / radar / Path(key).name
                total += size
                if dest.exists() and (how := verify(dest, etag, size)):
                    print(f"vorhanden  {dest.name} ({how})")
                    continue
                if args.dry_run:
                    print(f"würde laden {key} ({size / 1e6:.1f} MB)")
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                with urllib.request.urlopen(f"{BUCKET}/{key}", timeout=300) as r, open(tmp, "wb") as f:  # noqa: S310
                    f.write(r.read())
                how = verify(tmp, etag, size)
                if not how:
                    tmp.unlink()
                    raise RuntimeError(f"Prüfung fehlgeschlagen: {key}")
                tmp.replace(dest)
                print(f"geladen    {dest.name} ({size / 1e6:.1f} MB, {how})")
    print(f"Gesamtgröße: {total / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
