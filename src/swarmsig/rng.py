"""Zufallszahlen-Hierarchie (Abschnitt 8.5, Regel 4).

Jeder Zufall läuft über ``numpy.random.Generator``-Objekte, die aus einer
``SeedSequence``-Hierarchie abgeleitet werden. Es gibt keinen globalen Zufallszustand.

Hierarchie (Spawn-Schlüssel relativ zur Experiment-Wurzel):

    Wurzel(seed)
    ├── (0, m)                Designpunkt m
    │   ├── (0, m, 0)         Kovariaten θ_m
    │   └── (0, m, 1 + k)     Klasse k (feste Registry, siehe CLASS_REGISTRY)
    │       ├── … , 0         env     (Windfeld)
    │       ├── … , 1         par     (klassenspezifische Modellparameter)
    │       ├── … , 2         init    (Anfangsbedingungen)
    │       ├── … , 3         agent   (Rauschpfade, Kommunikationsrauschen)
    │       └── … , 4         sensor  (Sensorrauschen)
    └── (1, k)                experimentweite Ströme (z. B. Stratifizierung)

Abweichungen von der Skizze in 8.5 (gleiche Semantik, robustere Umsetzung):

* Kinder werden **zustandslos** über explizite Spawn-Schlüssel erzeugt statt über
  ``SeedSequence.spawn`` (das einen Zähler im Elternobjekt hochzählt). So liefert jeder
  Aufruf dieselben Ströme, egal wie oft und in welcher Reihenfolge abgefragt wird.
* Der Klassenindex kommt aus einer festen Registry. Läuft nur eine Teilmenge der Klassen
  oder kommt später eine Klasse hinzu, ändern sich die Ströme der übrigen Klassen nicht.
* Innerhalb eines Stroms werden benannte Unterströme über einen stabilen Hash des Namens
  abgeleitet (:func:`substream`). Dadurch verschiebt das Hinzufügen eines Parameters oder
  einer Zufallsquelle die anderen Ziehungen nicht, was für CRN-Sweeps wichtig ist.

Kontrollvarianten (8.6) und Sweep-Werte verwenden dieselben :class:`ClassSeeds`
(gemeinsame Zufallszahlen, CRN); unabhängige Klassen eines Designpunkts verwenden
verschiedene.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

#: Feste Reihenfolge aller Klassen; der Index bestimmt den Spawn-Schlüssel.
#: Neue Klassen nur **anhängen**, nie umsortieren (sonst ändern sich alle Ströme).
CLASS_REGISTRY: tuple[str, ...] = ("B1", "B2", "B3", "D1", "D2", "D2c", "D3", "I", "N0", "DF")  # DF hinten (O5): bisherige Ströme unverändert

STREAMS: tuple[str, ...] = ("env", "par", "init", "agent", "sensor")

_KEY_POINTS = 0
_KEY_EXPERIMENT = 1


def child(ss: np.random.SeedSequence, *keys: int) -> np.random.SeedSequence:
    """Zustandsloses Kind von ``ss`` mit angehängtem Spawn-Schlüssel ``keys``."""
    return np.random.SeedSequence(
        entropy=ss.entropy,
        spawn_key=tuple(ss.spawn_key) + tuple(int(k) for k in keys),
        pool_size=ss.pool_size,
    )


def name_key(name: str) -> int:
    """Stabiler, plattformunabhängiger 32-bit-Schlüssel für einen Namen (CRC32)."""
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


def substream(ss: np.random.SeedSequence, name: str) -> np.random.SeedSequence:
    """Benannter Unterstrom von ``ss`` (unabhängig von der Ziehungsreihenfolge)."""
    return child(ss, name_key(name))


def generator(ss: np.random.SeedSequence, name: str | None = None) -> np.random.Generator:
    """``Generator`` (PCG64) für ``ss`` bzw. dessen benannten Unterstrom."""
    return np.random.default_rng(substream(ss, name) if name is not None else ss)


@dataclass(frozen=True)
class ClassSeeds:
    """Die fünf Ströme einer Klasse in einem Designpunkt (8.5)."""

    cls: str
    env: np.random.SeedSequence
    par: np.random.SeedSequence
    init: np.random.SeedSequence
    agent: np.random.SeedSequence
    sensor: np.random.SeedSequence

    def rng(self, stream: str, name: str | None = None) -> np.random.Generator:
        """Generator für Strom ``stream`` (env/par/init/agent/sensor), optional benannt."""
        if stream not in STREAMS:
            raise KeyError(f"unbekannter Strom {stream!r}; erlaubt: {STREAMS}")
        return generator(getattr(self, stream), name)


@dataclass(frozen=True)
class PointSeeds:
    """Seeds eines Designpunkts."""

    index: int
    point: np.random.SeedSequence

    @property
    def covariates(self) -> np.random.SeedSequence:
        return child(self.point, 0)

    def cov_rng(self, name: str) -> np.random.Generator:
        """Generator für eine einzelne Kovariate (benannter Unterstrom)."""
        return generator(self.covariates, name)

    def for_class(self, cls: str) -> ClassSeeds:
        try:
            k = CLASS_REGISTRY.index(cls)
        except ValueError as exc:
            raise KeyError(f"Klasse {cls!r} nicht in CLASS_REGISTRY {CLASS_REGISTRY}") from exc
        cs = child(self.point, 1 + k)
        return ClassSeeds(cls, *(child(cs, i) for i in range(len(STREAMS))))

    @property
    def entropy_tag(self) -> str:
        """Kompakte Herkunftsangabe für das Ausgabeschema (Spalte ``seed_entropy``)."""
        return f"{self.point.entropy}:{'/'.join(str(k) for k in self.point.spawn_key)}"


def experiment_root(seed: int) -> np.random.SeedSequence:
    return np.random.SeedSequence(int(seed))


def point_seeds(seed: int, index: int) -> PointSeeds:
    """Seeds für Designpunkt ``index`` des Experiments mit Wurzel-Seed ``seed``."""
    return PointSeeds(int(index), child(experiment_root(seed), _KEY_POINTS, int(index)))


def design_seeds(seed: int, n_points: int) -> list[PointSeeds]:
    return [point_seeds(seed, m) for m in range(int(n_points))]


def experiment_rng(seed: int, name: str) -> np.random.Generator:
    """Experimentweiter Strom (z. B. für die Stratifizierung diskreter Faktoren, 8.4)."""
    return generator(child(experiment_root(seed), _KEY_EXPERIMENT), name)
