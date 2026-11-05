"""Szenario Flughafen (5.1): Geometrie, Flugmodi, Sichtlinienwinkel.

Koordinaten (2.1): Ursprung am Fußpunkt des Radarmasts, x Ost, y Nord, z oben; Radar-
Phasenzentrum bei (0, 0, h_r). Azimut von Nord im Uhrzeigersinn: ê = (sin φ, cos φ, 0).

Festlegungen (D, Design-Entscheidungen, in 5.1 nicht festgelegt):
* **Transit:** Der Schwarm startet um s0·T_burn entgegen dem Kurs versetzt, damit sein
  Schwerpunkt zu Beobachtungsbeginn nahe dem Designpunkt (R, φ, z0) liegt.
* **Kreisen:** Kreismittelpunkt ist der Designpunkt; Startwinkel zufällig (init-Strom);
  κ_h = (s0 / R_Kreis)², damit eine Kreisbahn mit Radius R_Kreis die Bewegungsgleichung
  des Schwerpunkts löst (Zentripetalbeschleunigung s0²/R = κ_h R).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..agents.drone import Reference
from ..agents.terms import Navigation


def unit_azimuth(az_rad: float) -> np.ndarray:
    return np.array([np.sin(az_rad), np.cos(az_rad), 0.0])


def design_centroid(R: float, azimuth: float, z0: float, h_r: float) -> np.ndarray:
    """Schwerpunkt mit Schrägentfernung R zum Radar-Phasenzentrum, Höhe z0, Azimut φ."""
    dz = z0 - h_r
    rho = np.sqrt(max(R**2 - dz**2, 0.0))
    return np.array([rho * np.sin(azimuth), rho * np.cos(azimuth), z0])


def cos_los(xbar: np.ndarray, ubar: np.ndarray, h_r: float) -> float:
    """cos θ_LOS = ê_0·ū/|ū| mit ê_0 = Sichtlinie vom Radar zum Schwerpunkt (2.1)."""
    e0 = xbar - np.array([0.0, 0.0, h_r])
    e0 = e0 / np.linalg.norm(e0)
    return float(e0 @ ubar / np.linalg.norm(ubar))


def los_class(abs_cos: float, edges: list[float]) -> str:
    """Winkelklasse nach |cos θ_LOS| (7.4): z. B. [0, 0.5), [0.5, 0.85), [0.85, 1]."""
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= abs_cos < hi or (hi == edges[-1] and abs_cos == hi):
            return f"[{lo},{hi})" if hi != edges[-1] else f"[{lo},{hi}]"
    return "nan"


@dataclass(frozen=True)
class FlightPlan:
    start_centroid: np.ndarray
    start_direction: np.ndarray  # Anfangsflugrichtung (horizontal)
    nav: Navigation
    reference: Reference  # für D2/D2-c (gleiche Bahn wie der Vogel-Schwerpunkt)


def plan_flight(*, mode: str, R: float, azimuth: float, z0: float, h_r: float, heading: float,
                circle_radius: float, s0: float, T_burn: float, kappa_n: float, kappa_z: float,
                start_angle: float) -> FlightPlan:
    c0 = design_centroid(R, azimuth, z0, h_r)
    if mode == "transit":
        e_T = unit_azimuth(heading)
        start = c0 - s0 * T_burn * e_T
        nav = Navigation("transit", z0=z0, kappa_z=kappa_z, e_T=e_T, kappa_n=kappa_n)
        ref = Reference("transit", x0=start.copy(), speed=s0, e_T=e_T)
        return FlightPlan(start, e_T, nav, ref)
    if mode == "loiter":
        radial = np.array([np.cos(start_angle), np.sin(start_angle), 0.0])
        tangent = np.array([-np.sin(start_angle), np.cos(start_angle), 0.0])  # gegen den Uhrzeigersinn
        start = c0 + circle_radius * radial
        nav = Navigation("loiter", z0=z0, kappa_z=kappa_z, x_c=c0.copy(),
                         kappa_h=(s0 / circle_radius) ** 2)
        ref = Reference("loiter", x0=c0.copy(), speed=s0, radius=circle_radius, phase0=start_angle)
        return FlightPlan(start, tangent, nav, ref)
    raise ValueError(f"unbekannter Flugmodus {mode!r}")
