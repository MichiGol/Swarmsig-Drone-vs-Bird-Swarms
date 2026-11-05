"""Radar-Systemmodell (O21, Stufe 1): Radargleichung, Regendämpfung, Regenecho und Erkennungsreichweite.

Alle Funktionen sind über NumPy-Felder vektorisiert, damit viele Monte-Carlo-Ziehungen auf einmal laufen.

    SNR  = P·T_int·G²·λ²·σ / ((4π)³·R⁴·k·T₀·F·L) · 10^(−2·γ·R/10)        (Energieform, kohärente Integration)
    SCR  = σ·I_c / (η·V),  V = π/(8 ln 2)·R²·θ²·ΔR,  η = π⁵·|K|²·Z·10⁻¹⁸/λ⁴,  Z = 200·R_r^1,6
    SCNR = 1 / (1/SNR + 1/SCR)

γ ist die spezifische Regendämpfung nach ITU-R P.838-3 (Bibliothek ITU-Rpy). Ziel und Regenecho in derselben
Entfernung erfahren dieselbe Dämpfung, deshalb hängt das SCR nicht von ihr ab. Quellen und Verteilungen der Parameter:
``reports/O21_Stufe0_Parameter.md``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

C0 = 299_792_458.0
KT0 = 1.380649e-23 * 290.0  # W/Hz
K2_WATER = 0.93
APERTURE_EFF = 0.6


def swerling1_snr(pd: float = 0.9, pfa: float = 1e-6) -> float:
    """Nötiges SNR (linear) für ein Swerling-1-Ziel bei einer Beobachtung: P_d = P_fa^(1/(1+SNR))."""
    return float(np.log(pfa) / np.log(pd) - 1.0)


@lru_cache(maxsize=256)
def rain_attenuation_db_per_km(rain_mm_h: float, f_ghz: float) -> float:
    """Spezifische Regendämpfung (dB/km) nach ITU-R P.838-3, horizontale Polarisation, horizontaler Pfad."""
    if rain_mm_h <= 0.0:
        return 0.0
    from itur.models import itu838

    return float(itu838.rain_specific_attenuation(float(rain_mm_h), float(f_ghz), 0.0, 0.0).value)


def rain_eta(rain_mm_h: float, wavelength_m: float) -> float:
    """Volumen-Rückstreuung des Regens η (m²/m³) aus Marshall-Palmer Z = 200·R^1,6 (Rayleigh-Näherung)."""
    if rain_mm_h <= 0.0:
        return 0.0
    z = 200.0 * rain_mm_h**1.6 * 1e-18  # mm⁶/m³ -> m⁶/m³
    return float(np.pi**5 * K2_WATER * z / wavelength_m**4)


def wavelength(f_ghz: float) -> float:
    return C0 / (f_ghz * 1e9)


def gain(aperture_m2: np.ndarray, lam: float) -> np.ndarray:
    return 4.0 * np.pi * aperture_m2 / lam**2


def beamwidth(aperture_m2: np.ndarray, lam: float) -> np.ndarray:
    """3-dB-Keulenbreite (rad) einer Kreisapertur mit Flächenwirkungsgrad 0,6: θ ≈ 1,2·λ/D."""
    d = np.sqrt(4.0 * aperture_m2 / (np.pi * APERTURE_EFF))
    return 1.2 * lam / d


def scnr(r_m: np.ndarray, *, f_ghz: float, aperture_m2: np.ndarray, power_w: np.ndarray, t_int_s: np.ndarray,
         nf_db: np.ndarray, loss_db: np.ndarray, bandwidth_hz: np.ndarray, clutter_supp_db: np.ndarray,
         rcs_m2: np.ndarray, rain_mm_h: float) -> np.ndarray:
    """Signal-zu-Rausch-plus-Clutter-Verhältnis (linear) in Entfernung r_m; alle Felder gleicher Form."""
    lam = wavelength(f_ghz)
    g = gain(aperture_m2, lam)
    att = 10.0 ** (-2.0 * rain_attenuation_db_per_km(rain_mm_h, f_ghz) * (r_m / 1000.0) / 10.0)
    snr = (power_w * t_int_s * g**2 * lam**2 * rcs_m2 * att
           / ((4.0 * np.pi) ** 3 * r_m**4 * KT0 * 10.0 ** (nf_db / 10.0) * 10.0 ** (loss_db / 10.0)))
    eta = rain_eta(rain_mm_h, lam)
    if eta <= 0.0:
        return snr
    theta = beamwidth(aperture_m2, lam)
    vol = np.pi / (8.0 * np.log(2.0)) * r_m**2 * theta**2 * (C0 / (2.0 * bandwidth_hz))
    scr = rcs_m2 * 10.0 ** (clutter_supp_db / 10.0) / (eta * vol)
    return 1.0 / (1.0 / snr + 1.0 / scr)


def micro_doppler_range(blade_rel_db: np.ndarray, snr_md_db: np.ndarray, **kw: object) -> np.ndarray:
    """Reichweite der Rotorblatt-Signatur (O21 Stufe 2): SNR_Blatt = SNR_Rumpf + Δ_b ≥ SNR_mD.

    Die Blattechos liegen spektral weit außerhalb des Regen-Dopplers. Es zählen daher nur Rauschen und Dämpfung, das
    Regenecho wird ausgeschaltet. Umgesetzt über eine äquivalente Rückstreufläche an der Swerling-Schwelle:
    σ_äq = σ·10^(Δ_b/10)·SNR_Swerling/10^(SNR_mD/10).
    """
    req = swerling1_snr()
    kw = dict(kw)
    kw["rcs_m2"] = np.asarray(kw["rcs_m2"]) * 10.0 ** (np.asarray(blade_rel_db) / 10.0) * req / 10.0 ** (np.asarray(snr_md_db) / 10.0)
    kw["clutter_supp_db"] = np.asarray(kw["clutter_supp_db"]) + 300.0
    return detection_range(req, **kw)


def detection_range(snr_req: float, r_lo: float = 10.0, r_hi: float = 1e5, iters: int = 60, **kw: object) -> np.ndarray:
    """Größte Entfernung mit SCNR ≥ snr_req, per Bisektion in log R (SCNR fällt monoton mit R).

    Liegt die Schwelle schon bei r_lo nicht vor, wird 0 zurückgegeben; reicht sie über r_hi hinaus, r_hi.
    """
    shape = np.broadcast(*[np.asarray(v) for v in kw.values() if isinstance(v, np.ndarray)]).shape
    lo = np.full(shape, np.log(r_lo))
    hi = np.full(shape, np.log(r_hi))
    ok_lo = scnr(np.exp(lo), **kw) >= snr_req
    ok_hi = scnr(np.exp(hi), **kw) >= snr_req
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        good = scnr(np.exp(mid), **kw) >= snr_req
        lo = np.where(good, mid, lo)
        hi = np.where(good, hi, mid)
    r = np.exp(lo)
    r = np.where(ok_hi, r_hi, r)
    return np.where(ok_lo, r, 0.0)
