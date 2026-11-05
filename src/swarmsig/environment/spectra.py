"""Turbulenz: Intensitäten und Längenskalen (MIL-F-8785C, 5.3.1) und von-Kármán-Spektrum (5.3.2).

MIL-F-8785C für niedrige Höhen (h < 1000 ft): **h in Fuß**, Längen in Fuß, dann in Meter.

    σ_w = 0,1·W20,   σ_u = σ_v = σ_w / (0,177 + 0,000823 h)^0,4,
    L_w = h,         L_u = L_v = h / (0,177 + 0,000823 h)^1,2

Isotropes Energiespektrum E(k) = C σ² L (kL)⁴ / (1 + (kL)²)^(17/6) mit ∫E dk = 3/2 σ²;
C = 3 / B(5/2, 1/3) ≈ 1,4528 (analytisch, unabhängig von L und σ). Die longitudinale
Integrallänge der isotropen Turbulenz ist L₁₁ = π/(2σ²) ∫E(k)/k dk = (π/4) C B(2, 5/6) L ≈ 0,7468 L.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import beta as _beta

FT_M = 0.3048  #: Fuß in Meter
H_MIN_FT = 10.0  #: MIL-F-8785C ist für h ≥ 10 ft definiert
H_MAX_FT = 1000.0  #: Gültigkeitsgrenze des Niedrighöhenmodells

#: Normierungskonstante des von-Kármán-Spektrums: ∫ x⁴/(1+x²)^(17/6) dx = B(5/2, 1/3)/2
VK_C = 1.5 / (0.5 * _beta(2.5, 1.0 / 3.0))
#: L₁₁ / L für das von-Kármán-Spektrum
VK_L11_OVER_L = (math.pi / 4.0) * VK_C * _beta(2.0, 5.0 / 6.0)


def von_karman_E(k: np.ndarray | float, sigma: float, L: float) -> np.ndarray:
    """E(k) in m³/s² (k in rad/m)."""
    x = np.asarray(k, dtype=float) * L
    return VK_C * sigma**2 * L * x**4 / (1.0 + x**2) ** (17.0 / 6.0)


def von_karman_energy(k_lo: float, k_hi: float, sigma: float, L: float) -> float:
    """∫_{k_lo}^{k_hi} E(k) dk (analytisch über die unvollständige Beta-Funktion)."""
    from scipy.special import betainc

    def cum(k: float) -> float:  # ∫_0^k E dk = Cσ² ∫_0^{kL} x⁴/(1+x²)^{17/6} dx
        if k <= 0:
            return 0.0
        if math.isinf(k):
            return 1.5 * sigma**2
        x2 = (k * L) ** 2
        # Substitution t = x²/(1+x²): ∫_0^X x⁴(1+x²)^{-17/6} dx = ½ B(5/2, 1/3) I_{t}(5/2, 1/3)
        return 1.5 * sigma**2 * float(betainc(2.5, 1.0 / 3.0, x2 / (1.0 + x2)))

    return cum(k_hi) - cum(k_lo)


def vk_length_for_integral_scale(L_11: float) -> float:
    """Spektralparameter L, dessen longitudinale Integrallänge L₁₁ ist."""
    return L_11 / VK_L11_OVER_L


@dataclass(frozen=True)
class TurbulenceParams:
    """Standardabweichungen (m/s) und Integrallängen (m) im windfesten System (u längs, v quer, w vertikal)."""

    sigma_u: float
    sigma_v: float
    sigma_w: float
    L_u: float
    L_v: float
    L_w: float


def _h_ft(h_m: float) -> float:
    return float(np.clip(h_m / FT_M, H_MIN_FT, H_MAX_FT))


def mil_low_altitude(h_m: float, W20: float) -> TurbulenceParams:
    """MIL-F-8785C (niedrige Höhen) aus der Flughöhe h (m) und dem Wind in 20 ft (m/s)."""
    h = _h_ft(h_m)
    q = 0.177 + 0.000823 * h
    sw = 0.1 * W20
    su = sw / q**0.4
    Lu = h / q**1.2 * FT_M
    return TurbulenceParams(su, su, sw, Lu, Lu, h * FT_M)


def mil_sigma_u_ratio(h_m: float, h_ref_m: float = 10.0) -> float:
    """σ_u(h)/σ_u(h_ref) nach MIL-F-8785C bei gleichem W20 (Höhenabhängigkeit der Intensität)."""
    q = lambda hm: 0.177 + 0.000823 * _h_ft(hm)  # noqa: E731
    return float((q(h_ref_m) / q(h_m)) ** 0.4)


def gust_scaled(h_m: float, U10: float, I_u10: float) -> TurbulenceParams:
    """MIL-F-8785C mit dem Niveau aus einer gemessenen Turbulenzintensität in 10 m.

    σ_u(h) = I_u10·U10·σ_u^MIL(h)/σ_u^MIL(10 m); Verhältnisse σ_w/σ_u und Längen wie MIL.
    """
    mil = mil_low_altitude(h_m, 1.0)
    su = I_u10 * U10 * mil_sigma_u_ratio(h_m, 10.0)
    return TurbulenceParams(su, su, su * mil.sigma_w / mil.sigma_u, mil.L_u, mil.L_v, mil.L_w)
