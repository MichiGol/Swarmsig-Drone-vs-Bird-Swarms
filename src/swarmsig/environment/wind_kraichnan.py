"""Gitterfreie Turbulenzsynthese (Kraichnan, 5.3.4) und turbulentes Windfeld (5.2 + 5.3).

    w'(x, t) = Σ_m a_m cos(k_m·η) + b_m sin(k_m·η),   η = x − U_c ê_U t   (Taylor, eingefroren)
    a_m, b_m = A_m (I − k̂_m k̂_mᵀ) ζ_m,   ζ_m ~ N(0, I₃)   → divergenzfrei

Abweichung von 5.3.4 (gleiche Kovarianz, weniger Moden): Die Wellenzahlen werden nicht aus der
Dichte ∝ E(k) gezogen, sondern **geschichtet**: logarithmisch gleich breite Schalen zwischen
k_lo = κ_lo/L und k_hi = 2π/ℓ_min, je Schale ``dirs_per_shell`` Moden mit log-gleichverteiltem
Betrag in der Schale und gleichverteilter Richtung; A_m² = E_Schale / dirs_per_shell. So ist der
Trägheitsbereich (Skalen der Schwarmausdehnung) mit wenigen Moden gut besetzt (vgl.
„kinematic simulation“, Fung et al. 1992). ℓ_min ist ein Filter: Wirbel kleiner als ℓ_min
(≈ Körpergröße) entfallen; ihr Anteil an der Varianz wird protokolliert.

Anisotropie wie in 5.3.3 (5.): Komponentenweise Reskalierung im windfesten System auf
(σ_u, σ_v, σ_w), eingerechnet in a_m, b_m. Das verletzt die Divergenzfreiheit leicht
(dokumentierte Vereinfachung). Das isotrope Feld hat die Standardabweichung σ_u und die
longitudinale Integrallänge L_u (Spektralparameter L = L_u/0,7468).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .spectra import TurbulenceParams, von_karman_energy, vk_length_for_integral_scale
from .wind_profile import PowerLawWind


@dataclass(frozen=True)
class FourierModes:
    k: np.ndarray  # (M, 3) Wellenvektoren in rad/m
    a: np.ndarray  # (M, 3) Amplituden der Kosinusterme in m/s (welt-festes System)
    b: np.ndarray  # (M, 3) Amplituden der Sinusterme in m/s
    resolved_fraction: float  # Anteil der Varianz des isotropen Spektrums in [k_lo, k_hi]

    @property
    def n_modes(self) -> int:
        return int(self.k.shape[0])

    @staticmethod
    def empty() -> "FourierModes":
        z = np.zeros((0, 3))
        return FourierModes(z, z.copy(), z.copy(), 0.0)


def wind_frame(direction_rad: float) -> np.ndarray:
    """Spalten: ê_U (Windrichtung, *wohin* der Wind weht), ê_quer, ê_z."""
    e_u = np.array([math.sin(direction_rad), math.cos(direction_rad), 0.0])
    e_z = np.array([0.0, 0.0, 1.0])
    e_v = np.cross(e_z, e_u)
    return np.column_stack([e_u, e_v, e_z])


def draw_modes(tp: TurbulenceParams, direction_rad: float, rng: np.random.Generator, *,
               l_min: float = 1.0, k_lo_L: float = 0.05, shells_per_decade: int = 8,
               dirs_per_shell: int = 2) -> FourierModes:
    """Zieht die Moden einer Turbulenzrealisierung (Zufall nur aus ``rng``)."""
    if tp.sigma_u <= 0.0:
        return FourierModes.empty()
    L = vk_length_for_integral_scale(tp.L_u)
    k_lo, k_hi = k_lo_L / L, 2.0 * math.pi / l_min
    n_sh = max(1, int(math.ceil(shells_per_decade * math.log10(k_hi / k_lo))))
    edges = np.geomspace(k_lo, k_hi, n_sh + 1)
    e_shell = np.array([von_karman_energy(edges[i], edges[i + 1], tp.sigma_u, L) for i in range(n_sh)])
    D = int(dirs_per_shell)
    M = n_sh * D
    # Beträge log-gleichverteilt in der Schale, Richtungen gleichverteilt auf der Kugel
    lo = np.repeat(np.log(edges[:-1]), D)
    hi = np.repeat(np.log(edges[1:]), D)
    kmag = np.exp(lo + (hi - lo) * rng.random(M))
    kdir = rng.standard_normal((M, 3))
    kdir /= np.linalg.norm(kdir, axis=1, keepdims=True)
    amp = np.sqrt(np.repeat(e_shell, D) / D)
    zeta_a = rng.standard_normal((M, 3))
    zeta_b = rng.standard_normal((M, 3))
    proj = lambda z: z - np.sum(z * kdir, axis=1, keepdims=True) * kdir  # noqa: E731
    a = amp[:, None] * proj(zeta_a)
    b = amp[:, None] * proj(zeta_b)
    # Anisotropie: S = R diag(1, σ_v/σ_u, σ_w/σ_u) Rᵀ
    R = wind_frame(direction_rad)
    S = R @ np.diag([1.0, tp.sigma_v / tp.sigma_u, tp.sigma_w / tp.sigma_u]) @ R.T
    resolved = float(e_shell.sum() / (1.5 * tp.sigma_u**2))
    return FourierModes(kmag[:, None] * kdir, a @ S.T, b @ S.T, resolved)


def turbulence_numpy(modes: FourierModes, x: np.ndarray, t: float, U_c: float,
                     direction_rad: float) -> np.ndarray:
    """Reine NumPy-Auswertung von w'(x, t) (Referenz für Tests)."""
    if modes.n_modes == 0:
        return np.zeros_like(x, dtype=float)
    e = wind_frame(direction_rad)[:, 0]
    eta = x - U_c * t * e[None, :]
    ph = eta @ modes.k.T  # (N, M)
    return np.cos(ph) @ modes.a + np.sin(ph) @ modes.b


@dataclass(frozen=True)
class TurbulentWind:
    """Mittleres Potenzprofil + eingefrorene Turbulenz, mit U_c = U(z0) advektiert (5.3.3, 6.)."""

    mean: PowerLawWind
    modes: FourierModes
    U_c: float  # Advektionsgeschwindigkeit der eingefrorenen Turbulenz

    def params(self) -> np.ndarray:
        """Parameterfeld für den Numba-Kern ``kernels.wind_eval``."""
        e = wind_frame(self.mean.direction_rad)[:, 0]
        return np.array([self.mean.U10, self.mean.alpha, e[0], e[1], self.U_c])

    def sample(self, x: np.ndarray, t: float) -> np.ndarray:
        from ..agents.kernels import wind_eval  # gemeinsame Implementierung mit dem Schnellpfad

        out = np.empty_like(np.asarray(x, dtype=float))
        wind_eval(np.ascontiguousarray(x, dtype=float), float(t), self.params(),
                  self.modes.k, self.modes.a, self.modes.b, out)
        return out
