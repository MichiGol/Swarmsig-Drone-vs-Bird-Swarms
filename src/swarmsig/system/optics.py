"""Optik-Systemmodell (O21, Stufe 2): Reichweiten von Kamera und Wärmebild sowie Stereo-Genauigkeit.

    Pixelreichweite (Johnson):   R_pix = s·f / (n·p)            s: kritische Abmessung, n: nötige Pixel, p: Pixelgröße
    Sichtreichweite (Koschmieder): R_vis = V/3,912 · ln(C₀/ε)   V: meteorologische Sichtweite, C₀: Eigenkontrast, ε: Schwelle
    Stereo (Tiefenfehler δZ ≈ Z²·δd/(f_px·B)):  Z_max = √(σ_max·f_px·B/δd)

Alle Funktionen arbeiten elementweise auf NumPy-Feldern. Quellen und Verteilungen: ``reports/O21_Stufe0_Parameter.md``
und ``reports/O21_Stufe2_prereg.md``.
"""

from __future__ import annotations

import numpy as np

KOSCHMIEDER = 3.912  # ln(1/0,02): Definition der meteorologischen Sichtweite


def pixel_range(size_m: np.ndarray, n_pixels: np.ndarray, pitch_m: float | np.ndarray,
                focal_m: np.ndarray) -> np.ndarray:
    """Entfernung, bis zu der das Ziel noch n Pixel über seine kritische Abmessung belegt."""
    return np.asarray(size_m) * np.asarray(focal_m) / (np.asarray(n_pixels) * np.asarray(pitch_m))


def visibility_range(visibility_m: np.ndarray | float, c0: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Entfernung, bei der der Kontrast C₀·exp(−3,912·R/V) auf die Schwelle ε fällt (0, falls C₀ ≤ ε)."""
    ratio = np.maximum(np.asarray(c0) / np.asarray(eps), 1.0)
    return np.asarray(visibility_m) / KOSCHMIEDER * np.log(ratio)


def stereo_range(sigma_max_m: float | np.ndarray, focal_px: np.ndarray, baseline_m: np.ndarray,
                 disparity_err_px: np.ndarray) -> np.ndarray:
    """Größte Entfernung, bei der der Tiefenfehler einer Stereokamera σ_max nicht übersteigt."""
    return np.sqrt(np.asarray(sigma_max_m) * np.asarray(focal_px) * np.asarray(baseline_m)
                   / np.asarray(disparity_err_px))
