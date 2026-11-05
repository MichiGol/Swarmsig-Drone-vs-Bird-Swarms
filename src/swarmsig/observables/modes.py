"""Pflichtabzug kollektiver Moden (Regel 11, Abschnitt 7.1).

A7 (primär): u_i ≈ U + Ω × (x_i − x̄) + D (x_i − x̄), kleinste Quadrate über alle Individuen.
Weil Σ_i (x_i − x̄) = 0 und r × r = 0, zerfallen die Normalgleichungen exakt:

    U = ū,   D = Σ r_i·δu⁰_i / Σ |r_i|²,   Ω = I⁻¹ Σ r_i × δu⁰_i,   I = Σ (|r_i|² 𝟙 − r_i r_iᵀ)

(r_i = x_i − x̄, δu⁰_i = u_i − ū). Danach δs_i = n̂·δu_i (Längskomponente, n̂ = U/|U|) und
δu⊥_i = δu_i − n̂ δs_i.

M  (sekundär):    nur Mittelwert; δs⁰_i = |u_i| − s̄ (klassische Definition).
A12 (Sensitivität): volles affines Feld U + A (x_i − x̄).

Speed-basierte Kandidaten (O3b, robust gegen nicht starren Kurvenflug, bei dem A7 eine scheinbare
Kopplung erzeugt, siehe ``reports/O3a_Echte_Voegel_Bericht.md``, Abschnitt 4):

L  : δs_i = |u_i| − (a + b·r_i)            Speed minus lineares Gefälle (kleinste Quadrate)
Q  : δs_i = |u_i| − Polynom 2. Grades in r_i
RL : δs_i = |u_i| − |U + Ω × r_i + D r_i|, danach lineares Gefälle abgezogen (Speed relativ zur
     starren A7-Bewegung)

Für sie ist δu⊥ wie bei M definiert (Abzug nur des Mittelwerts).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VARIANTS = ("A7", "M", "A12", "L", "Q", "RL")


@dataclass(frozen=True)
class Residuals:
    variant: str
    x: np.ndarray  # (n, 3) Positionen (für Paarabstände)
    du: np.ndarray  # (n, 3) Residuen δu_i
    ds: np.ndarray  # (n,)  Speed-Schwankung
    du_perp: np.ndarray  # (n, 3) Richtungsschwankung
    n_hat: np.ndarray  # (3,)
    U: np.ndarray  # (3,)
    Omega: np.ndarray  # (3,), NaN wenn nicht angepasst
    D: float  # NaN wenn nicht angepasst


def _split(du: np.ndarray, n_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ds = du @ n_hat
    return ds, du - ds[:, None] * n_hat[None, :]


def _unit(U: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(U))
    if nrm == 0:
        raise ValueError("mittlere Geschwindigkeit ist 0: Flugrichtung n̂ undefiniert")
    return U / nrm


def fit_rigid(x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Kleinste-Quadrate-Fit von Translation U, Rotation Ω und Dilatation D."""
    r = x - x.mean(axis=0)
    U = u.mean(axis=0)
    du0 = u - U
    rr = np.einsum("ij,ij->i", r, r)
    s2 = float(rr.sum())
    D = float(np.einsum("ij,ij->", r, du0) / s2) if s2 > 0 else 0.0
    inertia = s2 * np.eye(3) - r.T @ r
    L = np.cross(r, du0).sum(axis=0)
    Omega = np.linalg.lstsq(inertia, L, rcond=None)[0]
    return U, Omega, D


def remove_a7(x: np.ndarray, u: np.ndarray) -> Residuals:
    U, Omega, D = fit_rigid(x, u)
    r = x - x.mean(axis=0)
    du = u - U - np.cross(Omega[None, :], r) - D * r
    n_hat = _unit(U)
    ds, dperp = _split(du, n_hat)
    return Residuals("A7", x, du, ds, dperp, n_hat, U, Omega, D)


def remove_mean(x: np.ndarray, u: np.ndarray) -> Residuals:
    U = u.mean(axis=0)
    du = u - U
    n_hat = _unit(U)
    speed = np.linalg.norm(u, axis=1)
    ds = speed - speed.mean()
    dperp = du - (du @ n_hat)[:, None] * n_hat[None, :]
    nan3 = np.full(3, np.nan)
    return Residuals("M", x, du, ds, dperp, n_hat, U, nan3, float("nan"))


def remove_affine(x: np.ndarray, u: np.ndarray) -> Residuals:
    r = x - x.mean(axis=0)
    U = u.mean(axis=0)
    du0 = u - U
    A = np.linalg.lstsq(r, du0, rcond=None)[0].T  # du0_i ≈ A r_i
    du = du0 - r @ A.T
    n_hat = _unit(U)
    ds, dperp = _split(du, n_hat)
    nan3 = np.full(3, np.nan)
    return Residuals("A12", x, du, ds, dperp, n_hat, U, nan3, float("nan"))


def _speed_poly_residual(x: np.ndarray, s: np.ndarray, degree: int) -> np.ndarray:
    """s minus Polynom in r = x − x̄ (Grad 1 oder 2), kleinste Quadrate; bei Rangdefizit (z. B. z ≡ 0)
    die Lösung minimaler Norm."""
    r = x - x.mean(axis=0)
    cols = [np.ones(s.size), *r.T]
    if degree >= 2:
        cols += [r[:, a] * r[:, b] for a in range(3) for b in range(a, 3)]
    A = np.column_stack(cols)
    return s - A @ np.linalg.lstsq(A, s, rcond=None)[0]


def _speed_residuals(variant: str, x: np.ndarray, u: np.ndarray, ds: np.ndarray,
                     Omega: np.ndarray | None = None, D: float = float("nan")) -> Residuals:
    U = u.mean(axis=0)
    du = u - U
    n_hat = _unit(U)
    dperp = du - (du @ n_hat)[:, None] * n_hat[None, :]
    return Residuals(variant, x, du, ds, dperp, n_hat, U, np.full(3, np.nan) if Omega is None else Omega, D)


def remove_speed_linear(x: np.ndarray, u: np.ndarray) -> Residuals:
    return _speed_residuals("L", x, u, _speed_poly_residual(x, np.linalg.norm(u, axis=1), 1))


def remove_speed_quadratic(x: np.ndarray, u: np.ndarray) -> Residuals:
    return _speed_residuals("Q", x, u, _speed_poly_residual(x, np.linalg.norm(u, axis=1), 2))


def remove_rigid_speed(x: np.ndarray, u: np.ndarray) -> Residuals:
    U, Omega, D = fit_rigid(x, u)
    r = x - x.mean(axis=0)
    pred = U + np.cross(Omega[None, :], r) + D * r
    s = np.linalg.norm(u, axis=1) - np.linalg.norm(pred, axis=1)
    return _speed_residuals("RL", x, u, _speed_poly_residual(x, s, 1), Omega, D)


_REMOVERS = {"A7": remove_a7, "M": remove_mean, "A12": remove_affine,
             "L": remove_speed_linear, "Q": remove_speed_quadratic, "RL": remove_rigid_speed}


def remove(variant: str, x: np.ndarray, u: np.ndarray) -> Residuals:
    try:
        return _REMOVERS[variant](np.asarray(x, dtype=float), np.asarray(u, dtype=float))
    except KeyError as exc:
        raise ValueError(f"unbekannte Abzugsvariante {variant!r}; erlaubt: {VARIANTS}") from exc
