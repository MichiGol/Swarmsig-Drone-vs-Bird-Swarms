"""Euler–Maruyama-Integrator und Zeitschrittregel (4.9).

Schritt (semi-implizit: x wird mit dem *neuen* u fortgeschrieben):

    v^{n+1} = v^n + a^n Δt + σ √Δt ξ^n
    ŵ^{n+1} = ŵ^n + Δt/τ_est (w^n − ŵ^n)
    u^{n+1} = v^{n+1} + w^n − c ŵ^{n+1}
    x^{n+1} = x^n + u^{n+1} Δt

Wind (Abweichung von 4.9, Entscheidung 2026-10-11 für P3): w^n = w(x^m, t_{m+1}) wird nur alle
``wind_hold_steps`` Schritte (m = letzter Auswertungsschritt) neu ausgewertet und dazwischen
gehalten; derselbe Wert treibt Schätzer und Übertragung (4.9: w(x^n, t_n) für den Schätzer,
w(x^n, t_{n+1}) für u). Der Unterschied ist O(Δt) bzw. eine Verschiebung des eingefrorenen
Felds um höchstens U_c·Δt_hold (wenige cm) und spart einen Großteil der Windauswertungen.

Zeitschrittregel: Δt ≤ 0,1 / ρ_max. Primär ist ρ_max das 99,9-%-Quantil des größten
Eigenwertbetrags der lokalen Jacobi-Matrix (Pilotlauf, :func:`spectral_radius`); die
analytische Schätzung (:func:`rho_max_analytic`) dient der Plausibilitätskontrolle und,
solange keine Pilotläufe existieren, als konservative Vorgabe.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

from ..environment.wind_profile import NoWind, Wind
from ..state import SwarmState
from .wind_transfer import WindTransfer

#: Zulässige Zeitschritte: 1/f_ctrl ist für 10, 20, 50 Hz ein ganzzahliges Vielfaches (4.9).
ALLOWED_DT_S: tuple[float, ...] = (0.010, 0.005, 0.0025, 0.002, 0.001)


class AgentModel(Protocol):
    sigma: float
    transfer: WindTransfer

    def start(self, state: SwarmState) -> None: ...
    def drift(self, state: SwarmState, step: int) -> np.ndarray: ...
    def constrain(self, v: np.ndarray) -> np.ndarray: ...
    def after_step(self, state: SwarmState, step: int) -> None: ...


StepCallback = Callable[[SwarmState, SwarmState, int], None]


def em_step(state: SwarmState, model: AgentModel, wind: Wind, dt: float, step: int,
            t_new: float, noise_rng: np.random.Generator, w: np.ndarray | None = None) -> SwarmState:
    """Ein Schritt; ``w`` ist der (gehaltene) Wind, sonst w(x^n, t_{n+1})."""
    a = model.drift(state, step)
    xi = noise_rng.standard_normal(state.v.shape)
    v_new = model.constrain(state.v + a * dt + model.sigma * np.sqrt(dt) * xi)
    if w is None:
        w = wind.sample(state.x, t_new)
    u_new, w_hat_new = model.transfer.apply(v_new, w, w, state.w_hat, dt)
    x_new = state.x + u_new * dt
    return SwarmState(t_new, x_new, v_new, u_new, w_hat_new)


def simulate(model: AgentModel, state: SwarmState, *, dt: float, n_steps: int,
             noise_rng: np.random.Generator, wind: Wind | None = None, step0: int = 0,
             t_origin: float = 0.0, callbacks: Iterable[StepCallback] = (),
             wind_hold_steps: int = 1) -> SwarmState:
    """Integriert ``n_steps`` Schritte ab Schrittindex ``step0``.

    Zeiten werden exakt als ``t_origin + step·Δt`` gesetzt (keine Rundungsdrift).
    ``callbacks(prev, new, step)`` werden nach jedem Schritt aufgerufen (Recorder usw.).
    Der Wind wird in Schritten mit ``step % wind_hold_steps == 0`` neu ausgewertet.
    """
    wind = NoWind() if wind is None else wind
    hold = max(int(wind_hold_steps), 1)
    cbs = tuple(callbacks)
    w: np.ndarray | None = None
    for k in range(int(n_steps)):
        step = step0 + k
        t_new = t_origin + (step + 1) * dt
        if w is None or step % hold == 0:
            w = wind.sample(state.x, t_new)
        new = em_step(state, model, wind, dt, step, t_new, noise_rng, w)
        model.after_step(new, step + 1)
        for cb in cbs:
            cb(state, new, step)
        state = new
    return state


# ---------------------------------------------------------------------------
# Zeitschrittregel
# ---------------------------------------------------------------------------


def kappa_eff(kappa_s: float, mode: str, u_q: float = 0.0, u_star: float = 1.0) -> float:
    """Effektive Speed-Steifigkeit: κ_s (steif) bzw. 3 κ_s (u_q/u_*)² (marginal)."""
    if mode == "stiff":
        return float(kappa_s)
    if mode == "marginal":
        return float(3.0 * kappa_s * (u_q / u_star) ** 2)
    raise ValueError(mode)


def omega_hc(beta: float, r_hc: float) -> float:
    """ω_hc = √(768 β / r_hc) aus max|da_hc/dr| = 6 β r_hc⁶ / (r_hc/2)⁷ (4.9)."""
    return float(np.sqrt(768.0 * beta / r_hc)) if beta > 0 else 0.0


def rho_max_analytic(*, J: float = 0.0, kappa_eff: float = 0.0, tau_v: float | None = None,
                     c2: float = 0.0, c1: float = 0.0, omega_hc: float = 0.0,
                     omega_D: float = 0.0) -> float:
    """max(2J, κ_eff, 1/τ_v, c2, √c1, ω_hc, ω_D) (4.9)."""
    terms = [2.0 * J, kappa_eff, c2, np.sqrt(max(c1, 0.0)), omega_hc, omega_D]
    if tau_v is not None:
        terms.append(1.0 / tau_v)
    return float(max(terms))


def dt_from_rule(rho_max: float, *, safety: float = 0.1, dt_max: float | None = None,
                 allowed: Iterable[float] = ALLOWED_DT_S) -> tuple[float, bool]:
    """Größtes zulässiges Δt mit Δt ≤ safety/ρ_max (und ≤ dt_max).

    Rückgabe ``(dt, ok)``; ``ok = False``, wenn selbst das kleinste zulässige Δt die
    Bedingung verletzt (dann wird das kleinste zurückgegeben und der Lauf geflaggt).
    """
    bound = safety / rho_max if rho_max > 0 else np.inf
    cands = sorted((float(d) for d in allowed), reverse=True)
    for d in cands:
        if d <= bound * (1 + 1e-12) and (dt_max is None or d <= dt_max * (1 + 1e-12)):
            return d, True
    return cands[-1], False


def steps_per(period_s: float, dt: float) -> int:
    """Anzahl Schritte für ``period_s``; verlangt ein ganzzahliges Vielfaches (Raster, 4.9)."""
    q = period_s / dt
    k = int(round(q))
    if k < 1 or abs(q - k) > 1e-9 * max(1.0, q):
        raise ValueError(f"{period_s} s ist kein ganzzahliges Vielfaches von Δt = {dt} s")
    return k


def spectral_radius(rhs: Callable[[np.ndarray], np.ndarray], z: np.ndarray,
                    eps: float = 1e-6) -> float:
    """Größter Eigenwertbetrag der Jacobi-Matrix von ``rhs`` an der Stelle ``z``.

    Matrixfrei: Jacobi-Vektor-Produkte per zentraler Differenz, größter Eigenwert per
    Arnoldi (``scipy.sparse.linalg.eigs``). ``rhs`` muss deterministisch und frei von
    Seiteneffekten sein (Verzögerungsglieder als Konstanten, 4.9).
    """
    z = np.asarray(z, dtype=float).ravel()
    n = z.size
    scale = eps * max(1.0, float(np.linalg.norm(z)) / np.sqrt(n))

    def matvec(d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float).ravel()
        nd = np.linalg.norm(d)
        if nd == 0:
            return np.zeros(n)
        h = scale / nd
        return (np.asarray(rhs(z + h * d)).ravel() - np.asarray(rhs(z - h * d)).ravel()) / (2 * h)

    if n <= 64:
        jac = np.column_stack([matvec(e) for e in np.eye(n)])
        return float(np.max(np.abs(np.linalg.eigvals(jac))))
    op = LinearOperator((n, n), matvec=matvec, dtype=float)
    vals = eigs(op, k=1, which="LM", return_eigenvectors=False, tol=1e-4, maxiter=5000)
    return float(np.max(np.abs(vals)))
