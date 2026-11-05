"""Drohnenmodelle D1, D2, D2-c, D3 (4.5) und DF (Formation ohne Nachbarkopplung, O5) mit Realismus-Schicht (4.7).

Zwei Schichten: Schwarmregler (Befehl aus verrauschten, verzögerten Nachbarinformationen
x̃_j, ũ_j; eigener Zustand x_i, u_i exakt wie in den Formeln von 4.5) und Realismus-Schicht
(:mod:`.realism`). Wind über die einheitliche Übertragung mit c_D = 1.

Kommunikationsgraph: alle Nachbarn mit wahrem Abstand < r_c (Funkreichweite).

Modell I (bio-inspirierte Drohnen, 4.6) gehört zu P4 und ist noch nicht umgesetzt.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from ..config import DroneCfg, sample
from ..dynamics.integrator import rho_max_analytic
from ..dynamics.wind_transfer import WindTransfer
from ..state import SwarmState
from . import neighbors
from .realism import Comms, RealismParams, saturate_norm
from .terms import Navigation

RngFor = Callable[[str], np.random.Generator]

DRONE_KINDS = ("D1", "D2", "D2c", "D3", "DF")


@dataclass(frozen=True)
class DroneParams:
    kind: str
    s0: float
    r_c: float
    spacing: float
    kappa_s: float
    sigma_d: float
    c_comp: float
    tau_est: float
    # D1
    c_a: float
    c_c: float
    c_s: float
    # D2
    a: float
    b: float
    h: float
    eps: float
    c1: float
    c2: float
    r_int: float
    max_neighbors: int | None
    range_mode: str
    # D3
    p_rep: float
    r0_fric: float
    C_fric: float
    v_fric: float
    p_fric: float
    a_fric: float
    attraction: bool
    # DF (O5): Positionsverstärkung der Formationsregelung [1/s]
    k_form: float = 0.0

    def as_row(self, prefix: str = "d_") -> dict[str, object]:
        return {prefix + k: v for k, v in asdict(self).items()}


def draw_drone_params(kind: str, cfg: DroneCfg, s0: float, rng_for: RngFor,
                      spacing_override: float | None = None) -> DroneParams:
    """Zieht die Parameter eines Drohnenreglers (benannte Unterströme, siehe bird.py).

    ``spacing_override`` setzt den Wunschabstand (Kontrolle K-Dichte, „-dicht“-Variante).
    """
    if kind not in DRONE_KINDS:
        raise ValueError(f"unbekannte Drohnenklasse {kind!r}")
    d = lambda section, name: sample(getattr(section, name), rng_for(name))  # noqa: E731
    spacing = float(d(cfg, "spacing_m")) if spacing_override is None else float(spacing_override)
    r_c = float(d(cfg, "r_c_m"))
    c_c = float(d(cfg.D1, "c_c_per_s2"))
    # D (Design-Entscheidung): c_s = Faktor · c_c · r_s², damit ein einzelnes Paar im
    # Gleichgewicht c_c r = c_s / r beim Abstand r_s steht (Faktor 1).
    c_s = float(d(cfg.D1, "c_s_factor")) * c_c * spacing**2
    range_mode = str(d(cfg.D2, "range_mode"))
    # D1/D3: Nachbarbegrenzung aus der Config (Entscheidung 2026-10-11, SwarmLab max_neig);
    # D2: über den Reichweitenmodus.
    max_neighbors: int | None = cfg.max_neighbors if kind in ("D1", "D3") else None
    if kind in ("D2", "D2c"):
        if range_mode == "literal":
            r_int = r_c  # wörtlich nach 4.5: r_α = ‖r_c‖_σ
        elif range_mode == "local":
            r_int = min(r_c, cfg.D2.kappa_range * spacing)  # Olfati-Saber: r = κ d
        elif range_mode == "knn":
            r_int, max_neighbors = r_c, int(cfg.D2.knn)  # SwarmLab: k nächste innerhalb r_c
        else:
            raise ValueError(f"unbekannter D2-Reichweitenmodus {range_mode!r}")
    else:
        r_int, range_mode = r_c, ""
    return DroneParams(
        kind=kind, s0=float(s0), r_c=r_c, spacing=spacing,
        kappa_s=float(d(cfg, "kappa_s_per_s")), sigma_d=float(d(cfg, "sigma_d")),
        c_comp=float(d(cfg, "c_comp")), tau_est=float(d(cfg, "tau_est_s")),
        c_a=float(d(cfg.D1, "c_a_per_s")), c_c=c_c, c_s=c_s,
        a=float(d(cfg.D2, "a")), b=float(d(cfg.D2, "b")), h=float(d(cfg.D2, "h")),
        eps=float(d(cfg.D2, "eps")), c1=float(d(cfg.D2, "c1_per_s2")),
        c2=float(d(cfg.D2, "c2_per_s")), r_int=float(r_int),
        max_neighbors=max_neighbors, range_mode=range_mode,
        p_rep=float(d(cfg.D3, "p_rep_per_s")), r0_fric=float(d(cfg.D3, "r0_fric_m")),
        C_fric=float(d(cfg.D3, "C_fric")), v_fric=float(d(cfg.D3, "v_fric_mps")),
        p_fric=float(d(cfg.D3, "p_fric_per_s")), a_fric=float(d(cfg.D3, "a_fric_mps2")),
        attraction=bool(cfg.D3.attraction),
        k_form=float(d(cfg.DF, "k_form_per_s")) if (kind == "DF" and cfg.DF is not None) else 0.0,
    )


# ---------------------------------------------------------------------------
# Olfati-Saber-Hilfsfunktionen (D2, gegen Original prüfen: offener Punkt 5)
# ---------------------------------------------------------------------------


def sigma_norm(z: np.ndarray | float, eps: float) -> np.ndarray:
    """‖z‖_σ = (√(1 + ε‖z‖²) − 1)/ε für Abstände ‖z‖ (Skalar oder Array)."""
    z = np.asarray(z, dtype=float)
    return (np.sqrt(1.0 + eps * z**2) - 1.0) / eps


def rho_h(z: np.ndarray, h: float) -> np.ndarray:
    """Bump-Funktion: 1 auf [0, h), ½[1 + cos(π (z−h)/(1−h))] auf [h, 1], sonst 0."""
    z = np.asarray(z, dtype=float)
    mid = 0.5 * (1.0 + np.cos(np.pi * (z - h) / (1.0 - h)))
    return np.where((z >= 0) & (z < h), 1.0, np.where((z >= h) & (z <= 1.0), mid, 0.0))


def sigma1(z: np.ndarray) -> np.ndarray:
    return z / np.sqrt(1.0 + z**2)


def phi(z: np.ndarray, a: float, b: float) -> np.ndarray:
    """φ(z) = ½[(a+b) σ₁(z+c) + (a−b)], c = |a−b|/√(4ab); φ(0) = 0 für 0 < a ≤ b."""
    c = abs(a - b) / np.sqrt(4.0 * a * b)
    return 0.5 * ((a + b) * sigma1(np.asarray(z, dtype=float) + c) + (a - b))


def phi_alpha(z: np.ndarray, p: DroneParams) -> np.ndarray:
    r_alpha = sigma_norm(p.r_int, p.eps)
    d_alpha = sigma_norm(p.spacing, p.eps)
    return rho_h(z / r_alpha, p.h) * phi(z - d_alpha, p.a, p.b)


def braking_curve(r: np.ndarray, a: float, p: float) -> np.ndarray:
    """D(r, a, p) der Vásárhelyi-Reibung: 0 (r<0), p·r (0 ≤ r ≤ a/p²), √(2ar − a²/p²) sonst."""
    r = np.asarray(r, dtype=float)
    lin = p * r
    sq = np.sqrt(np.maximum(2.0 * a * r - a**2 / p**2, 0.0))
    return np.where(r < 0, 0.0, np.where(r <= a / p**2, lin, sq))


# ---------------------------------------------------------------------------
# Virtueller Referenzagent (D2, D2-c)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reference:
    """Virtueller Referenzagent (x_r, u_r): gerade Bahn (Transit) oder Kreisbahn (Kreisen)."""

    mode: str
    x0: np.ndarray  # Startposition (Transit) bzw. Kreismittelpunkt inkl. Höhe (Kreisen)
    speed: float
    e_T: np.ndarray | None = None
    radius: float = 0.0
    phase0: float = 0.0
    t0: float = 0.0

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        tau = t - self.t0
        if self.mode == "transit":
            assert self.e_T is not None
            return self.x0 + self.speed * self.e_T * tau, self.speed * self.e_T
        omega = self.speed / self.radius
        ph = self.phase0 + omega * tau
        x = self.x0 + self.radius * np.array([np.cos(ph), np.sin(ph), 0.0])
        u = self.speed * np.array([-np.sin(ph), np.cos(ph), 0.0])
        return x, u


# ---------------------------------------------------------------------------
# Laufzeitmodell
# ---------------------------------------------------------------------------


class DroneModel:
    """Laufzeitobjekt eines Drohnenlaufs (Regler + Realismus-Schicht)."""

    def __init__(self, p: DroneParams, rp: RealismParams, nav: Navigation, dt: float, n: int,
                 rng_for: RngFor, max_packet_age_s: float, reference: Reference | None = None) -> None:
        if p.kind in ("D2", "D2c", "DF") and reference is None:
            raise ValueError("D2/D2-c/DF brauchen einen Referenzagenten")
        self.p = p
        self.rp = rp
        self.nav = nav
        self.ref = reference
        self.dt = float(dt)
        self.n = int(n)
        self.sigma = p.sigma_d
        self.transfer = WindTransfer(1.0, p.tau_est)  # c_D = 1 (4.1, 9.3)
        self.comms = Comms(n, dt, rp, max_packet_age_s, rng_for)
        self.cmd: np.ndarray | None = None
        self.n_ctrl = self.comms.n_ctrl
        self.offsets = np.zeros((self.n, 3))  # DF: feste Versätze in der Formation

    # --- Schnittstelle für den Integrator ---------------------------------------
    def start(self, state: SwarmState) -> None:
        if self.p.kind == "DF":  # Startformation relativ zum Schwerpunkt wird gehalten
            self.offsets = state.x - state.x.mean(axis=0)
        self.comms.start(state)

    def drift(self, state: SwarmState, step: int) -> np.ndarray:
        if self.cmd is None or step % self.n_ctrl == 0:
            self.cmd = self.command(state)
        if self.p.kind in ("D2", "D2c"):
            acc = self.cmd  # Beschleunigungsbefehl, nur gesättigt (4.7)
        else:
            v_est = self.comms.own_estimate(state)[2]  # innerer Regelkreis auf der eigenen Schätzung
            acc = (self.cmd - v_est) / self.rp.tau_v
        return saturate_norm(acc, self.rp.a_max)

    def constrain(self, v: np.ndarray) -> np.ndarray:
        return saturate_norm(v, self.rp.v_max)

    def after_step(self, state: SwarmState, step: int) -> None:  # noqa: ARG002
        self.comms.after_step(state)

    # --- Regler ----------------------------------------------------------------
    def graph(self, x: np.ndarray, r: float, max_neighbors: int | None = None) -> neighbors.NeighborGraph:
        g = neighbors.metric(x, r)
        if max_neighbors is not None:
            g = neighbors.limit_nearest(g, x, max_neighbors)
        return g

    def command(self, state: SwarmState) -> np.ndarray:
        kind = self.p.kind
        if kind == "D1":
            return self._cmd_d1(state)
        if kind in ("D2", "D2c"):
            return self._cmd_d2(state, centroid=(kind == "D2c"))
        if kind == "D3":
            return self._cmd_d3(state)
        if kind == "DF":
            return self._cmd_df(state)
        raise ValueError(kind)

    def _own_heading(self, v: np.ndarray) -> np.ndarray:
        s = np.linalg.norm(v, axis=1)
        out = np.zeros_like(v)
        ok = s > 0
        out[ok] = v[ok] / s[ok, None]
        return out

    def _cmd_d1(self, state: SwarmState) -> np.ndarray:
        """Reynolds-Boids (metrisch, leiterlos) nach 4.5 (D1).

        Abweichung (Entscheidung 2026-10-11): Vorfaktor τ_v statt Δt_c. Mit
        dv/dt = (v_cmd − v)/τ_v wirken die Terme dann als Beschleunigung, unabhängig vom
        Regeltakt; sonst würde ein Sweep über f_ctrl zugleich die Reglerstärke ändern.
        """
        p, tau = self.p, self.rp.tau_v
        xo, uo, vo = self.comms.own_estimate(state)  # eigener Zustand (exakt oder geschätzt)
        g = self.graph(state.x, p.r_c, p.max_neighbors)
        obs = self.comms.observe(g)
        src = g.src
        d = obs.x_nb - xo[src]
        r = np.linalg.norm(d, axis=1)
        align = g.segment_mean(obs.u_nb - uo[src])  # ⟨ũ⟩ − u_i
        coh = g.segment_mean(d)  # ⟨x̃⟩ − x_i
        m = (r < p.spacing) & (r > 0)
        sep = g.segment_sum(np.where(m[:, None], d / np.where(m, r, 1.0)[:, None] ** 2, 0.0))
        s = np.linalg.norm(vo, axis=1)
        speed = -p.kappa_s * (s - p.s0)[:, None] * self._own_heading(vo)
        a_nav = self.nav.acceleration(state.x.mean(axis=0), state.u.mean(axis=0))
        return vo + tau * (p.c_a * align + p.c_c * coh - p.c_s * sep + speed) + tau * a_nav

    def _cmd_d2(self, state: SwarmState, centroid: bool) -> np.ndarray:
        """Olfati-Saber (2006), α-Agenten + Navigation; D2-c: Navigation auf den Schwerpunkt."""
        p = self.p
        xo, uo, _ = self.comms.own_estimate(state)
        g = self.graph(state.x, p.r_int, p.max_neighbors)
        obs = self.comms.observe(g)
        src = g.src
        dx = obs.x_nb - xo[src]
        dist = np.linalg.norm(dx, axis=1)
        z = sigma_norm(dist, p.eps)
        n_ij = dx / np.sqrt(1.0 + p.eps * dist**2)[:, None]
        grad = g.segment_sum(phi_alpha(z, p)[:, None] * n_ij)
        a_ij = rho_h(z / sigma_norm(p.r_int, p.eps), p.h)
        cons = g.segment_sum(a_ij[:, None] * (obs.u_nb - uo[src]))
        assert self.ref is not None
        x_r, u_r = self.ref.at(state.t)
        if centroid:
            nav = -p.c1 * (state.x.mean(axis=0) - x_r) - p.c2 * (state.u.mean(axis=0) - u_r)
            nav = np.broadcast_to(nav, state.x.shape)
        else:
            nav = -p.c1 * (xo - x_r) - p.c2 * (uo - u_r)
        return grad + cons + nav

    def _cmd_d3(self, state: SwarmState) -> np.ndarray:
        """Vásárhelyi et al. (2018): Eigenantrieb, Abstoßung, Reibung; Wände -> a_nav.

        Formeln nach Original bzw. SwarmLab-Referenz (compute_vel_vasarhelyi.m), mit
        * Reibungsrichtung (ũ_j − u_i)/|ũ_j − u_i| (reduziert die Geschwindigkeitsdifferenz),
        * Navigation als Zusatz τ_v·a_nav zum Sollwert (D): mit dv/dt = (v_cmd − v)/τ_v
          wirkt a_nav dann als additive Beschleunigung wie bei den Vögeln,
        * optionaler Anziehung jenseits von r0_rep (SwarmLab-Modifikation, ``attraction``).
        """
        p, rp = self.p, self.rp
        xo, uo, vo = self.comms.own_estimate(state)
        g = self.graph(state.x, p.r_c, p.max_neighbors)
        obs = self.comms.observe(g)
        src = g.src
        dx = obs.x_nb - xo[src]
        r = np.linalg.norm(dx, axis=1)
        e = np.where((r > 0)[:, None], dx / np.where(r > 0, r, 1.0)[:, None], 0.0)
        rep = np.where((r < p.spacing)[:, None], -p.p_rep * (p.spacing - r)[:, None] * e, 0.0)
        if p.attraction:
            rep = rep + np.where((r >= p.spacing)[:, None], p.p_rep * (r - p.spacing)[:, None] * e, 0.0)
        dv = obs.u_nb - uo[src]
        vij = np.linalg.norm(dv, axis=1)
        vmax_fr = np.maximum(p.v_fric, braking_curve(r - p.r0_fric, p.a_fric, p.p_fric))
        act = vij > vmax_fr
        fr = np.where(act[:, None], p.C_fric * (vij - vmax_fr)[:, None] * dv / np.where(act, vij, 1.0)[:, None], 0.0)
        a_nav = self.nav.acceleration(state.x.mean(axis=0), state.u.mean(axis=0))
        v_tilde = (p.s0 * self._own_heading(vo) + g.segment_sum(rep) + g.segment_sum(fr)
                   + rp.tau_v * a_nav)
        return saturate_norm(v_tilde, rp.v_max)

    def _cmd_df(self, state: SwarmState) -> np.ndarray:
        """Formation ohne Nachbarkopplung (O5): jede Drohne folgt ihrer verschobenen Sollbahn.

        v_cmd,i = u_r + k_form (x_r + o_i − x̂_i), o_i = Startversatz zum Schwerpunkt; x̂_i ist die eigene
        (gegebenenfalls GPS-fehlerbehaftete) Positionsschätzung. Wie bei Lichtshow- oder Wegpunktdrohnen gibt es
        keinen Bezug zu Nachbarn, die Abweichungen entstehen nur aus Rauschen, Wind und eigenen Messfehlern.
        """
        xo, _, _ = self.comms.own_estimate(state)
        assert self.ref is not None
        x_r, u_r = self.ref.at(state.t)
        return saturate_norm(u_r + self.p.k_form * (x_r + self.offsets - xo), self.rp.v_max)

    # --- Zeitschrittregel -----------------------------------------------------
    def rho_max_estimate(self) -> float:
        return drone_rho_max(self.p, self.rp)

    @property
    def latency_rounding_s(self) -> float:
        return self.comms.latency_rounding_s


def drone_rho_max(p: DroneParams, rp: RealismParams) -> float:
    """Analytische Schätzung von ρ_max (4.9).

    Befehle werden gehalten (Zero-Order-Hold); für den Integrator wirkt bei D1/D3 nur die
    Rückführung 1/τ_v. Bei D2 werden c2, √c1 und ω_D (aus der Steigung der Kraft φ_α)
    konservativ mitgenommen.
    """
    if p.kind in ("D2", "D2c"):
        r = np.linspace(1e-3, p.r_int, 4000)
        force = phi_alpha(sigma_norm(r, p.eps), p) * r / np.sqrt(1 + p.eps * r**2)
        omega_d = float(np.sqrt(np.max(np.abs(np.gradient(force, r)))))
        return rho_max_analytic(c2=p.c2, c1=p.c1, omega_D=omega_d)
    return max(rho_max_analytic(tau_v=rp.tau_v), p.k_form)
