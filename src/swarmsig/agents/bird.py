"""Vogelmodelle B1 (Star, große Schwärme) und B3 (Kleinschwärme), Abschnitte 4.2 und 4.4.

    dv_i = (a_align + a_speed + a_coh + a_nav) dt + σ dW_i

B1: topologische Nachbarn, Ausrichtung im Bodensystem, marginale Speed-Regelung.
B3: hierarchische, verzögerte Ausrichtung; Nachbarschaft topologisch oder metrisch und
Speed-Regelung steif oder marginal werden je Lauf gezogen.
(B2, Inertial Spin Model, ist optional und gehört nach ``swarmsig/optional``.)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from ..config import BirdCfg, sample
from ..dynamics.history import History
from ..dynamics.integrator import kappa_eff, omega_hc, rho_max_analytic
from ..dynamics.wind_transfer import WindTransfer
from ..state import SwarmState
from . import neighbors
from .hierarchy import Hierarchy, hierarchical_obs
from .terms import Align, Cohesion, NeighborObs, Navigation, Speed

RngFor = Callable[[str], np.random.Generator]


@lru_cache(maxsize=8)
def load_posterior(path: str) -> pd.DataFrame:
    """Akzeptierte Parametersätze der Kalibrierung (``scripts/calibrate_bird_model.py``)."""
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"Posterior {path} ist leer")
    return df.reset_index(drop=True)


@dataclass(frozen=True)
class BirdParams:
    model: str
    neighborhood: str
    n_c: int
    r_c: float
    speed_control: str
    s0: float
    J: float
    sigma: float
    kappa_s: float
    u_star: float
    beta: float
    r0: float
    r_w: float
    r_hc: float
    eta_h: float
    tau0: float
    c_comp: float
    tau_est: float
    dt_hint: float = 0.0  # Zeitschritt aus der Kalibrierung dieses Parametersatzes (0 = keiner)
    # Zusammenhalt über die n_coh nächsten Nachbarn (Entscheidung 2026-10-11, gegen Zerfall);
    # 0 = über dieselben Nachbarn wie die Ausrichtung (wörtlich 4.2)
    n_coh: int = 0

    def as_row(self, prefix: str = "b_") -> dict[str, object]:
        return {prefix + k: v for k, v in asdict(self).items()}


def draw_bird_params(model: str, cfg: BirdCfg, s0: float, rng_for: RngFor) -> BirdParams:
    """Zieht die Parameter eines Vogelmodells.

    Jeder Parameter hat seinen eigenen benannten Unterstrom (``rng_for(name)``), damit
    Wahl-Variablen (z. B. die Nachbarschaftsart in B3) die übrigen Ziehungen nicht
    verschieben. ``s0`` ist eine Designkovariate (gleich für Vögel und Drohnen, 9.3).
    """
    post = load_posterior(cfg.calibration_posterior) if cfg.calibration_posterior else None
    # Strukturvarianten (Nachbarschaft, Speed-Regelung) bleiben Designfaktoren mit den Anteilen
    # aus der Config (4.4: je 50 %); die Kalibrierzeile wird passend zur Variante gezogen.
    structural = {k: sample(getattr(cfg, k), rng_for(k)) for k in ("neighborhood", "speed_control")}
    row = None
    if post is not None:
        cand = post
        for k, val in structural.items():
            if k in post.columns and (post[k] == val).any():
                cand = cand[cand[k] == val]
        row = cand.iloc[int(rng_for("posterior_row").integers(len(cand)))]

    def d(name: str) -> object:
        if name in structural:
            return structural[name]
        if row is not None and name in row.index:
            return row[name]
        return sample(getattr(cfg, name), rng_for(name))

    return BirdParams(
        model=model,
        neighborhood=str(d("neighborhood")),
        n_c=int(d("n_c")),
        r_c=float(d("r_c_m")),
        speed_control=str(d("speed_control")),
        s0=float(s0),
        J=float(d("J_per_s")),
        sigma=float(d("sigma")),
        kappa_s=float(d("kappa_s_per_s")),
        u_star=float(d("u_star_mps")),
        beta=float(d("beta_mps2")),
        r0=float(d("r0_m")),
        r_w=float(d("r_w_m")),
        r_hc=float(d("r_hc_m")),
        eta_h=float(d("eta_h")),
        tau0=float(d("tau0_s")),
        c_comp=float(d("c_comp")),
        tau_est=float(d("tau_est_s")),
        dt_hint=float(row["dt_s"]) if row is not None and "dt_s" in row.index else 0.0,
        n_coh=int(d("n_coh")),
    )


def bird_rho_max(p: BirdParams, u_q: float) -> float:
    """Analytische Plausibilitätsschätzung von ρ_max (4.9): max(2J, κ_eff, ω_hc).

    ``u_q`` ist das 99,9-%-Quantil von |s − s0| (aus einem Pilotlauf; bis dahin Config-Wert).
    """
    return rho_max_analytic(J=p.J, kappa_eff=kappa_eff(p.kappa_s, p.speed_control, u_q, p.u_star),
                            omega_hc=omega_hc(p.beta, p.r_hc))


class BirdModel:
    """Laufzeitobjekt eines Vogelmodells (ein Objekt pro Lauf)."""

    def __init__(self, p: BirdParams, nav: Navigation, dt: float, n: int,
                 rank_rng: np.random.Generator | None = None) -> None:
        self.p = p
        self.nav = nav
        self.dt = float(dt)
        self.n = int(n)
        self.sigma = p.sigma
        self.transfer = WindTransfer(p.c_comp, p.tau_est)
        self.align = Align(p.J)
        self.speed = Speed(p.kappa_s, p.s0, p.speed_control, p.u_star)
        self.coh = Cohesion(p.beta, p.r0, p.r_w, p.r_hc)
        self.hierarchy: Hierarchy | None = None
        self.history: History | None = None
        if p.model == "B3":
            if rank_rng is None:
                raise ValueError("B3 braucht einen Generator für die Ränge")
            self.hierarchy = Hierarchy.draw(n, p.eta_h, p.tau0, dt, rank_rng)
            if self.hierarchy.max_lag_steps > 0:
                self.history = History(self.hierarchy.max_lag_steps + 1, n, fields=("u",))

    # --- Schnittstelle für den Integrator ---------------------------------------
    def start(self, state: SwarmState) -> None:
        if self.history is not None:
            self.history.push(u=state.u)

    def neighbor_graph(self, x: np.ndarray) -> neighbors.NeighborGraph:
        return neighbors.build(x, self.p.neighborhood, n_c=self.p.n_c, r_c=self.p.r_c)

    def observe(self, state: SwarmState, g: neighbors.NeighborGraph) -> NeighborObs:
        if self.hierarchy is not None:
            return hierarchical_obs(state, g, self.hierarchy, self.history)
        return NeighborObs.exact(state, g)

    def drift(self, state: SwarmState, step: int) -> np.ndarray:  # noqa: ARG002
        g = self.neighbor_graph(state.x)
        obs = self.observe(state, g)
        if self.p.n_coh > 0:
            obs_coh = NeighborObs.exact(state, neighbors.topological(state.x, self.p.n_coh))
        else:
            obs_coh = obs
        return (self.align(state, obs) + self.speed(state, obs) + self.coh(state, obs_coh)
                + self.nav(state, obs))

    def constrain(self, v: np.ndarray) -> np.ndarray:
        return v

    def after_step(self, state: SwarmState, step: int) -> None:  # noqa: ARG002
        if self.history is not None:
            self.history.push(u=state.u)

    # --- Zeitschrittregel -----------------------------------------------------
    def rho_max_estimate(self, u_q: float) -> float:
        return bird_rho_max(self.p, u_q)

    def rhs(self, state: SwarmState) -> np.ndarray:
        """Deterministische rechte Seite (ẋ, v̇) ohne Wind, für :func:`spectral_radius`."""
        a = self.drift(state, 0)
        return np.concatenate([state.u.ravel(), a.ravel()])
