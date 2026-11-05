"""Schnellpfad: Numba-Kerne für Vögel und Drohnen, mit und ohne Wind (P1, P3).

Übernimmt ein fertig aufgebautes NumPy-Modell (:class:`~swarmsig.agents.bird.BirdModel`
bzw. :class:`~swarmsig.agents.drone.DroneModel`) samt gezogener Parameter, Puffer und
Zufallsströme und integriert in Numba. Callbacks (Frame-Recorder, Einschwing-Stichprobe)
werden nur für die Schritte aufgerufen, in denen ein Ereignis liegt; dazwischen läuft die
Schleife vollständig kompiliert.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from ..agents import kernels as K
from ..agents.bird import BirdModel
from ..agents.drone import DroneModel
from ..agents.terms import Navigation
from ..environment.wind_kraichnan import TurbulentWind
from ..state import SwarmState

StepCallback = Callable[[SwarmState, SwarmState, int], None]


def _nav_fields(nav: Navigation) -> tuple[int, float, float, float, float, float, float, float, float]:
    e_T = nav.e_T if nav.e_T is not None else np.zeros(3)
    x_c = nav.x_c if nav.x_c is not None else np.zeros(3)
    mode = 0 if nav.mode == "transit" else 1
    return mode, nav.kappa_n, nav.kappa_z, nav.z0, nav.kappa_h, e_T[0], e_T[1], x_c[0], x_c[1]


class _WindArrays:
    """Windfeld als Felder für die Kerne; ``None`` = kein Wind."""

    def __init__(self, wind: TurbulentWind | None, n: int, hold: int) -> None:
        self.on = wind is not None
        self.hold = max(int(hold), 1)
        if wind is None:
            self.wp = np.zeros(K.N_WP)
            self.mk = np.zeros((0, 3))
            self.ma = np.zeros((0, 3))
            self.mb = np.zeros((0, 3))
        else:
            self.wp = wind.params()
            self.mk = np.ascontiguousarray(wind.modes.k)
            self.ma = np.ascontiguousarray(wind.modes.a)
            self.mb = np.ascontiguousarray(wind.modes.b)
        self.w_cur = np.zeros((n, 3))


def _transfer_values(model: BirdModel | DroneModel) -> tuple[float, float]:
    """(c, τ_est) der Windübertragung des Modells (Skalare je Lauf)."""
    c = float(np.asarray(model.transfer.c_comp, dtype=float).reshape(-1)[0])
    tau = float(np.asarray(model.transfer.tau_est, dtype=float).reshape(-1)[0])
    return c, tau


class FastBird:
    def __init__(self, model: BirdModel, dt: float, wind: TurbulentWind | None = None,
                 wind_hold_steps: int = 1) -> None:
        p = model.p
        mode, kn, kz, z0, kh, etx, ety, xcx, xcy = _nav_fields(model.nav)
        pf = np.zeros(K.N_BF)
        pf[[K.B_J, K.B_KAPPA, K.B_S0, K.B_USTAR, K.B_BETA, K.B_R0, K.B_RW, K.B_RHC]] = (
            p.J, p.kappa_s, p.s0, p.u_star, p.beta, p.r0, p.r_w, p.r_hc)
        pf[[K.B_ETA, K.B_TAU0, K.B_RC, K.B_SIGMA, K.B_DT]] = (p.eta_h, p.tau0, p.r_c, p.sigma, dt)
        pf[[K.B_KN, K.B_KZ, K.B_Z0, K.B_KH, K.B_ETX, K.B_ETY, K.B_XCX, K.B_XCY]] = (
            kn, kz, z0, kh, etx, ety, xcx, xcy)
        if p.model != "B3":  # B1: keine Hierarchie
            pf[K.B_ETA] = 0.0
            pf[K.B_TAU0] = 0.0
        pi = np.zeros(K.N_BI, np.int64)
        pi[K.BI_TOPO] = 1 if p.neighborhood == "topological" else 0
        pi[K.BI_NC] = p.n_c
        pi[K.BI_MARGINAL] = 1 if p.speed_control == "marginal" else 0
        pi[K.BI_NAV] = mode
        pi[K.BI_NCOH] = p.n_coh
        self.wind = _WindArrays(wind, model.n, wind_hold_steps)
        pi[K.BI_WIND] = 1 if self.wind.on else 0
        pi[K.BI_WHOLD] = self.wind.hold
        pf[K.B_CCOMP], pf[K.B_TAUEST] = _transfer_values(model)
        self.pf, self.pi = pf, pi
        n = model.n
        self.rank = model.hierarchy.rank.copy() if model.hierarchy is not None else np.zeros(n)
        if model.history is not None and pf[K.B_TAU0] > 0:
            self.hist = model.history._buf["u"].copy()
            self.hc = np.array([model.history._head, model.history.count], np.int64)
        else:
            self.hist = np.zeros((1, n, 3))
            self.hc = np.array([0, 1], np.int64)

    def run(self, x: np.ndarray, v: np.ndarray, u: np.ndarray, w_hat: np.ndarray, n_steps: int,
            step0: int, t0: float, rngs: dict[str, np.random.Generator]) -> None:
        w = self.wind
        K.bird_run(x, v, u, n_steps, step0, t0, self.pf, self.pi, self.rank, self.hist, self.hc,
                   rngs["noise"], w.wp, w.mk, w.ma, w.mb, w_hat, w.w_cur)


class FastDrone:
    def __init__(self, model: DroneModel, dt: float, wind: TurbulentWind | None = None,
                 wind_hold_steps: int = 1) -> None:
        p, rp, c = model.p, model.rp, model.comms
        mode, kn, kz, z0, kh, etx, ety, xcx, xcy = _nav_fields(model.nav)
        pf = np.zeros(K.N_DF)
        vals = {
            K.D_S0: p.s0, K.D_RC: p.r_c, K.D_SPACING: p.spacing, K.D_KAPPA: p.kappa_s,
            K.D_SIGMA: p.sigma_d, K.D_CA: p.c_a, K.D_CC: p.c_c, K.D_CS: p.c_s, K.D_A: p.a,
            K.D_B: p.b, K.D_H: p.h, K.D_EPS: p.eps, K.D_C1: p.c1, K.D_C2: p.c2,
            K.D_RINT: p.r_int, K.D_PREP: p.p_rep, K.D_R0F: p.r0_fric, K.D_CF: p.C_fric,
            K.D_VF: p.v_fric, K.D_PF: p.p_fric, K.D_AF: p.a_fric, K.D_TAUV: rp.tau_v,
            K.D_AMAX: rp.a_max, K.D_VMAX: rp.v_max, K.D_PLOSS: rp.p_loss, K.D_DT: dt,
            K.D_SG: rp.sigma_g, K.D_SGV: rp.sigma_gv, K.D_OUA: c._a, K.D_OUB: c._b,
            K.D_KN: kn, K.D_KZ: kz, K.D_Z0: z0, K.D_KH: kh, K.D_ETX: etx, K.D_ETY: ety,
            K.D_XCX: xcx, K.D_XCY: xcy, K.D_KFORM: p.k_form,
        }
        for k, val in vals.items():
            pf[k] = val
        ref = model.ref
        ref_mode = 0
        if ref is not None:
            ref_mode = 0 if ref.mode == "transit" else 1
            e_T = ref.e_T if ref.e_T is not None else np.zeros(3)
            pf[[K.D_RX0, K.D_RY0, K.D_RZ0]] = ref.x0
            pf[[K.D_RSPEED, K.D_RETX, K.D_RETY, K.D_RRAD, K.D_RPH0, K.D_RT0]] = (
                ref.speed, e_T[0], e_T[1], ref.radius if ref.radius > 0 else 1.0, ref.phase0, ref.t0)
        pi = np.zeros(K.N_DI, np.int64)
        pi[K.DI_KIND] = {"D1": K.KIND_D1, "D2": K.KIND_D2, "D2c": K.KIND_D2C, "D3": K.KIND_D3,
                         "DF": K.KIND_DF}[p.kind]
        pi[K.DI_NCTRL] = c.n_ctrl
        pi[K.DI_LAGD] = c.lag_d
        pi[K.DI_MAXAGE] = c.max_age
        pi[K.DI_KNN] = p.max_neighbors or 0
        pi[K.DI_ATTR] = 1 if p.attraction else 0
        pi[K.DI_NAV] = mode
        pi[K.DI_REF] = ref_mode
        self.wind = _WindArrays(wind, model.n, wind_hold_steps)
        pi[K.DI_WIND] = 1 if self.wind.on else 0
        pi[K.DI_WHOLD] = self.wind.hold
        pf[K.D_CCOMP], pf[K.D_TAUEST] = _transfer_values(model)
        pf[K.D_WI], pf[K.D_WS] = c.w_ind, c.w_sh
        pi[K.DI_OWN] = 1 if rp.own_state_noise else 0
        self.pf, self.pi = pf, pi
        n = model.n
        self.hist_x = c.history._buf["x"].copy()
        self.hist_u = c.history._buf["u"].copy()
        self.hc = np.array([c.history._head, c.history.count], np.int64)
        self.age = np.zeros((n, n), np.int64)
        self.eps_x = c.eps_x.copy()
        self.eps_u = c.eps_u.copy()
        self.eps_xs = c.eps_xs.copy()
        self.eps_us = c.eps_us.copy()
        self.rng_gxs, self.rng_gvs = c._rng_gxs, c._rng_gvs
        self.cmd = np.zeros((n, 3))
        self.offs = model.offsets.copy()
        self.has_cmd = np.zeros(1, np.int64)
        self.rng_gx, self.rng_gv, self.rng_loss = c._rng_gx, c._rng_gv, c._rng_loss

    def run(self, x: np.ndarray, v: np.ndarray, u: np.ndarray, w_hat: np.ndarray, n_steps: int,
            step0: int, t0: float, rngs: dict[str, np.random.Generator]) -> None:
        w = self.wind
        K.drone_run(x, v, u, n_steps, step0, t0, self.pf, self.pi, self.cmd, self.has_cmd,
                    self.hist_x, self.hist_u, self.hc, self.age, self.eps_x, self.eps_u,
                    rngs["noise"], self.rng_gx, self.rng_gv, self.rng_loss,
                    w.wp, w.mk, w.ma, w.mb, w_hat, w.w_cur,
                    self.eps_xs, self.eps_us, self.rng_gxs, self.rng_gvs, self.offs)

    def own_errors(self) -> tuple[np.ndarray, np.ndarray]:
        """Aktuelle GPS-Fehler (Position, Geschwindigkeit) aller Drohnen für ``drone_command``."""
        wi, ws = self.pf[K.D_WI], self.pf[K.D_WS]
        return wi * self.eps_x + ws * self.eps_xs, wi * self.eps_u + ws * self.eps_us


def make_fast(model: BirdModel | DroneModel, dt: float, wind: TurbulentWind | None = None,
              wind_hold_steps: int = 1) -> FastBird | FastDrone:
    if isinstance(model, BirdModel):
        return FastBird(model, dt, wind, wind_hold_steps)
    if isinstance(model, DroneModel):
        return FastDrone(model, dt, wind, wind_hold_steps)
    raise TypeError(type(model))


def event_steps(times: Iterable[float], dt: float, t_origin: float = 0.0) -> list[int]:
    """Schrittindizes n, deren Intervall [t_n, t_{n+1}] einen Ereigniszeitpunkt enthält."""
    out = set()
    for t in times:
        n = int(np.ceil((t - t_origin) / dt - 1e-9)) - 1
        out.add(max(n, 0))
    return sorted(out)


def simulate_fast(fast: FastBird | FastDrone, state: SwarmState, *, dt: float, n_steps: int,
                  noise_rng: np.random.Generator, event_times: Iterable[float] = (),
                  callbacks: Iterable[StepCallback] = (), t_origin: float = 0.0) -> SwarmState:
    """Wie :func:`~swarmsig.dynamics.integrator.simulate`, aber kompiliert (Wind über ``fast``).

    Callbacks erhalten ``(prev, new, step)`` nur für Schritte, deren Intervall einen der
    ``event_times`` enthält (Frames, Stichprobenzeitpunkte).
    """
    x, v, u, w_hat = state.x.copy(), state.v.copy(), state.u.copy(), state.w_hat.copy()
    rngs = {"noise": noise_rng}
    cbs = tuple(callbacks)
    step = 0
    for ev in [s for s in event_steps(event_times, dt, t_origin) if s < n_steps] + [n_steps]:
        if ev > step:
            fast.run(x, v, u, w_hat, ev - step, step, t_origin, rngs)
            step = ev
        if step >= n_steps:
            break
        prev = SwarmState(t_origin + step * dt, x.copy(), v.copy(), u.copy(), w_hat.copy())
        fast.run(x, v, u, w_hat, 1, step, t_origin, rngs)
        step += 1
        new = SwarmState(t_origin + step * dt, x.copy(), v.copy(), u.copy(), w_hat.copy())
        for cb in cbs:
            cb(prev, new, step - 1)
    return SwarmState(t_origin + step * dt, x, v, u, w_hat)
