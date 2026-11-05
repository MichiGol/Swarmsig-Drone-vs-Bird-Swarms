"""Numba-Schnellpfad gegen die NumPy-Referenz: gleiche Beschleunigungen/Befehle und gleiche
Trajektorien (gleiche Zufallsströme), für alle Modelle und Varianten."""

import numpy as np
import pytest

from swarmsig.agents import kernels as K
from swarmsig.agents.bird import BirdModel, BirdParams
from swarmsig.agents.drone import DroneModel, DroneParams, Reference
from swarmsig.agents.realism import RealismParams
from swarmsig.agents.terms import Navigation
from swarmsig.dynamics.fast import FastBird, make_fast, simulate_fast
from swarmsig.dynamics.init import ellipsoid_positions
from swarmsig.dynamics.integrator import simulate
from swarmsig.rng import name_key
from swarmsig.sensors.interpolate import FrameRecorder
from swarmsig.state import SwarmState

DT = 0.005
E_T = np.array([0.0, 1.0, 0.0])


def _state(n, spacing, seed):
    rng = np.random.default_rng(seed)
    x = ellipsoid_positions(n, np.array([0.0, 0.0, 100.0]), spacing, (1.0, 1.0, 0.5), E_T, rng)
    v = 10.0 * E_T + 0.4 * rng.standard_normal((n, 3))
    return SwarmState.initial(0.0, x, v)


def _nav(mode="transit"):
    if mode == "transit":
        return Navigation("transit", z0=100.0, kappa_z=0.05, e_T=E_T, kappa_n=0.5)
    return Navigation("loiter", z0=100.0, kappa_z=0.05, x_c=np.array([50.0, 0.0, 100.0]), kappa_h=0.004)


def _bird(model="B1", neigh="topological", speed="marginal", n=60, nav="transit", seed=1, n_coh=0):
    p = BirdParams(model, neigh, 7, 2.5, speed, 11.0, 4.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.5, 0.2,
                   1.5 if model == "B3" else 0.0, 0.2 if model == "B3" else 0.0, 0.0, 5.0,
                   n_coh=n_coh)
    return BirdModel(p, _nav(nav), DT, n, rank_rng=np.random.default_rng(seed))


@pytest.mark.parametrize("model,neigh,speed,nav,n_coh", [
    ("B1", "topological", "marginal", "transit", 0),
    ("B3", "metric", "stiff", "loiter", 0),
    ("B3", "topological", "marginal", "transit", 0),
    ("B1", "topological", "marginal", "transit", 20),
    ("B3", "metric", "stiff", "loiter", 15),
])
def test_bird_drift_matches_numpy(model, neigh, speed, nav, n_coh):
    m = _bird(model, neigh, speed, nav=nav, n_coh=n_coh)
    s = _state(60, 1.0, 3)
    m.start(s)
    rng = np.random.default_rng(5)
    for _ in range(30):  # Verlauf in den Verzögerungspuffer schreiben
        s = s.replace(u=s.u + 0.05 * rng.standard_normal(s.u.shape))
        m.after_step(s, 0)
    ref = m.drift(s, 0)
    fb = FastBird(m, DT)
    navv = np.zeros(3)
    K._nav(s.x, s.u, fb.pi[K.BI_NAV], *fb.pf[[K.B_KN, K.B_KZ, K.B_Z0, K.B_KH, K.B_ETX, K.B_ETY,
                                              K.B_XCX, K.B_XCY]], navv)
    out = np.empty_like(s.x)
    K.bird_drift(s.x, s.u, s.v, fb.pf, fb.pi, fb.rank, fb.hist, fb.hc[0], fb.hc[1], navv, out)
    assert np.max(np.abs(out - ref)) < 1e-11


def test_bird_trajectory_matches_numpy():
    s0 = _state(50, 1.0, 4)
    m1 = _bird("B3", "topological", "marginal", n=50, n_coh=20)
    m2 = _bird("B3", "topological", "marginal", n=50, n_coh=20)
    m1.start(s0)
    m2.start(s0)
    a = simulate(m1, s0, dt=DT, n_steps=80, noise_rng=np.random.default_rng(9))
    b = simulate_fast(make_fast(m2, DT), s0, dt=DT, n_steps=80, noise_rng=np.random.default_rng(9))
    assert np.max(np.abs(a.x - b.x)) < 1e-9 and np.max(np.abs(a.v - b.v)) < 1e-9


def _drone(kind, *, range_mode="local", attraction=True, p_loss=0.3, n=40, seed=0, nav="transit",
           knn=None, own=False, shared=0.0):
    spacing = 4.0
    if kind in ("D1", "D3"):
        range_mode = "knn" if knn else "local"
    dp = DroneParams(kind=kind, s0=10.0, r_c=25.0, spacing=spacing, kappa_s=1.0, sigma_d=0.1,
                     c_comp=1.0, tau_est=5.0, c_a=1.5, c_c=0.1, c_s=0.1 * spacing**2, a=5.0, b=5.0,
                     h=0.2, eps=0.1, c1=0.04, c2=0.4,
                     r_int=1.2 * spacing if (range_mode == "local" and kind in ("D2", "D2c")) else 25.0,
                     max_neighbors=(knn or 10) if range_mode == "knn" else None, range_mode=range_mode,
                     p_rep=0.03, r0_fric=85.3, C_fric=0.05, v_fric=0.63, p_fric=3.2, a_fric=4.16,
                     attraction=attraction, k_form=0.8 if kind == "DF" else 0.0)
    rp = RealismParams(f_ctrl=20.0, tau_d=0.05, sigma_g=0.3, sigma_gv=0.05, tau_g=5.0,
                       p_loss=p_loss, tau_v=0.3, a_max=6.0, v_max=20.0, gps_shared_frac=shared,
                       own_state_noise=own)
    if nav == "transit":
        ref = Reference("transit", x0=np.array([0.0, 0.0, 100.0]), speed=10.0, e_T=E_T)
    else:
        ref = Reference("loiter", x0=np.array([50.0, 0.0, 100.0]), speed=10.0, radius=50.0, phase0=np.pi)
    rng_for = lambda name: np.random.default_rng([seed, name_key(name)])  # noqa: E731
    return DroneModel(dp, rp, _nav(nav), DT, n, rng_for, 1.0, ref)


VARIANTS = [("D1", "local", True), ("D2", "local", True), ("D2", "knn", True), ("DF", "local", True),
            ("D2c", "local", True), ("D3", "local", True), ("D3", "local", False),
            ("D1", "knn", True), ("D3", "knn", True)]


def _mk(kind, mode, attr, **kw):
    knn = 10 if (kind in ("D1", "D3") and mode == "knn") else None
    return _drone(kind, range_mode=mode, attraction=attr, knn=knn, **kw)


@pytest.mark.parametrize("kind,mode,attr", VARIANTS)
@pytest.mark.parametrize("p_loss", [0.0, 1.0])
def test_drone_command_matches_numpy(kind, mode, attr, p_loss):
    m = _mk(kind, mode, attr, p_loss=p_loss)
    s = _state(40, 4.0, 2)
    m.start(s)
    fd = make_fast(m, DT)  # kopiert Puffer und GPS-Fehler vor dem NumPy-Aufruf
    ref = m.command(s)
    cmd = np.zeros_like(s.x)
    ex, eu = fd.own_errors()
    K.drone_command(s.x, s.u, s.v, s.t, fd.pf, fd.pi, fd.hist_x, fd.hist_u, fd.hc[0], fd.hc[1],
                    fd.age, np.random.default_rng(0), cmd, ex, eu, fd.offs)
    assert np.max(np.abs(cmd - ref)) < 1e-10


@pytest.mark.parametrize("kind,mode,attr", VARIANTS)
@pytest.mark.parametrize("nav", ["transit", "loiter"])
def test_drone_trajectory_matches_numpy(kind, mode, attr, nav):
    s0 = _state(40, 4.0, 6)
    m1 = _mk(kind, mode, attr, nav=nav)
    m2 = _mk(kind, mode, attr, nav=nav)
    m1.start(s0)
    m2.start(s0)
    a = simulate(m1, s0, dt=DT, n_steps=120, noise_rng=np.random.default_rng(11))
    b = simulate_fast(make_fast(m2, DT), s0, dt=DT, n_steps=120, noise_rng=np.random.default_rng(11))
    assert np.max(np.abs(a.x - b.x)) < 1e-8 and np.max(np.abs(a.v - b.v)) < 1e-8


def test_fast_recorder_frames_match_numpy():
    s0 = _state(30, 1.0, 8)
    frames = np.array([0.0, 0.0125, 0.1, 0.2, 0.3333])
    m1, m2 = _bird(n=30), _bird(n=30)
    m1.start(s0)
    m2.start(s0)
    r1, r2 = FrameRecorder(frames, 30), FrameRecorder(frames, 30)
    simulate(m1, s0, dt=DT, n_steps=70, noise_rng=np.random.default_rng(3), callbacks=(r1,))
    simulate_fast(make_fast(m2, DT), s0, dt=DT, n_steps=70, noise_rng=np.random.default_rng(3),
                  event_times=frames, callbacks=(r2,))
    t1, t2 = r1.trajectory(), r2.trajectory()
    assert np.max(np.abs(t1.x - t2.x)) < 1e-9 and np.max(np.abs(t1.u - t2.u)) < 1e-9



# --- Erweiterung 2026-10-11: eigener GPS-Fehler im Regler, gemeinsamer Fehleranteil -----------

OWN_VARIANTS = [("D1", "local"), ("D2", "local"), ("D2c", "local"), ("D3", "local"), ("DF", "local")]


@pytest.mark.parametrize("kind,mode", OWN_VARIANTS)
@pytest.mark.parametrize("own,shared", [(True, 0.0), (True, 0.6), (False, 0.6)])
def test_drone_command_with_own_gps_matches_numpy(kind, mode, own, shared):
    m = _drone(kind, range_mode=mode, own=own, shared=shared, p_loss=0.0)  # Verlust ohne Zufall
    s = _state(40, 4.0, 2)
    m.start(s)
    for _ in range(25):  # GPS-Fehler und Verlauf entwickeln
        m.comms.after_step(s)
    fd = make_fast(m, DT)
    ref = m.command(s)
    cmd = np.zeros_like(s.x)
    ex, eu = fd.own_errors()
    K.drone_command(s.x, s.u, s.v, s.t, fd.pf, fd.pi, fd.hist_x, fd.hist_u, fd.hc[0], fd.hc[1],
                    fd.age, np.random.default_rng(0), cmd, ex, eu, fd.offs)
    assert np.max(np.abs(cmd - ref)) < 1e-10


@pytest.mark.parametrize("kind,mode", OWN_VARIANTS)
@pytest.mark.parametrize("with_wind", [False, True])
def test_drone_trajectory_with_own_gps_matches_numpy(kind, mode, with_wind):
    from test_wind import _wind

    wind = _wind(seed=4) if with_wind else None
    s_ = _state(30, 4.0, 6)
    s0 = SwarmState.initial(0.0, s_.x, s_.v, wind.sample(s_.x, 0.0) if wind else None, 1.0)
    m1 = _drone(kind, range_mode=mode, own=True, shared=0.7, n=30)
    m2 = _drone(kind, range_mode=mode, own=True, shared=0.7, n=30)
    m1.start(s0)
    m2.start(s0)
    a = simulate(m1, s0, dt=DT, n_steps=110, noise_rng=np.random.default_rng(13), wind=wind, wind_hold_steps=2)
    b = simulate_fast(make_fast(m2, DT, wind, 2), s0, dt=DT, n_steps=110,
                      noise_rng=np.random.default_rng(13), event_times=[0.07, 0.33])
    assert np.max(np.abs(a.x - b.x)) < 1e-8 and np.max(np.abs(a.v - b.v)) < 1e-8


def test_shared_gps_error_is_common_and_own_error_moves_drones():
    m_sh = _drone("D2", own=True, shared=1.0, n=20)
    s = _state(20, 4.0, 3)
    m_sh.start(s)
    e = m_sh.comms.err_x()
    assert np.allclose(e, e[0])  # ρ = 1: alle Drohnen haben denselben Fehler
    # Eigener Fehler verschiebt die Bahnen (gleiche Zufallszahlen, nur Schalter verschieden)
    m_on, m_off = _drone("D2", own=True, n=20), _drone("D2", own=False, n=20)
    m_on.start(s)
    m_off.start(s)
    a = simulate(m_on, s, dt=DT, n_steps=400, noise_rng=np.random.default_rng(1))
    b = simulate(m_off, s, dt=DT, n_steps=400, noise_rng=np.random.default_rng(1))
    assert np.max(np.abs(a.x - b.x)) > 1e-3


def test_formation_drones_hold_offsets_and_ignore_neighbours():
    """O5, DF: ohne Rauschen bleibt die Startformation erhalten; ein Nachbar beeinflusst keinen Befehl."""
    m = _drone("DF", p_loss=0.0)
    s0 = _state(40, 4.0, 3)
    m.start(s0)
    m.sigma = 0.0
    fd = make_fast(m, DT)
    fd.pf[K.D_SIGMA] = 0.0
    end = simulate_fast(fd, s0, dt=DT, n_steps=4000, noise_rng=np.random.default_rng(5))
    off_end = end.x - end.x.mean(axis=0)
    assert np.max(np.abs(off_end - m.offsets)) < 0.05
    cmd_a = m.command(s0)
    moved = s0.replace(x=s0.x.copy())
    moved.x[0] += 5.0
    cmd_b = m.command(moved)
    assert np.allclose(cmd_a[1:], cmd_b[1:]) and not np.allclose(cmd_a[0], cmd_b[0])
