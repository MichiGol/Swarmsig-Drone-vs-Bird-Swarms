"""13.1 Wind (P3): von-Kármán-Spektrum, MIL-F-8785C, Kraichnan-Synthese (Varianz, Divergenz,
Strukturfunktion), Numba gegen NumPy mit Wind, Wetterziehung und Kontrollen K-Wind/K-Calm."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from swarmsig.config import EnvironmentCfg
from swarmsig.dynamics.fast import make_fast, simulate_fast
from swarmsig.dynamics.integrator import simulate
from swarmsig.environment.spectra import (VK_C, VK_L11_OVER_L, TurbulenceParams, gust_scaled,
                                          mil_low_altitude, von_karman_E, von_karman_energy,
                                          vk_length_for_integral_scale)
from swarmsig.environment.weather import build_wind, draw_weather, load_weather_table
from swarmsig.environment.wind_kraichnan import TurbulentWind, draw_modes, turbulence_numpy
from swarmsig.environment.wind_profile import PowerLawWind
from swarmsig.rng import name_key
from swarmsig.state import SwarmState

from test_kernels import DT, _bird, _mk, _state


# --- Spektrum und MIL ----------------------------------------------------------------------

def test_von_karman_normalisation_and_integral_scale():
    assert abs(VK_C - 1.4528) < 1e-4  # Referenzwert aus 5.3.2
    sigma, L = 1.3, 80.0
    tot, _ = quad(lambda k: von_karman_E(k, sigma, L), 0, np.inf, limit=400)
    assert abs(tot - 1.5 * sigma**2) < 1e-6
    assert abs(von_karman_energy(0.0, np.inf, sigma, L) - 1.5 * sigma**2) < 1e-12
    part, _ = quad(lambda k: von_karman_E(k, sigma, L), 0.01, 0.2, limit=400)
    assert abs(von_karman_energy(0.01, 0.2, sigma, L) - part) < 1e-8
    # L11 = π/(2σ²) ∫ E(k)/k dk
    i, _ = quad(lambda k: von_karman_E(k, sigma, L) / k, 0, np.inf, limit=400)
    assert abs(math.pi / (2 * sigma**2) * i / L - VK_L11_OVER_L) < 1e-6
    assert abs(VK_L11_OVER_L - 0.7468) < 1e-4


def test_mil_low_altitude_example():
    tp = mil_low_altitude(100.0, 5.0)  # Beispiel aus 5.3.1: L_w ≈ 100 m, L_u ≈ 263 m
    assert abs(tp.L_w - 100.0) < 1e-9 and abs(tp.L_u - 263.0) < 1.0
    assert abs(tp.sigma_w - 0.5) < 1e-12 and tp.sigma_u == tp.sigma_v > tp.sigma_w


def test_gust_scaled_matches_measured_intensity_at_10m():
    tp = gust_scaled(10.0, 6.0, 0.15)
    assert abs(tp.sigma_u - 0.9) < 1e-12
    hi = gust_scaled(300.0, 6.0, 0.15)
    assert hi.sigma_u < tp.sigma_u  # MIL: Intensität nimmt mit der Höhe ab


# --- Kraichnan-Synthese ----------------------------------------------------------------------

def _iso(sigma=1.0, L_u=120.0):
    return TurbulenceParams(sigma, sigma, sigma, L_u, L_u, L_u)


def test_modes_are_divergence_free_when_isotropic():
    m = draw_modes(_iso(), 0.7, np.random.default_rng(1))
    assert np.max(np.abs(np.sum(m.k * m.a, axis=1))) < 1e-12
    assert np.max(np.abs(np.sum(m.k * m.b, axis=1))) < 1e-12
    assert 0.97 < m.resolved_fraction < 1.0


def _theory_D_LL(r, sigma, L_u, k_lo, k_hi):
    """Longitudinale Strukturfunktion des abgeschnittenen von-Kármán-Spektrums."""
    L = vk_length_for_integral_scale(L_u)

    def g(k):
        x = k * r
        kern = (np.sin(x) - x * np.cos(x)) / x**3 if x > 1e-3 else 1.0 / 3.0 - x**2 / 30.0
        return von_karman_E(k, sigma, L) * (1.0 / 3.0 - kern)

    val, _ = quad(g, k_lo, k_hi, limit=2000)
    return 4.0 * val  # D_LL = 2(R(0) − R(r)), R(r) = 2∫E(k)·kern dk


def test_variance_and_structure_function_ensemble():
    sigma, L_u, l_min = 1.0, 120.0, 1.0
    L = vk_length_for_integral_scale(L_u)
    k_lo, k_hi = 0.05 / L, 2 * math.pi / l_min
    rs = np.array([3.0, 15.0, 60.0])
    # Punktpaare weit auseinander (≫ L), damit sie annähernd unabhängig sind
    base = np.array([[5000.0 * q, -3000.0 * q, 200.0] for q in range(6)])
    var, dll = [], {r: [] for r in rs}
    for s in range(500):
        m = draw_modes(_iso(sigma, L_u), 0.3, np.random.default_rng(1000 + s))
        w0 = turbulence_numpy(m, base, 0.0, 0.0, 0.3)
        var.append(w0**2)
        for r in rs:
            w1 = turbulence_numpy(m, base + np.array([r, 0.0, 0.0]), 0.0, 0.0, 0.3)
            dll[r].append((w1[:, 0] - w0[:, 0]) ** 2)
    res = von_karman_energy(k_lo, k_hi, sigma, L) / (1.5 * sigma**2)
    v = np.mean(np.concatenate(var), axis=0)
    assert np.all(np.abs(v / (sigma**2 * res) - 1.0) < 0.08), v
    for r in rs:
        est = np.mean(np.concatenate(dll[r]))
        th = _theory_D_LL(r, sigma, L_u, k_lo, k_hi)
        assert abs(est / th - 1.0) < 0.12, (r, est, th)


def test_anisotropic_scaling_in_wind_frame():
    tp = TurbulenceParams(1.0, 0.5, 0.25, 100.0, 100.0, 100.0)
    direction = math.radians(90.0)  # Wind weht nach Osten: längs = x
    pts = np.array([[4000.0 * q, 0.0, 150.0] for q in range(8)])
    acc = []
    for s in range(300):
        m = draw_modes(tp, direction, np.random.default_rng(s))
        acc.append(turbulence_numpy(m, pts, 0.0, 0.0, direction) ** 2)
    v = np.mean(np.concatenate(acc), axis=0) / 0.98
    assert abs(v[0] - 1.0) < 0.1 and abs(v[1] - 0.25) < 0.03 and abs(v[2] - 0.0625) < 0.01


def test_numba_wind_eval_matches_numpy_and_is_frozen():
    m = draw_modes(_iso(0.8, 200.0), 1.1, np.random.default_rng(3))
    w = TurbulentWind(PowerLawWind(5.0, 0.2, 1.1), m, 7.0)
    x = np.random.default_rng(4).uniform(-500, 500, (40, 3)) + np.array([0.0, 0.0, 600.0])
    ref = PowerLawWind(5.0, 0.2, 1.1).sample(x, 0.0) + turbulence_numpy(m, x, 2.5, 7.0, 1.1)
    assert np.max(np.abs(w.sample(x, 2.5) - ref)) < 1e-10
    # eingefroren (Taylor): w(x + U_c ê t, t) − Profil = w'(x, 0)
    e = np.array([math.sin(1.1), math.cos(1.1), 0.0])
    a = w.sample(x + 7.0 * 3.0 * e, 3.0) - PowerLawWind(5.0, 0.2, 1.1).sample(x, 0.0)
    b = w.sample(x, 0.0) - PowerLawWind(5.0, 0.2, 1.1).sample(x, 0.0)
    assert np.max(np.abs(a - b)) < 1e-9


# --- Kerne mit Wind gegen NumPy --------------------------------------------------------------

def _wind(seed=5, sigma=0.9):
    m = draw_modes(_iso(sigma, 150.0), 0.4, np.random.default_rng(seed))
    return TurbulentWind(PowerLawWind(6.0, 0.25, 0.4), m, 6.0 * (100 / 10) ** 0.25)


@pytest.mark.parametrize("hold", [1, 3])
def test_bird_trajectory_with_wind_matches_numpy(hold):
    wind = _wind()
    s_ = _state(40, 1.0, 4)
    s0 = SwarmState.initial(0.0, s_.x, s_.v, wind.sample(s_.x, 0.0), 0.3)
    m1 = _bird("B3", "topological", "marginal", n=40, n_coh=20)
    m2 = _bird("B3", "topological", "marginal", n=40, n_coh=20)
    for m in (m1, m2):
        from swarmsig.dynamics.wind_transfer import WindTransfer
        m.transfer = WindTransfer(0.3, 4.0)
        m.start(s0)
    a = simulate(m1, s0, dt=DT, n_steps=90, noise_rng=np.random.default_rng(9), wind=wind,
                 wind_hold_steps=hold)
    b = simulate_fast(make_fast(m2, DT, wind, hold), s0, dt=DT, n_steps=90,
                      noise_rng=np.random.default_rng(9), event_times=[0.1, 0.2337])
    for f in ("x", "v", "u", "w_hat"):
        assert np.max(np.abs(getattr(a, f) - getattr(b, f))) < 1e-9, f
    assert np.max(np.abs(a.u - a.v)) > 0.1  # Wind wirkt tatsächlich


@pytest.mark.parametrize("kind,mode", [("D1", "local"), ("D2", "local"), ("D3", "local")])
def test_drone_trajectory_with_wind_matches_numpy(kind, mode):
    wind = _wind(seed=8)
    s_ = _state(30, 4.0, 6)
    s0 = SwarmState.initial(0.0, s_.x, s_.v, wind.sample(s_.x, 0.0), 1.0)
    m1, m2 = _mk(kind, mode, True, n=30), _mk(kind, mode, True, n=30)
    m1.start(s0)
    m2.start(s0)
    a = simulate(m1, s0, dt=DT, n_steps=100, noise_rng=np.random.default_rng(11), wind=wind,
                 wind_hold_steps=2)
    b = simulate_fast(make_fast(m2, DT, wind, 2), s0, dt=DT, n_steps=100,
                      noise_rng=np.random.default_rng(11), event_times=[0.05, 0.31])
    for f in ("x", "v", "u", "w_hat"):
        assert np.max(np.abs(getattr(a, f) - getattr(b, f))) < 1e-8, f


def test_no_wind_kernel_path_unchanged():
    """Ohne Wind ist u = v (P1-Verhalten)."""
    s0 = _state(20, 4.0, 7)
    m = _mk("D3", "local", True, n=20)
    m.start(s0)
    b = simulate_fast(make_fast(m, DT), s0, dt=DT, n_steps=40, noise_rng=np.random.default_rng(2))
    assert np.array_equal(b.u, b.v)


# --- Wetter ----------------------------------------------------------------------------------

@pytest.fixture()
def weather_table(tmp_path):
    df = pd.DataFrame({"station": ["A", "B", "C"], "time_utc": pd.to_datetime(["2020-10-01 10:00"] * 3),
                       "U10": [4.0, 0.0, 9.0], "dir_from_deg": [270.0, np.nan, 0.0],
                       "I_u10": [0.2, np.nan, 0.1]})
    p = tmp_path / "w.parquet"
    df.to_parquet(p)
    load_weather_table.cache_clear()
    return str(p)


def test_draw_weather_from_table(weather_table):
    env = EnvironmentCfg(wind="turbulent", weather_table=weather_table)
    seen = {}
    for s in range(60):
        rng_for = lambda name, s=s: np.random.default_rng([s, name_key(name)])  # noqa: E731
        ws = draw_weather(env, rng_for)
        seen[ws.station] = ws
        assert 0.1 <= ws.alpha <= 0.4 and ws.source == "dwd"
    assert set(seen) == {"A", "B", "C"}
    assert abs(seen["A"].wind_dir_rad - math.radians(90.0)) < 1e-12  # aus West → weht nach Ost
    assert abs(seen["C"].wind_dir_rad - math.pi) < 1e-12  # aus Nord → weht nach Süd
    assert seen["B"].I_u10 == env.I_u10_default and 0 <= seen["B"].wind_dir_rad < 2 * math.pi


def test_build_wind_hold_and_mean_only(weather_table):
    env = EnvironmentCfg(wind="turbulent", weather_table=weather_table)
    ws = draw_weather(env, lambda name: np.random.default_rng(name_key(name)))
    ws_ = build_wind(env, ws, 120.0, 0.001, np.random.default_rng(1))
    assert ws_.hold_steps == 5 and ws_.wind.modes.n_modes > 0
    assert build_wind(env, ws, 120.0, 0.0025, np.random.default_rng(1)).hold_steps == 2
    assert build_wind(env, ws, 120.0, 0.01, np.random.default_rng(1)).hold_steps == 1
    mean = build_wind(EnvironmentCfg(wind="mean", weather_table=weather_table), ws, 120.0, 0.005,
                      np.random.default_rng(1))
    assert mean.wind.modes.n_modes == 0


# --- Integration: Designpunkt mit Wind und Kontrollen ------------------------------------------

def test_design_point_with_wind_and_controls(smoke_cfg, weather_table):
    from swarmsig.experiment.design import draw_design
    from swarmsig.experiment.run import Provenance, run_design_point

    cfg = smoke_cfg.with_overrides({
        "environment": {"wind": "turbulent", "weather_table": weather_table},
        "experiment": {"controls": ["K0", "K-Wind", "K-Calm"]},
        "scenario": {"N_bands": {"S": {"values": [12], "weight": 1.0}}},
    })
    theta, ps = draw_design(cfg)[0]
    assert theta.weather.source == "dwd"
    rows = run_design_point(theta, ps, cfg, Provenance.of(cfg), drone_classes=["D3"])
    by = {(r["class"], r["control"]): r for r in rows}
    bird = theta.bird_model
    assert set(by) == {(bird, "K0"), ("D3", "K0"), ("D3", "K-Wind"), (bird, "K-Calm"), ("D3", "K-Calm")}
    assert by[(bird, "K0")]["wind_on"] and by[("D3", "K0")]["U10"] == theta.weather.U10
    assert by[("D3", "K0")]["c_comp"] == 1.0
    assert by[("D3", "K-Wind")]["c_comp"] == by[(bird, "K0")]["b_c_comp"]
    assert by[("D3", "K-Wind")]["tau_est"] == by[(bird, "K0")]["b_tau_est"]
    assert not by[("D3", "K-Calm")]["wind_on"] and np.isnan(by[("D3", "K-Calm")]["U10"])


def test_skip_birds_with_calm_control_runs_drones_only(smoke_cfg, weather_table):
    from swarmsig.experiment.design import draw_design
    from swarmsig.experiment.run import Provenance, run_design_point

    cfg = smoke_cfg.with_overrides({
        "environment": {"wind": "turbulent", "weather_table": weather_table},
        "experiment": {"controls": ["K0", "K-Calm"], "skip_birds": True},
        "scenario": {"N_bands": {"S": {"values": [12], "weight": 1.0}}},
    })
    theta, ps = draw_design(cfg)[0]
    rows = run_design_point(theta, ps, cfg, Provenance.of(cfg), drone_classes=["D1"])
    assert [(r["class"], r["control"], r["wind_on"]) for r in rows] == [("D1", "K0", True), ("D1", "K-Calm", False)]
