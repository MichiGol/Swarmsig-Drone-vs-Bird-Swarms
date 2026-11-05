"""13.1 wind_transfer: c = 0 -> u = v + w exakt; c = 1, konstanter Wind -> u -> v mit
Zeitkonstante τ_est; Sinus-Wind mit Periode ≪ τ_est wirkt voll durch."""

import numpy as np

from swarmsig.dynamics.wind_transfer import WindTransfer
from swarmsig.environment.wind_profile import ConstantWind, PowerLawWind, SineWind


def _run(transfer, wind, v, dt, n_steps):
    x = np.zeros_like(v)
    w_hat = wind.sample(x, 0.0) * 0.0
    out = []
    for k in range(n_steps):
        u, w_hat = transfer.apply(v, wind.sample(x, k * dt), wind.sample(x, (k + 1) * dt), w_hat, dt)
        out.append(u.copy())
    return np.array(out)


def test_c0_full_drift():
    v = np.array([[10.0, 0.0, 0.0], [0.0, 9.0, 1.0]])
    w = ConstantWind((3.0, -2.0, 0.5))
    u = _run(WindTransfer(0.0, 5.0), w, v, 0.01, 10)
    assert np.allclose(u, v + np.array([3.0, -2.0, 0.5]), atol=0, rtol=0)


def test_c1_constant_wind_relaxes_with_tau():
    v = np.array([[10.0, 0.0, 0.0]])
    tau, dt = 2.0, 0.001
    u = _run(WindTransfer(1.0, tau), ConstantWind((4.0, 0.0, 0.0)), v, dt, int(round(tau / dt)))
    # nach t = τ ist der unkompensierte Anteil auf e^{-1} abgefallen (Diskretisierung ~Δt/τ)
    assert abs((u[-1, 0, 0] - 10.0) / 4.0 - np.exp(-1)) < 2e-3


def test_fast_sine_passes_through():
    v = np.array([[0.0, 0.0, 0.0]])
    period, tau, dt = 0.2, 10.0, 0.001
    u = _run(WindTransfer(1.0, tau), SineWind((2.0, 0.0, 0.0), period), v, dt, 2000)
    amp = 0.5 * (u[-400:, 0, 0].max() - u[-400:, 0, 0].min())
    assert abs(amp - 2.0) / 2.0 < 0.05


def test_power_law_profile():
    w = PowerLawWind(U10=5.0, alpha=0.2, direction_rad=np.pi / 2)  # weht nach Osten
    s = w.sample(np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 100.0]]), 0.0)
    assert np.allclose(s[0], [5.0, 0.0, 0.0])
    assert np.allclose(s[1, 0], 5.0 * 10**0.2)
