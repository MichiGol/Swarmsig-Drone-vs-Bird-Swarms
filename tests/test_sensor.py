"""Interpolation auf Frames (4.9) und idealer Sensor (6.1)."""

import numpy as np

from swarmsig.sensors.ideal import IdealSensor
from swarmsig.sensors.interpolate import FrameRecorder, hermite
from swarmsig.state import SwarmState, Trajectory


def test_hermite_exact_for_cubics():
    c = np.array([1.0, -2.0, 0.5, 0.3])
    x = lambda t: c[0] + c[1] * t + c[2] * t**2 + c[3] * t**3  # noqa: E731
    v = lambda t: c[1] + 2 * c[2] * t + 3 * c[3] * t**2  # noqa: E731
    t0, t1 = 0.2, 0.7
    for t in (0.2, 0.31, 0.55, 0.7):
        xi, ui = hermite(t0, np.array([x(t0)]), np.array([v(t0)]), t1, np.array([x(t1)]),
                         np.array([v(t1)]), t)
        assert abs(xi[0] - x(t)) < 1e-12 and abs(ui[0] - v(t)) < 1e-12


def test_recorder_on_and_off_grid():
    dt = 0.005
    times = np.array([0.0, 0.0125, 0.02, 0.0333])
    rec = FrameRecorder(times, 1)
    s = SwarmState.initial(0.0, np.zeros((1, 3)), np.array([[2.0, 0.0, 0.0]]))
    for n in range(10):
        new = s.replace(t=(n + 1) * dt, x=s.x + s.u * dt)
        rec(s, new, n)
        s = new
    tr = rec.trajectory()
    assert np.allclose(tr.x[:, 0, 0], 2.0 * times)
    assert np.allclose(tr.u[:, 0, 0], 2.0)


def test_ideal_sensor_central_difference(rng):
    f = 10.0
    sensor = IdealSensor(sigma_pos=0.0, f_cam=f, q=1, p_miss=0.0)
    t = sensor.frame_times(60.0, 1.0)
    assert t.size == 11 + 2 and abs(t[0] - 59.9) < 1e-12
    x = np.stack([np.column_stack([3.0 * tt + np.arange(4), np.zeros(4), np.zeros(4)]) for tt in t])
    obs = sensor.observe(Trajectory(t, x, np.zeros_like(x)), rng)
    assert obs.n_frames == 11 and obs.valid.all()
    assert np.allclose(obs.u[..., 0], 3.0)


def test_ideal_sensor_noise_and_misses(rng):
    sensor = IdealSensor(sigma_pos=0.2, f_cam=10.0, q=1, p_miss=0.3)
    t = sensor.frame_times(0.0, 20.0)
    x = np.zeros((t.size, 500, 3))
    obs = sensor.observe(Trajectory(t, x, np.zeros_like(x)), rng)
    assert abs(obs.valid.mean() - 0.7**3) < 0.02
    expected = 0.2 / (np.sqrt(2) * 1 * 0.1)  # σ_pos/(√2 q δ)
    assert abs(np.std(obs.u[..., 0]) - expected) / expected < 0.05
