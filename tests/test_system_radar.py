"""O21 Stufe 1: Radar-Systemmodell (Radargleichung, Regendämpfung, Regenecho, Reichweite)."""

import numpy as np
import pytest

from swarmsig.system import radar as R

BASE = dict(aperture_m2=np.array([0.15]), power_w=np.array([10.0]), t_int_s=np.array([0.05]),
            nf_db=np.array([4.0]), loss_db=np.array([6.0]), bandwidth_hz=np.array([1e8]),
            clutter_supp_db=np.array([20.0]))


def rng_range(rcs, rain=0.0, f=9.5, **over):
    kw = {**BASE, **over}
    return float(R.detection_range(R.swerling1_snr(), f_ghz=f, rcs_m2=np.array([rcs]), rain_mm_h=rain, **kw)[0])


def test_swerling1_threshold():
    assert abs(10 * np.log10(R.swerling1_snr(0.9, 1e-6)) - 21.14) < 0.01


def test_range_scales_with_fourth_root_of_rcs():
    assert rng_range(0.16) / rng_range(0.01) == pytest.approx(2.0, rel=1e-6)


def test_itu_rain_attenuation_values():
    assert R.rain_attenuation_db_per_km(10.0, 9.5) == pytest.approx(0.187, rel=0.02)
    assert R.rain_attenuation_db_per_km(0.0, 9.5) == 0.0
    assert R.rain_attenuation_db_per_km(50.0, 3.0) < 0.05


def test_rain_attenuation_hurts_x_band_much_more_than_s_band():
    """Nur Dämpfung (Regenecho vollständig unterdrückt): X-Band verliert stark, S-Band kaum."""
    off = dict(clutter_supp_db=np.array([300.0]))
    x0, x50 = rng_range(0.02, 0.0, 9.5, **off), rng_range(0.02, 50.0, 9.5, **off)
    s0, s50 = rng_range(0.02, 0.0, 3.0, **off), rng_range(0.02, 50.0, 3.0, **off)
    assert x50 < 0.6 * x0
    assert s50 > 0.95 * s0


def test_rain_clutter_limits_wide_beams():
    """Regenecho: schwächere Unterdrückung -> kürzere Reichweite; bei breiter S-Band-Keule wirkt es deutlich."""
    s_weak = rng_range(0.02, 50.0, 3.0, clutter_supp_db=np.array([10.0]))
    s_strong = rng_range(0.02, 50.0, 3.0, clutter_supp_db=np.array([40.0]))
    assert s_weak < 0.5 * s_strong


def test_strong_clutter_suppression_gives_noise_limited_range():
    a = rng_range(0.02, 25.0, 9.5, clutter_supp_db=np.array([200.0]))
    lam, rcs = R.wavelength(9.5), 0.02
    # rein rauschbegrenzt, aber mit Dämpfung: SNR(a) muss die Schwelle treffen
    kw = {**BASE, "clutter_supp_db": np.array([200.0])}
    s = R.scnr(np.array([a]), f_ghz=9.5, rcs_m2=np.array([rcs]), rain_mm_h=25.0, **kw)[0]
    assert s == pytest.approx(R.swerling1_snr(), rel=1e-6)
    assert lam > 0


def test_plausible_clear_air_drone_range():
    r = rng_range(0.017)
    assert 1_000 < r < 15_000
