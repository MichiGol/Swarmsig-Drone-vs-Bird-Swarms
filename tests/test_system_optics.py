"""O21 Stufe 2: Optikmodell (Johnson-Pixel, Koschmieder-Sicht, Stereo) und Mikro-Doppler-Reichweite."""

import numpy as np
import pytest

from swarmsig.system import optics as O
from swarmsig.system import radar as R


def test_pixel_range():
    assert O.pixel_range(0.35, 8, 3.45e-6, 0.3) == pytest.approx(0.35 * 0.3 / (8 * 3.45e-6))


def test_visibility_range_equals_visibility_at_two_percent_contrast():
    assert O.visibility_range(10_000.0, np.exp(3.912) * 0.02, 0.02) == pytest.approx(10_000.0, rel=1e-9)
    assert O.visibility_range(10_000.0, 0.01, 0.02) == 0.0  # Eigenkontrast unter der Schwelle


def test_stereo_range():
    assert O.stereo_range(0.05, 4000.0, 20.0, 0.5) == pytest.approx(np.sqrt(8000.0))
    # Tiefenfehler an der Grenze: δZ = Z²·δd/(f·B) = σ_max
    z = O.stereo_range(0.05, 4000.0, 20.0, 0.5)
    assert z**2 * 0.5 / (4000.0 * 20.0) == pytest.approx(0.05)


KW = dict(f_ghz=9.5, aperture_m2=np.array([0.15]), power_w=np.array([10.0]), t_int_s=np.array([0.05]),
          nf_db=np.array([4.0]), loss_db=np.array([6.0]), bandwidth_hz=np.array([1e8]),
          clutter_supp_db=np.array([20.0]), rcs_m2=np.array([0.02]))


def test_micro_doppler_shorter_than_detection_and_ignores_clutter():
    det = R.detection_range(R.swerling1_snr(), rain_mm_h=0.0, **KW)[0]
    md = R.micro_doppler_range(np.array([-15.0]), np.array([12.0]), rain_mm_h=0.0, **KW)[0]
    assert md < det
    # Δ_b = −15 dB, SNR_mD = 12 dB statt 21,1 dB: Faktor (10^(−1,5)·10^(0,911))^(1/4)
    assert md / det == pytest.approx((10 ** -1.5 * 10 ** (2.114 - 1.2)) ** 0.25, rel=1e-3)
    md_rain = R.micro_doppler_range(np.array([-15.0]), np.array([12.0]), rain_mm_h=10.0,
                                    **{**KW, "clutter_supp_db": np.array([0.0])})[0]
    md_rain_ref = R.micro_doppler_range(np.array([-15.0]), np.array([12.0]), rain_mm_h=10.0, **KW)[0]
    assert md_rain == pytest.approx(md_rain_ref, rel=1e-9)  # Regenecho spielt keine Rolle
    assert md_rain < md  # Dämpfung schon
