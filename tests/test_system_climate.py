"""O21 Stufe 3: DWD-Leser und Sonnenhöhe."""

import zipfile

import numpy as np
import pandas as pd
import pytest

from swarmsig.system import climate as C

LAT, LON = 52.4644, 9.6779  # Hannover (02014)


def _day(date: str) -> np.ndarray:
    t = pd.date_range(f"{date} 00:00", f"{date} 23:59", freq="1min")
    return C.solar_elevation_deg(t, LAT, LON)


@pytest.mark.parametrize(("date", "expected"), [("2023-06-21", 90 - LAT + 23.436), ("2023-12-22", 90 - LAT - 23.436),
                                                ("2023-03-21", 90 - LAT + 0.23)])
def test_solar_noon_elevation(date, expected):
    """Mittagshöhe = 90° − Breite + Deklination (Solstitien ±23,436°; 21.3.2023 etwa 14 h nach dem Äquinoktium)."""
    assert _day(date).max() == pytest.approx(expected, abs=0.1)


def test_midnight_and_day_length():
    e = _day("2023-06-21")
    assert e.min() == pytest.approx(-(90 - LAT - 23.436), abs=0.1)
    h0 = np.degrees(np.arccos(-np.tan(np.radians(LAT)) * np.tan(np.radians(23.436))))
    assert np.mean(e > 0.0) == pytest.approx(2 * h0 / 15.0 / 24.0, abs=0.01)  # ≈ 16,6 h Tag


def test_accepts_series():
    t = pd.Series(pd.date_range("2023-06-21 11:00", periods=3, freq="10min"))
    np.testing.assert_allclose(C.solar_elevation_deg(t, LAT, LON), C.solar_elevation_deg(pd.DatetimeIndex(t), LAT, LON))


def test_nearest_log_index():
    """Grenzen liegen bei den geometrischen Mitteln √10 ≈ 3,16 und √1000 ≈ 31,6."""
    idx = C.nearest_log_index(np.array([0.5, 3.0, 3.3, 50.0, 1000.0]), np.array([1.0, 10.0, 100.0]))
    np.testing.assert_array_equal(idx, [0, 0, 1, 2, 2])


def _zip(path, name, text):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("Metadaten_Geographie_00691.txt", "nicht lesen")
        zf.writestr(name, text)
    return path


def test_read_10min_product(tmp_path):
    z = _zip(tmp_path / "rr.zip", "produkt_zehn_min_rr_20200101_20251231_00691.txt",
             "STATIONS_ID;MESS_DATUM;  QN;RWS_DAU_10;RWS_10;RWS_IND_10;eor\n"
             "        691;202001010000;    3;   0;   0.00;   0;eor\n"
             "        691;202001010010;    3;  10;   0.35;   1;eor\n"
             "        691;202001010020;    3;-999;-999;-999;eor\n")
    df = C.read_dwd_product(z)
    assert list(df.columns) == ["station_id", "QN", "RWS_DAU_10", "RWS_10", "RWS_IND_10", "time"]
    assert list(df["time"]) == list(pd.date_range("2020-01-01 00:00", periods=3, freq="10min"))
    np.testing.assert_array_equal(df["RWS_10"].to_numpy(), [0.0, 0.35, np.nan])
    assert df["station_id"].eq(691).all()


def test_read_hourly_product(tmp_path):
    z = _zip(tmp_path / "vv.zip", "produkt_vv_stunde_19490101_20251231_00691.txt",
             "STATIONS_ID;MESS_DATUM;QN_8;V_VV_I;V_VV;eor\n"
             "        691;2025123121;    3;   I; 24000;eor\n"
             "        691;2025123122;    3;   P;  -999;eor\n")
    df = C.read_dwd_product(z)
    assert list(df["time"]) == [pd.Timestamp("2025-12-31 21:00"), pd.Timestamp("2025-12-31 22:00")]
    assert list(df["V_VV_I"]) == ["I", "P"]
    np.testing.assert_array_equal(df["V_VV"].to_numpy(dtype=float), [24000.0, np.nan])


def test_parse_station_list(tmp_path):
    rows = ["Stations_id von_datum bis_datum Stationshoehe geoBreite geoLaenge Stationsname Bundesland Abgabe",
            "----------- --------- --------- ------------- --------- --------- ------------------------------"
            "----------- ---------- ------",
            "01975 19950828 20261016             11     53.6332    9.9881 Hamburg-Fuhlsbüttel                "
            "      Hamburg                                  Frei      ",
            "02014 19920429 20261016             55     52.4644    9.6779 Hannover                           "
            "      Niedersachsen                            Frei      "]
    p = tmp_path / "stations.txt"
    p.write_bytes("\n".join(rows).encode("latin1"))
    st = C.parse_station_list(p)
    assert st.loc[1975, "station"] == "Hamburg-Fuhlsbüttel"
    assert st.loc[2014, ["lat", "lon", "height_m"]].tolist() == [52.4644, 9.6779, 55.0]
    assert st.loc[2014, "state"] == "Niedersachsen"
