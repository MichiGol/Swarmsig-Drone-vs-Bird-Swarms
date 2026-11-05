"""Integrationstests (13.2), bewusst winzig (N = 30, wenige Sekunden Simulationszeit):
Determinismus, Stratifizierung, Schema, unabhängige Realisierungen vs. CRN."""

import numpy as np
import pytest

from swarmsig.experiment.design import allocate, draw_design
from swarmsig.experiment.run import Provenance, run_class, run_design_point
from swarmsig.io import read_runs, required_columns, write_runs

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture(scope="module")
def tiny(smoke_cfg):
    design = draw_design(smoke_cfg)
    return smoke_cfg, design, Provenance.of(smoke_cfg)


def test_ordered_design_largest_first(smoke_cfg):
    from swarmsig.experiment.runner import ordered_design

    cfg = smoke_cfg.with_overrides({"scenario": {"N_plan": {10: 2, 20: 3, 30: 2}},
                                    "experiment": {"bird_classes": {"XS": ["B1", "B3"]}}})
    plain = ordered_design(cfg)
    big = ordered_design(cfg, largest_first=True)
    assert [t.N for t, _ in plain] == [10, 10, 20, 20, 20, 30, 30]
    assert [t.N for t, _ in big] == [30, 30, 20, 20, 20, 10, 10]
    assert {t.index: t for t, _ in plain} == {t.index: t for t, _ in big}  # gleiche Punkte und Seeds
    only = ordered_design(cfg.with_overrides({"experiment": {"only_N": [20]}}), largest_first=True)
    assert [t.index for t, _ in only] == [2, 3, 4]


def test_allocate_stratified(rng):
    out = allocate(["S", "M", "L"], [0.6, 0.3, 0.1], 100, rng)
    assert (out.count("S"), out.count("M"), out.count("L")) == (60, 30, 10)


def test_design_is_reproducible(smoke_cfg):
    a = [t for t, _ in draw_design(smoke_cfg)]
    b = [t for t, _ in draw_design(smoke_cfg)]
    assert a == b
    assert {t.bird_model for t in a} == {"B1", "B3"}


def test_run_is_deterministic(tiny):
    cfg, design, prov = tiny
    theta, ps = design[0]
    r1, _ = run_class("D3", theta, ps, cfg, prov)
    r2, _ = run_class("D3", theta, ps, cfg, prov)
    for k, v in r1.items():
        if k == "wall_time_s":
            continue
        if isinstance(v, float) and np.isnan(v):
            assert np.isnan(r2[k])
        else:
            assert r2[k] == v, k


def test_design_point_rows_and_schema(tiny, tmp_path):
    cfg, design, prov = tiny
    theta, ps = design[1]
    rows = run_design_point(theta, ps, cfg, prov, drone_classes=["D1", "D2c"])
    assert [r["class"] for r in rows] == [theta.bird_model, "D1", "D2c"]
    df = write_runs(rows, tmp_path / "runs.parquet")
    assert set(required_columns()) <= set(df.columns)
    back = read_runs(tmp_path / "runs.parquet")
    assert len(back) == 3 and back["design_point_id"].nunique() == 1


def test_classes_get_independent_noise(tiny):
    cfg, design, prov = tiny
    theta, ps = design[0]
    a = ps.for_class("D1").rng("agent", "process_noise").random(3)
    b = ps.for_class("D2").rng("agent", "process_noise").random(3)
    assert not np.array_equal(a, b)


def test_k_density_control_reuses_seeds(tiny):
    cfg, design, prov = tiny
    theta, ps = design[1]
    rows = run_design_point(theta, ps, cfg, prov, drone_classes=["D1"], controls=["K0", "K-Dichte"])
    k0, kd = rows[1], rows[2]
    assert kd["control"] == "K-Dichte"
    assert kd["d_spacing"] != k0["d_spacing"]
    assert kd["d_tau_d"] == k0["d_tau_d"]  # gleiche Seeds: übrige Parameter identisch (CRN)
    assert 1.0 <= kd["d_spacing"] / rows[0]["nn_dist"] <= 1.5



def test_sensor_variants_leave_base_row_unchanged(tiny):
    """O2: Varianten werden auf derselben Flugbahn ausgewertet; die Grundzeile bleibt bit-gleich,
    und eine Variante ohne Rauschen mit gleichem q reproduziert die Grundzeile."""
    cfg, design, prov = tiny
    theta, ps = design[0]
    base, _ = run_class("D1", theta, ps, cfg, prov)
    cfg_v = cfg.with_overrides({"sensor": {"variants": [
        {"label": "ref", "sigma_pos_m": 0.0, "q_frames": 1},
        {"label": "s05q5", "sigma_pos_m": 0.05, "q_frames": 5}]}})
    r, _ = run_class("D1", theta, ps, cfg_v, prov)
    variants = r.pop("_sensor_variants")
    assert [v["control"] for v in variants] == ["K0|S:ref", "K0|S:s05q5"]
    keys = [k for k, v in base.items() if isinstance(v, float) and k != "wall_time_s"]
    for k in keys:
        a, b, c = base[k], r[k], variants[0][k]
        if np.isnan(a):
            assert np.isnan(b) and np.isnan(c), k
        else:
            assert a == b == c, k
    assert variants[1]["sigma_pos_m"] == 0.05 and variants[1]["q_frames"] == 5
    assert variants[1]["T_nn_A7"] != base["T_nn_A7"]
