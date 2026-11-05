import numpy as np
import pytest

from swarmsig.config import (Choice, LogUniform, TruncWeibull, Uniform, config_hash, load_config,
                             sample)
from conftest import ROOT


def test_configs_load(base_cfg):
    p1 = load_config(ROOT / "configs" / "p1_ideal.yaml")
    assert p1.experiment.id == "P1_ideal_v1"
    assert p1.drones.D3.r0_fric_m == base_cfg.drones.D3.r0_fric_m  # aus base geerbt
    assert config_hash(p1) == config_hash(load_config(ROOT / "configs" / "p1_ideal.yaml"))
    assert config_hash(p1) != config_hash(base_cfg)


def test_with_overrides_revalidates(base_cfg):
    c2 = base_cfg.with_overrides({"dynamics": {"dt_s": 0.002}})
    assert c2.dynamics.dt_s == 0.002 and base_cfg.dynamics.dt_s == 0.005
    with pytest.raises(Exception):
        base_cfg.with_overrides({"dynamics": {"unbekannt": 1}})


def test_sampling_ranges(rng):
    u = [sample(Uniform(dist="uniform", low=2, high=3), rng) for _ in range(500)]
    assert 2 <= min(u) and max(u) <= 3
    lu = [sample(LogUniform(dist="loguniform", low=0.02, high=0.5), rng) for _ in range(500)]
    assert 0.02 <= min(lu) and max(lu) <= 0.5
    w = [sample(TruncWeibull(dist="weibull", shape=2, scale=4.5, low=0, high=15), rng) for _ in range(2000)]
    assert 0 <= min(w) and max(w) <= 15
    assert abs(np.mean(w) - 4.5 * 0.886) < 0.15  # Weibull(k=2)-Mittel ≈ 0,886 λ
    c = [sample(Choice(dist="choice", values=["a", "b"], weights=[0.9, 0.1]), rng) for _ in range(2000)]
    assert 0.85 < c.count("a") / 2000 < 0.95
    assert sample(4.0, rng) == 4.0


def test_invalid_distribution_rejected():
    with pytest.raises(Exception):
        Uniform(dist="uniform", low=3, high=2)
    with pytest.raises(Exception):
        LogUniform(dist="loguniform", low=0, high=1)
