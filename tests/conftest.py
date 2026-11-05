from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swarmsig.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def base_cfg() -> Config:
    return load_config(ROOT / "configs" / "base.yaml")


@pytest.fixture(scope="session")
def smoke_cfg() -> Config:
    return load_config(ROOT / "configs" / "smoke.yaml")


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)
