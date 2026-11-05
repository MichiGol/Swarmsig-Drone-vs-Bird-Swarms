"""Ausführung vieler Designpunkte mit Checkpointing und Zeitlimit (8.7).

Jeder Designpunkt wird als eigene Parquet-Datei ``dp_000123.parquet`` gespeichert; bereits
vorhandene Designpunkte werden übersprungen (Wiederaufnahme nach Abbruch). Designpunkte
sind unabhängig und laufen optional mit ``joblib`` parallel, in Index-Reihenfolge (beim
festen Plan: kleine Gruppen zuerst) oder mit ``largest_first`` nach absteigendem N (Indizes
und Seeds bleiben gleich). Mit ``time_limit_s`` startet nach Ablauf kein neuer Designpunkt
mehr; laufende werden noch fertig gerechnet.
"""

from __future__ import annotations

import time
from pathlib import Path

from joblib import Parallel, delayed

from ..config import Config, dump_config
from ..io import write_runs
from ..observables.smalln import NNBiasTable
from ..rng import PointSeeds
from .design import Covariates, draw_design
from .run import Provenance, run_design_point


def _one(theta: Covariates, ps: PointSeeds, cfg: Config, prov: Provenance, out_dir: Path,
         nn_table_path: str | None, deadline: float | None) -> str:
    path = out_dir / f"dp_{theta.index:06d}.parquet"
    if path.exists():
        return "skip"
    if deadline is not None and time.time() > deadline:
        return "timeout"
    table = NNBiasTable.load(nn_table_path) if nn_table_path else None
    rows = run_design_point(theta, ps, cfg, prov, nn_table=table)
    tmp = path.with_name(path.stem + ".tmp")
    write_runs(rows, tmp)
    tmp.replace(path)
    return "done"


def ordered_design(cfg: Config, *, limit: int | None = None,
                   largest_first: bool = False) -> list[tuple[Covariates, PointSeeds]]:
    """Designpunkte in Abarbeitungsreihenfolge (Filter ``only_N``, dann Reihenfolge, dann ``limit``)."""
    design = draw_design(cfg)
    if cfg.experiment.only_N is not None:
        design = [(t, ps) for t, ps in design if t.N in set(cfg.experiment.only_N)]
    if largest_first:
        design = sorted(design, key=lambda tp: (-tp[0].N, tp[0].index))
    if limit is not None:
        design = design[: int(limit)]
    return design


def run_experiment(cfg: Config, out_dir: str | Path, *, n_jobs: int = 1, limit: int | None = None,
                   repo: str | Path | None = None, time_limit_s: float | None = None,
                   verbose: int = 0, largest_first: bool = False) -> dict[str, int]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out / "config_effective.yaml")
    prov = Provenance.of(cfg, repo)
    design = ordered_design(cfg, limit=limit, largest_first=largest_first)
    table = cfg.observables.nn_bias_table
    deadline = time.time() + time_limit_s if time_limit_s else None
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_one)(theta, ps, cfg, prov, out, table, deadline) for theta, ps in design
    )
    return {k: results.count(k) for k in ("done", "skip", "timeout")}
