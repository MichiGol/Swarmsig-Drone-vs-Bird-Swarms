# Bird flock or drone swarm?

### 🌐 Website: **[swarmsig-drone-vs-bird-flocks.netlify.app](https://swarmsig-drone-vs-bird-flocks.netlify.app/)**

Can an airport radar tell drone swarms from bird flocks by how closely group members match speed? Simulations and
real data show swarm drones couple more strongly than modelled birds; some real birds couple more still. In a model
with real weather, the radar holds up while cameras struggle at night. Hobby project, not peer-reviewed.

**Paper:** [Do they move together? (PDF)](reports/paper/swarmsig_paper.pdf) · [Markdown](reports/paper/swarmsig_paper.md).
The website explains everything interactively, including the math behind every model.

This repository is the **reproduction package**: code, configurations, tests, the paper, the pre-registrations and
the result tables. Raw data and simulation outputs are not included; the steps below download or recompute them.

| Path | Content |
|---|---|
| `src/swarmsig/` | Python package: bird and drone models, wind, sensors, coupling statistics, null models, experiment runner, radar and optics model |
| `configs/` | one YAML file per simulation study, every parameter with its source |
| `scripts/` | simulation runner, data download and preparation, analyses |
| `tests/` | 193 unit and integration tests |
| `data/` | small input tables the simulations need; everything else is downloaded → [data/README.md](data/README.md) |
| `reports/*_prereg.md` | the pre-registrations (German), written and committed before each run |
| `reports/*.csv`, `reports/figures/*.csv`, `*.json` | the result tables: AUC′ values with bootstrap intervals, real-data values per time window, system-model tables |

## Setup

Python ≥ 3.11 (tested with 3.12 on Windows 11; exact versions in `requirements.lock.txt`).

```bash
pip install -e ".[dev,fast,analysis]"
python -m pytest        # 193 tests, about 1 minute
```

## Reproducing the results

Rerunning an analysis overwrites its table in `reports/`. In a git checkout, `git diff reports/` therefore shows any
difference from the published results. Analyses that need no downloaded data, for example
`python scripts/o21_stage1_ranges.py`, reproduce their tables byte for byte.

**1. Get the data** (details, links and licenses in [data/README.md](data/README.md))

```bash
python scripts/download_dwd_o21.py              # airport weather for Part 2 (about 50 MB)
python scripts/prepare_dwd_o21.py
python scripts/prepare_dwd_o21.py --years 2018-2018 --out data/processed/dwd_o21_weather_2018.parquet
# trajectory data: download the files listed in data/README.md into data/raw/, then
python scripts/bird_data.py
python scripts/vasarhelyi_data.py
```

**2. Run the simulations** (each 5–40 minutes on a laptop; all 19 together take several hours)

```bash
python scripts/run_phase.py --config configs/<config>.yaml --out results/<folder> --jobs 4
```

| Config → folder | Tests |
|---|---|
| `p1_small` → `P1_small_v1` | first study, small groups, ideal sensor, no wind |
| `p1_confirm` → `P1_confirm_v1`; `p1_sens_sd03`, `p1_sens_sd10` → `P1_sens_sd03`, `P1_sens_sd10` | confirmation run; drone mechanical noise |
| `p3_wind` → `P3_wind_v1`; `p3_gustfast` → `P3_gustfast_v1` | real autumn wind and turbulence; faster wind estimation of the drones |
| `p3_noise_sd03`, `p3_noise_sd10` → same names | drone noise in wind |
| `p3_pos_gps_own`, `_gps_ref`, `_rtk_own`, `_rtk_ref` → same names | standard GPS and RTK positioning |
| `p3_sensor` → `P3_sensor_v1` | measurement noise and velocity differencing |
| `o3c_robust` → `O3c_robust_v1` | simulation under the robust statistics Q and L |
| `o5_calm_all`, `o5_df_base`, `o5_df_calm`, `o5_df_gps`, `o5_df_rtk` → same names in capitals (`O5_calm_all`, …) | formation drones and wind |

The variants of `P1_confirm_v1` and `P3_wind_v1` simulate only drones; their analyses take the birds from the main
run, so run all 19 studies before the analyses.

**3. Run the analyses**, in this order (later ones read tables written by earlier ones):

```bash
python scripts/analyze_p1.py --runs results/P1_small_v1 --out reports/P1_results.md
python scripts/analyze_confirm.py
python scripts/analyze_p3.py
python scripts/analyze_p3_noise.py
python scripts/analyze_p3_gps.py
python scripts/analyze_p3_sensor.py
python scripts/analyze_real_birds.py
python scripts/analyze_real_birds.py --variants Q,A7,M,L,RL --report reports/O3b_echte_Voegel_results.md
python scripts/analyze_vasarhelyi.py
python scripts/o3b_statistic_suite.py          # the turning artefact on synthetic groups
python scripts/analyze_o3c.py
python scripts/analyze_o5_calm.py
python scripts/analyze_o5.py
python scripts/analyze_swifts.py               # needs the chimney-swift file (587 MB)
python scripts/analyze_temporal.py             # needs the chimney-swift file
python scripts/o21_stage1_ranges.py            # Part 2: radar ranges
python scripts/o21_stage1_sensitivity.py
python scripts/o21_stage2_cues.py              # discrimination cues
python scripts/o21_stage3_climate.py           # six years of real weather
python scripts/o21_stage4_birds.py             # bird load
```

The bird-model calibration and the null-model tables in `data/processed/` can be recomputed with
`scripts/calibrate_bird_model.py` and `scripts/build_nn_bias_table.py`.

## Statement on the use of AI

The idea, research question, project plan, sensor concept, choice of real jackdaws as the reference, and the idea for
the system model in Part 2 are my own. I developed the research plan, including models, hypotheses, statistics, and
validation steps; an AI chatbot was used as a sounding board for the radar equations. I located and downloaded the
public flight data for real birds and drones and provided it for the analysis. I made the key scientific decisions,
including which studies to run and the radar-for-detection/camera-for-discrimination concept. I used Claude
(Anthropic) as an AI assistant to implement the plan: it worked out the detailed methods, wrote the code, ran the
simulations and data analyses, and wrote the reports, the paper text, and the website. I specified the requirements,
reviewed the results, and take full responsibility for the correctness and integrity of the work. No AI system is an
author; all scientific claims and any errors are my responsibility.

## Data sources

Trajectory data (CC0, Zenodo mirrors of Dryad): homing pigeons (Papadopoulou et al. 2022, data of Sankey et al.
2021), jackdaws (Ling et al. 2019), chimney swifts (Evangelista et al. 2017), a 30-drone outdoor swarm (Vásárhelyi et
al. 2018). Weather: Deutscher Wetterdienst, Climate Data Center (CC BY 4.0). Bird-migration profiles:
[Aloft](https://aloftdata.eu), source baltrad (CC0). Details: [data/README.md](data/README.md).
