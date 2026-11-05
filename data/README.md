# Data

## Included (`processed/`, 6 MB)

| File | Content | Why it is included |
|---|---|---|
| `b1_posterior.parquet`, `b3_posterior.parquet` | calibrated parameter sets of the bird models B1 and B3 | the simulations draw from them; recomputing takes long (`scripts/calibrate_bird_model.py`) |
| `nn_bias_table_small.parquet`, `nn_bias_table_robust.parquet` | null-model bias E₀[c_nn] over group size N | bias correction of T_nn (`scripts/build_nn_bias_table.py`) |
| `dwd_wind_autumn.parquet` | autumn wind and gusts at the four airports, 2010–2025 | the wind in the simulations; keeps them exactly reproducible even if DWD revises its archive |
| `aloft_hourly_2018.parquet` | hourly bird densities in 200 m height bins, three weather radars, 2018 | the bird load in Part 2; saves a download of 260 MB |

## Downloaded (into `data/raw/`, not included)

| Data set | Source | License | Put into `data/raw/` | Size |
|---|---|---|---|---|
| Jackdaw flocks (Ling et al. 2019) | [Zenodo 5007024](https://doi.org/10.5281/zenodo.5007024) (mirror of Dryad [doi:10.5061/dryad.kb8js06](https://doi.org/10.5061/dryad.kb8js06)) | CC0 | `ling2019_jackdaws/DataS1.txt` | 17 MB |
| Homing pigeon flocks (Papadopoulou et al. 2022, data of Sankey et al. 2021) | [Zenodo 6027115](https://doi.org/10.5281/zenodo.6027115) (Dryad [doi:10.5061/dryad.905qfttkc](https://doi.org/10.5061/dryad.905qfttkc)) | CC0 | `papadopoulou2022_pigeons/all_emp_data.csv` | 17 MB |
| 30-drone outdoor swarm (Vásárhelyi et al. 2018) | [Zenodo 4983910](https://doi.org/10.5281/zenodo.4983910) (Dryad [doi:10.5061/dryad.mq85r61](https://doi.org/10.5061/dryad.mq85r61)) | CC0 | `vasarhelyi2018/flight_logs.zip` | 7 MB |
| Chimney swifts (Evangelista et al. 2017), only for `analyze_swifts.py` and `analyze_temporal.py` | [Zenodo 5082229](https://doi.org/10.5281/zenodo.5082229) (Dryad [doi:10.5061/dryad.p68f8](https://doi.org/10.5061/dryad.p68f8)) | CC0 | `evangelista2017_swifts/A_nodes.hdf5` | 587 MB |
| Airport weather (DWD stations 00427, 00691, 01975, 02014) | [DWD Climate Data Center](https://opendata.dwd.de/climate_environment/CDC/) | CC BY 4.0 | `dwd/` via `python scripts/download_dwd_o21.py` | about 50 MB |
| Bird-migration profiles, other years | [Aloft](https://aloftdata.eu) | CC0 | `aloft/` via `python scripts/download_aloft.py` | up to 1 GB |

Dryad blocks script downloads, so the trajectory files come from the official Zenodo mirrors (same files). The
readers parse them as plain data only. `scripts/bird_data.py` and `scripts/vasarhelyi_data.py` then write the
processed trajectory files; `scripts/prepare_dwd_o21.py` and `scripts/prepare_aloft.py` do the same for weather and
bird profiles.

## Attribution

- **DWD:** Source: Deutscher Wetterdienst (DWD), Climate Data Center, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  `dwd_wind_autumn.parquet` is an aggregated and reformatted extract of the station data.
- **Aloft:** CC0 waiver. Suggested citation: Desmet, P. et al. (2025), *Scientific Data*,
  [doi:10.1038/s41597-025-04641-5](https://doi.org/10.1038/s41597-025-04641-5). We acknowledge the European Operational
  Program for Exchange of Weather Radar Information (EUMETNET/OPERA) for providing access to European radar data; full
  text in the [Aloft FAQ](https://aloftdata.eu/faq/).
- **Trajectory data:** please cite the original articles listed in the [paper](../reports/paper/swarmsig_paper.md#references).
