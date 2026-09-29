# Basketball Archetype Modeling

Reproducible player archetype experiments for combined D1/D2 college basketball data.

## Current Experiment

The earlier weighted setup:

- division-specific z-score, then recombine D1 and D2
- winsorize each feature at the 1st/99th percentile within division
- no `mins_per_game`
- no `height_inches`
- include efficiency stats
- six equal-weight stat groups
- `method = nnls`
- `k = 6..10`
- init methods tested: `uniform`, `furthest_sum`, `furthest_first`, `aa_plus_plus`

Feature groups:

- efficiency: `eFG`, `three_pct`, `ft_pct`
- usage: `FTR`, `usg`
- shooting style: `three_share`, `three_pa_per_100_team_poss`
- ball movement: `tov_pct`, `ast_pct`, `ast_tov`
- rebounding: `orb_pct`, `drb_pct`
- defense: `stl_pct`, `blk_pct`

Each group contributes the same total squared-distance weight. For example, each efficiency feature contributes `1/18` because the efficiency group has weight `1/6` and has three features.

## Files

- `scripts/run_archetype_experiment_grid.py`: reusable archetype experiment runner
- `configs/division_zscore_equal_group_weights_no_minutes_no_height.csv`: fixed feature weights
- `outputs/weighted_no_height_minutes_sample3000/experiment_summary.csv`: 20 raw sampled runs
- `outputs/weighted_no_height_minutes_sample3000/best_init_by_k.csv`: best init per k
- `outputs/weighted_no_height_minutes_sample3000/best_5_weighted_archetype_z_stats_wide.csv`: archetype z-vector review file

## Role-First Experiment

The `role_first_2026_09_29` experiment tests the next role-first design:

- division-specific z-score, then recombine D1 and D2
- winsorize each feature at the 1st/99th percentile within division
- no `mins_per_game`
- `method = nnls`
- `k = 6, 7, 8`
- init methods tested: `uniform`, `furthest_sum`, `furthest_first`, `aa_plus_plus`

Four feature/weight versions were tested:

- `role_only_no_height`
- `role_only_height_025`
- `quality_lite_no_height`
- `quality_lite_height_025`

The quality-lite versions keep `eFG`, `three_pct`, and `ft_pct`, but give the whole quality group only 10% total squared-distance influence. The height versions add `height_inches` with a direct z-score multiplier of `0.25`.

Key files:

- `outputs/role_first_2026_09_29/all_48_raw_runs_summary.csv`
- `outputs/role_first_2026_09_29/best_12_conceptual_runs.csv`
- `outputs/role_first_2026_09_29/best_12_archetype_z_stats_wide.csv`
- `outputs/role_first_2026_09_29/best_12_archetype_z_stats_long.csv`

## Data

Large source datasets are intentionally not committed. Keep them local under `data/` or point scripts at their location with `--input`.

The latest local input used while developing this repo was:

```text
/Users/adriankong/Desktop/archetype dataset/combined_d1_d2_exports/archetype_model_features_step43.csv
```

## Example

```bash
python scripts/run_archetype_experiment_grid.py \
  --input "/path/to/archetype_model_features_step43.csv" \
  --output-dir outputs/weighted_no_height_minutes_sample3000 \
  --k-values 6,7,8,9,10 \
  --scaling-modes division_zscore \
  --feature-sets no_height_no_minutes \
  --methods nnls \
  --inits uniform,furthest_sum,furthest_first,aa_plus_plus \
  --n-init 1 \
  --max-iter 80 \
  --sample-size 3000 \
  --weights-file configs/division_zscore_equal_group_weights_no_minutes_no_height.csv
```
