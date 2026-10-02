# D2-to-D1 Transfer Impact Modeling Handoff

Cleanup note: this handoff was written before the repo was reorganized. The
curated model-ready inputs now live in `data/`, and the selected model artifacts
live in `models/transfer_baseline/` and `models/transfer_with_archetypes/`.
Paths below that reference the old working tree are retained as historical
technical detail; the corresponding old files are under `_archive_legacy/`.

This handoff is for someone who wants to edit the model features and rerun the D2-to-D1 transfer impact experiments.

## What This Pipeline Does

The training script builds a model-ready D2-to-D1 transfer table and evaluates models for four D1 outcome targets:

- `rapm`: Hoop Explorer RAPM net
- `bpr`: EvanMiya BPR
- `porpag`: Bart Torvik PORPAG
- `bpm`: Bart Torvik BPM

It evaluates two feature regimes:

- `no_minutes`: D2/source features plus destination context, without known D1 playing time
- `with_minutes`: the same feature set plus `d1_GP` and `d1_mins_per_game`

Targets are not imputed. Rows are dropped per target when that target is missing.

## Main Script

Use:

```bash
python scripts/train_transfer_impact_models.py
```

Important feature-edit locations in the script:

- `D2_NUMERIC_FEATURES`: D2 box, efficiency, usage, rebound, defense, and per-40 inputs
- `CONTEXT_NUMERIC_FEATURES`: season/height context
- `WITH_MINUTES_FEATURES`: D1 minutes fields used only in the `with_minutes` regime
- `ARCHETYPE_NUMERIC_FEATURES`: soft archetype vector fields
- `CATEGORICAL_FEATURES`: categorical variables before one-hot encoding
- `POWER_4_CONFERENCES`: conference tier mapping for `d1_conf_tier`
- `TARGETS`: model targets
- `models()`: model families and hyperparameters
- `split_masks()`: train/test split logic

## Required Inputs

These are the minimum files needed to reproduce the current runs from the local workspace.

### Core transfer/outcome file

```text
combined_d1_d2_exports/d2_to_d1_transfers_with_hoop_rapm_evanmiya_bpr.csv
```

Contains D2-to-D1 transfer rows plus joined RAPM/BPR targets.

### Combined D1/D2 model-ready player database

```text
combined_d1_d2_exports/combined_d1_d2_player_database_model_ready_step42_drop_skipped_height_schools.csv
```

Used for D1 outcomes such as PORPAG/BPM and for primary D2 advanced-rate joins.

### Supplemental D2 advanced stat sources

```text
fix_d2_data/exports/d2_player_database_with_heights_classes_step26.csv
fix_d2_data/working/clean_step18_roster_urls_24_heights.csv
d2_database_export/d2_player_seasons_2021_22_to_2025_26_website_priority.csv
d2_database_export/d2_player_seasons_2021_22_to_2025_26_website_priority_min5mpg_gp5_with_heights.csv
/Users/adriankong/Downloads/d2_player_seasons (1).csv
```

These are used to fill D2 possession-rate fields such as `d2_usg`, `d2_AST_pct`, `d2_TOV_pct`, `d2_ORB_pct`, `d2_DRB_pct`, `d2_Blk_pct`, and `d2_Stl_pct`.

### Optional archetype vector file

```text
outputs/no_minutes_k8_fixed_k6_drop5_7/all_d1_d2_player_seasons_k8_no_minutes_and_fixed_k6_drop5_7_vectors.csv
```

Equivalent local source used during development:

```text
combined_d1_d2_exports/archetype_experiments_step43_no_minutes_full_k8_nnls_vectors/k8__pooled_zscore__no_height_no_minutes_no_efficiency__method-nnls__init-uniform__ninit-1__all_d1_d2_player_seasons_k8_no_minutes_and_fixed_k6_drop5_7_vectors.csv
```

This provides the active no-minutes fixed k6 weights: Low Usage Connector, Rim
Protecting Big, Lead Guard, Defensive Spacer, Scoring Big, and Pure Shooter.

## Current Best Feature Set

The strongest clean with-minutes setup so far is destination conference tier plus known D1 minutes, without archetypes.

It includes:

- D2 box stats
- D2 shooting/efficiency stats
- D2 possession/rate stats
- D2 per-40 stats
- D2/D1 height and season context
- destination `d1_team`
- destination `d1_conf`
- destination `d1_conf_tier`
- `d1_GP`
- `d1_mins_per_game`

`d1_conf_tier` is currently:

```text
power_4 = ACC, B10, B12, SEC
mid_major = everything else
```

## Commands To Reproduce Runs

Baseline with destination conference tier:

```bash
python scripts/train_transfer_impact_models.py \
  --out-dir outputs/transfer_impact_models_conf_tier
```

Destination conference tier plus archetype vectors:

```bash
python scripts/train_transfer_impact_models.py \
  --include-archetypes \
  --archetype-vectors outputs/no_minutes_k8_fixed_k6_drop5_7/all_d1_d2_player_seasons_k8_no_minutes_and_fixed_k6_drop5_7_vectors.csv \
  --archetype-mode no_minutes_drop5_7_fixed \
  --out-dir outputs/transfer_impact_models_with_archetypes_conf_tier
```

If running from the original local workspace instead of this repo, use the default paths already embedded in the script.

## Current With-Minutes Results

Destination conference tier, no archetypes:

| Target | Best model | Test n | MAE | RMSE | R2 | Pearson | Spearman |
|---|---|---:|---:|---:|---:|---:|---:|
| BPM | multi gradient boosting | 159 | 2.008 | 2.345 | 0.405 | 0.649 | 0.620 |
| BPR | gradient boosting | 162 | 1.252 | 1.571 | 0.419 | 0.652 | 0.664 |
| PORPAG | ridge | 227 | 0.620 | 0.819 | 0.476 | 0.707 | 0.717 |
| RAPM | multi extra trees | 159 | 1.768 | 2.212 | 0.278 | 0.529 | 0.488 |

Destination conference tier plus archetypes:

| Target | Best model | Test n | MAE | RMSE | R2 | Pearson | Spearman |
|---|---|---:|---:|---:|---:|---:|---:|
| BPM | multi gradient boosting | 159 | 1.992 | 2.364 | 0.396 | 0.644 | 0.628 |
| BPR | gradient boosting | 162 | 1.261 | 1.566 | 0.423 | 0.655 | 0.661 |
| PORPAG | ridge | 227 | 0.625 | 0.826 | 0.467 | 0.699 | 0.714 |
| RAPM | multi extra trees | 159 | 1.761 | 2.204 | 0.283 | 0.534 | 0.487 |

## Output Files

Each run writes:

```text
baseline_transfer_impact_model_ready.csv
baseline_impact_model_results.csv
baseline_impact_model_predictions.csv
baseline_impact_model_calibration_deciles.csv
baseline_impact_model_run_summary.json
feature_target_coverage.csv
```

The most important file for comparing experiments is:

```text
baseline_impact_model_results.csv
```

## Notes For Optimizing

- Keep a held-out season split when comparing features so results stay comparable.
- Do not impute targets. Missing target rows should be dropped for that target.
- Be careful adding destination strength: use prior-year or preseason-known values to avoid leakage.
- Raw `d1_conf` and `d1_team` are already included, but `d1_conf_tier` gave a large lift by grouping sparse high-major context.
- Archetypes currently add only marginal value after `d1_conf_tier`; they may work better with alternative vector definitions or interactions.
