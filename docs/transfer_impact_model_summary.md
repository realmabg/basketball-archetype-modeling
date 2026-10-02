# D2-to-D1 Transfer Impact Model Summary

This project includes a transfer impact model that predicts how Division II men's basketball transfers perform after moving to Division I, using their D2 production plus D1 destination context.

## Model Targets

| Target | Source | Column |
|---|---|---|
| RAPM | Hoop Explorer | `hoop_rapm_net` |
| BPR | EvanMiya | `evanmiya_bpr` |
| PORPAG | BartTorvik | `d1_PORPAG` |
| BPM | BartTorvik | `d1_bpm` |

Targets are not imputed. Rows are dropped separately for each target when that target is missing.

## Dataset

- Transfer sample: 703 D2-to-D1 transfer rows
- No-archetype model-ready file: `data/transfer_model_ready_no_archetypes.csv`
- With-archetype model-ready file: `data/transfer_model_ready_with_archetypes.csv`
- Main training scripts: `models/transfer_baseline/train_transfer_impact_models.py` and `models/transfer_with_archetypes/train_transfer_impact_models.py`

## Model Inputs

Both versions use D2/source production, D1 destination context, and known D1 playing time.

### D2 Box Score Inputs

- `d2_GP`
- `d2_GS`
- `d2_MIN`
- `d2_MPG`
- `d2_PTS`
- `d2_PPG`
- `d2_FGM`
- `d2_FGA`
- `d2_3PM`
- `d2_3PA`
- `d2_FTM`
- `d2_FTA`
- `d2_TRB`
- `d2_AST`
- `d2_TOV`
- `d2_STL`
- `d2_BLK`

### D2 Efficiency, Usage, Rate, and Per-40 Inputs

- `d2_height_inches`
- `d2_eFG`
- `d2_TS_pct`
- `d2_FTR`
- `d2_three_share`
- `d2_AST_TOV`
- `d2_TOV_pct`
- `d2_AST_pct`
- `d2_usg`
- `d2_ORB_pct`
- `d2_DRB_pct`
- `d2_Blk_pct`
- `d2_Stl_pct`
- `d2_PTS_per_40`
- `d2_ORB_per_40`
- `d2_DRB_per_40`
- `d2_TRB_per_40`
- `d2_AST_per_40`
- `d2_TOV_per_40`
- `d2_STL_per_40`
- `d2_BLK_per_40`
- `d2_three_pct`
- `d2_ft_pct`
- `d2_three_pa_per_100_team_poss_model`

### Context and Destination Inputs

- `d2_season_end_year`
- `next_d1_season_end_year`
- `d1_height_inches`
- `d2_source_family`
- `d2_conf`
- `d2_team`
- `d1_conf`
- `d1_conf_tier`
- `d1_team`
- `next_d1_season`

`d1_conf_tier` groups ACC, Big Ten, Big 12, and SEC as `power_4`; every other conference is `mid_major`.

### Known D1 Minutes Inputs

The reported results below use the with-minutes regime, which adds:

- `d1_GP`
- `d1_mins_per_game`

### Archetype Inputs

The with-archetypes version also adds D2 archetype soft-vector fields:

- `d2_k6_no_minutes_drop5_7_archetype_confidence`
- `d2_k6_no_minutes_drop5_7_kept_mass_from_k8_nonjunk`
- `d2_k6_no_minutes_drop5_7_removed_mass_from_k8_5_7`
- Low Usage Connector weight
- Rim Protecting Big weight
- Lead Guard weight
- Defensive Spacer weight
- Scoring Big weight
- Pure Shooter weight
- `d2_archetype_vector_missing`

## Results Without Archetypes

These are the best current with-minutes results from the destination conference tier run without archetype features.

| Target | Best Model | Test N | MAE | RMSE | R2 | Pearson | Spearman |
|---|---|---:|---:|---:|---:|---:|---:|
| BPR | Gradient Boosting | 162 | 1.252 | 1.571 | 0.419 | 0.652 | 0.664 |
| PORPAG | Ridge | 227 | 0.620 | 0.819 | 0.476 | 0.707 | 0.717 |
| BPM | Multi Extra Trees | 159 | 1.959 | 2.319 | 0.419 | 0.669 | 0.613 |
| RAPM | Multi Extra Trees | 159 | 1.768 | 2.212 | 0.278 | 0.529 | 0.488 |

## Results With Archetypes

These are the best current with-minutes results from the destination conference tier run with D2 archetype soft-vector features included.

| Target | Best Model | Test N | MAE | RMSE | R2 | Pearson | Spearman |
|---|---|---:|---:|---:|---:|---:|---:|
| BPR | Ridge | 162 | 1.297 | 1.608 | 0.392 | 0.638 | 0.651 |
| PORPAG | Ridge | 227 | 0.623 | 0.820 | 0.475 | 0.706 | 0.718 |
| BPM | Multi Gradient Boosting | 159 | 1.978 | 2.346 | 0.405 | 0.648 | 0.659 |
| RAPM | Multi Extra Trees | 159 | 1.759 | 2.202 | 0.285 | 0.535 | 0.493 |

## Takeaway

The model has meaningful predictive signal for D2-to-D1 translation. PORPAG,
BPM, and BPR are the strongest targets, with held-out R2 values around
0.39-0.48 depending on the target and feature set. RAPM is noisier but still
shows signal, with R2 around 0.28.

Adding the active no-minutes archetype features gives a small RAPM gain and
keeps PORPAG essentially unchanged, while BPR and BPM move modestly depending on
the metric used to pick the best model. The archetype features appear useful as
style/context descriptors, but destination context and minutes variables remain
the strongest drivers in the current setup.

## Modeling Folders

The curated model output folders are:

- `models/transfer_baseline/`
- `models/transfer_with_archetypes/`

Each folder contains:

- `baseline_transfer_impact_model_ready.csv`
- `baseline_impact_model_results.csv`
- `baseline_impact_model_predictions.csv`
- `baseline_impact_model_calibration_deciles.csv`
- `baseline_impact_model_run_summary.json`
- `feature_target_coverage.csv`
