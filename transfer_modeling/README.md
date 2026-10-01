# Transfer Modeling

This folder contains the reproducibility materials for the D2-to-D1 transfer impact modeling work.

The model predicts D1 outcomes for D2 men's basketball transfers using D2/source production, D1 destination context, known D1 playing time, and optionally D2 archetype soft-vector features.

## Start Here

- `transfer_impact_model_summary.md`: paper-facing summary with feature inputs and side-by-side results with and without archetype features.
- `TRANSFER_IMPACT_MODEL_HANDOFF.md`: longer technical handoff for rerunning or editing the modeling pipeline.
- `scripts/train_baseline_transfer_impact_models.py`: training script used for the current runs.

## Output Folders

### `no_archetypes_conf_tier/`

Model run using D2/source features, destination context, conference tier, and known D1 minutes, without archetype features.

### `with_archetypes_conf_tier/`

Same setup, plus D2 archetype soft-vector features.

### `no_d2_minutes_no_archetypes/`

Rerun that removes raw D2 minute features `d2_MIN` and `d2_MPG`, without archetype features. The normal `no_minutes` / `with_minutes` regimes still refer to whether known D1 minutes are excluded or included.

### `no_d2_minutes_fixed_k6_archetypes/`

Same no-D2-minutes setup, plus D2 fixed-centroid 1-6 archetype soft-vector features from `finalized_player_dataset_with_archetypes/all_d1_d2_player_seasons_with_k8_and_fixed_k6_reprojected_archetypes.csv`. This is the drop-7/8 approach that keeps original k8 centers 1-6 fixed and reprojects players onto those centers.

### No-D2-Minutes Summary Files

- `no_d2_minutes_summary.md`: compact paper-facing summary of the two reruns.
- `no_d2_minutes_comparison.csv`: best model by target/regime for the two reruns.

## Files In Each Output Folder

- `baseline_transfer_impact_model_ready.csv`: model-ready transfer table used by the run.
- `baseline_impact_model_results.csv`: model performance by target, model family, and split.
- `baseline_impact_model_predictions.csv`: player-level actual vs predicted values.
- `baseline_impact_model_calibration_deciles.csv`: prediction calibration buckets.
- `baseline_impact_model_run_summary.json`: compact run metadata and target coverage.
- `feature_target_coverage.csv`: non-null coverage for model features and targets.

## Main Targets

- `rapm`: Hoop Explorer RAPM net, `hoop_rapm_net`
- `bpr`: EvanMiya BPR, `evanmiya_bpr`
- `porpag`: BartTorvik PORPAG, `d1_PORPAG`
- `bpm`: BartTorvik BPM, `d1_bpm`

## Current Result Snapshot

With known D1 minutes, held-out season split:

| Target | No Archetype R2 | With Archetype R2 |
|---|---:|---:|
| BPR | 0.419 | 0.423 |
| PORPAG | 0.476 | 0.467 |
| BPM | 0.419 | 0.422 |
| RAPM | 0.278 | 0.283 |

## Notes

- Targets are not imputed; missing target rows are dropped separately per target.
- `d1_conf_tier` groups ACC, Big Ten, Big 12, and SEC as `power_4`; all other conferences are `mid_major`.
- The with-archetypes run uses D2 archetype vectors only, not next-season D1 archetypes.
- The original run outputs also remain under `combined_d1_d2_exports/`.

