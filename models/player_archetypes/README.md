# Player Archetype Model

This folder contains the selected player archetype model artifacts.

## Selected Run

Run id:

```text
k8__pooled_zscore__no_height_no_minutes_no_efficiency__method-nnls__init-uniform__ninit-1
```

The model was run with the Python `archetypes` package using a pooled z-score
feature scale, no height features, no raw minutes features, and no efficiency
features.

## Inputs

Use:

```text
../../data/archetype_model_features.csv
```

From the repository root, a fresh experiment can be launched with:

```bash
python models/player_archetypes/run_archetype_model.py \
  --input data/archetype_model_features.csv \
  --output-dir models/player_archetypes/runs
```

## Important Files

- `run_archetype_model.py`: archetypal-analysis experiment runner.
- `k8_no_minutes_profiles.csv`: active k8 no-minutes archetype profile table.
- `k8_no_minutes_scaling.csv`: feature clipping and scaling values for the
  active selected run.
- `k6_no_minutes_drop5_7_labels.csv`: fixed k6 labels, source k8 ids, z-score
  rationale, distribution, confidence, and removed-mass diagnostics.
- `k6_summary.csv`: active fixed k6 distribution and removed-mass diagnostics.
- `k6_actual_game_stats.csv`: basketball-stat profile summary for the cleaned k6
  archetypes.
- `vector_summary.csv`: compact summary of the final vector export.

The active fixed k6 labels are:

1. Low Usage Connector
2. Rim Protecting Big
3. Lead Guard
4. Defensive Spacer
5. Scoring Big
6. Pure Shooter

The final row-level archetype assignments and weights are included in:

```text
../../data/enriched_d1_d2_player_stats_with_archetypes.csv
```
