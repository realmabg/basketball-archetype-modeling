# Player Archetype Model

This folder contains the selected player archetype model artifacts.

## Selected Run

Run id:

```text
k8__pooled_zscore__no_height_no_efficiency__method-pgd__init-uniform__ninit-1
```

The model was run with the Python `archetypes` package using a pooled z-score
feature scale, no height features, and no efficiency features.

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
- `reproject_to_k6.py`: fixed-center k6 reprojection helper retained for
  follow-up experiments; it reads the curated enriched D1/D2 archetype file by default.
- `k8_profiles.csv`: selected k8 archetype profile table.
- `k8_scaling.csv`: feature clipping and scaling values for the selected run.
- `k6_redistribution_matrix.csv`: mapping from original dominant k8 archetypes
  to cleaned k6 archetypes after dropping archetypes 7 and 8.
- `k6_summary.csv`: cleaned k6 distribution and removed-mass diagnostics.
- `k6_actual_game_stats.csv`: basketball-stat profile summary for the cleaned k6
  archetypes.
- `vector_summary.csv`: compact summary of the final vector export.

The final row-level archetype assignments and weights are included in:

```text
../../data/enriched_d1_d2_player_stats_with_archetypes.csv
```
