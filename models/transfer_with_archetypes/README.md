# Transfer Impact Model: With Archetypes

This folder contains the D2-to-D1 transfer impact model with D2 archetype
soft-vector features.

## Input

```text
../../data/transfer_model_ready_with_archetypes.csv
```

The saved outputs in this folder were produced from the archetype feature set.
The training script can also rebuild the model-ready table from archived
upstream inputs:

```bash
python models/transfer_with_archetypes/train_transfer_impact_models.py \
  --include-archetypes \
  --out-dir models/transfer_with_archetypes/runs
```

## Files

- `train_transfer_impact_models.py`: model training/evaluation script.
- `results.csv`: model performance by target, model family, and split.
- `predictions.csv`: player-level actual and predicted outcomes.
- `calibration_deciles.csv`: prediction calibration buckets.
- `run_summary.json`: run metadata.
- `feature_target_coverage.csv`: feature and target coverage.

The archetype features are the D2 `k6_drop7_8_*` soft-vector fields.
