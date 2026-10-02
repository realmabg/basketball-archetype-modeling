# Transfer Impact Model: Baseline

This folder contains the D2-to-D1 transfer impact model without archetype
features.

## Input

```text
../../data/transfer_model_ready_no_archetypes.csv
```

The saved outputs in this folder were produced from that baseline feature set.
The training script can also rebuild the model-ready table from archived
upstream inputs:

```bash
python models/transfer_baseline/train_transfer_impact_models.py \
  --out-dir models/transfer_baseline/runs
```

## Files

- `train_transfer_impact_models.py`: model training/evaluation script.
- `results.csv`: model performance by target, model family, and split.
- `predictions.csv`: player-level actual and predicted outcomes.
- `calibration_deciles.csv`: prediction calibration buckets.
- `run_summary.json`: run metadata.
- `feature_target_coverage.csv`: feature and target coverage.

The main targets are RAPM, BPR, PORPAG, and BPM.
