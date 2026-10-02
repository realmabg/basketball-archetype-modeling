# Transfer Impact Model Handoff

This repository keeps the curated D2-to-D1 transfer impact modeling inputs, scripts, and selected outputs.

## Curated Inputs

- `data/transfer_model_ready_no_archetypes.csv`: baseline transfer model table.
- `data/transfer_model_ready_with_archetypes.csv`: same transfer table with D2 archetype soft-vector features.
- `data/enriched_d1_d2_player_stats_with_archetypes.csv`: full D1/D2 player-season stats with final k8 and cleaned k6 archetype fields.

## Model Folders

- `models/transfer_baseline/`: baseline model script and outputs.
- `models/transfer_with_archetypes/`: archetype-feature model script and outputs.

## Rerun Commands

Baseline:

```bash
python models/transfer_baseline/train_transfer_impact_models.py \
  --model-ready data/transfer_model_ready_no_archetypes.csv \
  --out-dir models/transfer_baseline/runs
```

With archetypes:

```bash
python models/transfer_with_archetypes/train_transfer_impact_models.py \
  --model-ready data/transfer_model_ready_with_archetypes.csv \
  --out-dir models/transfer_with_archetypes/runs
```

## Targets

- `rapm`: Hoop Explorer RAPM net, `hoop_rapm_net`
- `bpr`: EvanMiya BPR, `evanmiya_bpr`
- `porpag`: BartTorvik PORPAG, `d1_PORPAG`
- `bpm`: BartTorvik BPM, `d1_bpm`

Targets are not imputed. Rows are dropped separately per target when a target is missing.

## Notes

The scripts can still rebuild model-ready files from upstream sources if paths are supplied explicitly, but the clean repo workflow starts from the curated model-ready CSVs in `data/`.
