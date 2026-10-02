# D2 Archetypes and Transfer Impact Models

This repository is organized as a curated deliverable for the Division II men's
basketball player archetype and D2-to-D1 transfer impact modeling work.

## Repository Layout

```text
data/
  raw_d2_player_stats.csv
  archetype_model_features.csv
  enriched_d1_d2_player_stats_with_archetypes.csv
  transfer_model_ready_no_archetypes.csv
  transfer_model_ready_with_archetypes.csv
  data_manifest.csv
  sources.md

models/
  player_archetypes/
  transfer_baseline/
  transfer_with_archetypes/

docs/
  transfer_impact_model_summary.md
  transfer_impact_model_handoff.md

```

## Main Data Files

- `data/raw_d2_player_stats.csv`: clean D2-only box-score player-season stats.
- `data/archetype_model_features.csv`: exact curated D1/D2 feature table used by
  the player archetype model.
- `data/enriched_d1_d2_player_stats_with_archetypes.csv`: full D1/D2
  player-season stats with final k8 and cleaned k6 archetype assignments and
  soft-vector weights joined in.
- `data/transfer_model_ready_no_archetypes.csv`: D2-to-D1 transfer modeling table
  for the baseline transfer impact model.
- `data/transfer_model_ready_with_archetypes.csv`: same transfer modeling table
  with D2 archetype-vector features included.

## Model Folders

- `models/player_archetypes/`: archetypal-analysis runner, selected k8/k6
  profile outputs, scaling, and summary artifacts.
- `models/transfer_baseline/`: transfer impact model script and outputs without
  archetype features.
- `models/transfer_with_archetypes/`: transfer impact model script and outputs
  with D2 archetype soft-vector features.

## Reproducibility Note

This repo now preserves the curated inputs and model code needed to rerun the
archetype and transfer-impact modeling layers. The older raw scrape/PBP build
work, audit files, experiment grids, and duplicated packages are intentionally
not part of the clean public workflow.
