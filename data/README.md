# Data Files

## `raw_d2_player_stats.csv`

Clean D2-only player-season box-score stats. This is the closest curated file to
the raw D2 statistical source used downstream.

Rows: see `data_manifest.csv`.

## `archetype_model_features.csv`

Curated D1/D2 player-season feature table used as the input to the player
archetype model. This file contains the role/usage/rate inputs used by the
archetypal-analysis run.

## `enriched_d1_d2_player_stats_with_archetypes.csv`

Full D1/D2 player-season stats with final archetype fields joined in. It
includes the original player/stat columns plus:

- `k8_archetype_id`
- `k8_archetype_confidence`
- `k8_archetype_1_weight` through `k8_archetype_8_weight`
- `k6_drop7_8_archetype_id`
- `k6_drop7_8_archetype_confidence`
- `k6_drop7_8_kept_mass_from_k8_1_to_6`
- `k6_drop7_8_removed_mass_from_k8_7_to_8`
- `k6_drop7_8_archetype_1_weight` through
  `k6_drop7_8_archetype_6_weight`

Fifty rows from the full D1/D2 stats table do not have final archetype values
because they were not present in the finalized archetype export.

## `transfer_model_ready_no_archetypes.csv`

One row per D2-to-D1 transfer used by the baseline transfer impact model. This
file includes D2 source stats, D1 destination context, known D1 minutes fields,
and joined D1 outcome targets.

## `transfer_model_ready_with_archetypes.csv`

Same transfer sample as the baseline table, with D2 archetype-vector features
included for the archetype transfer model.
