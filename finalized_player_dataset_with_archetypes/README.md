# Finalized Player Dataset With Archetypes

This folder contains the finalized all-player D1/D2 player-season archetype export used by the transfer modeling work.

## Main File

`all_d1_d2_player_seasons_with_k8_and_fixed_k6_reprojected_archetypes.csv`

This file includes 37,262 player-season rows:

- D1 2021-22: 4,016
- D1 2022-23: 4,017
- D1 2023-24: 4,017
- D1 2024-25: 4,069
- D1 2025-26: 4,101
- D2 2021-22: 3,494
- D2 2022-23: 3,417
- D2 2023-24: 3,306
- D2 2024-25: 3,405
- D2 2025-26: 3,420

## Archetype Columns

The file keeps the original 8-archetype solution and adds the fixed-centroid 1-6 reprojection that removes archetypes 7 and 8.

Original raw k8 columns:

- `k8_archetype_id`
- `k8_archetype_confidence`
- `k8_archetype_1_weight` through `k8_archetype_8_weight`

Fixed 1-6 reprojection columns:

- `k6_fixed_1to6_archetype_id`
- `k6_fixed_1to6_archetype_confidence`
- `k6_fixed_1to6_projection_sse`
- `k6_fixed_1to6_kept_mass_from_k8_1_to_6`
- `k6_fixed_1to6_removed_mass_from_k8_7_to_8`
- `k6_fixed_1to6_archetype_1_weight` through `k6_fixed_1to6_archetype_6_weight`

## Fixed-Centroid Drop-7/8 Logic

This is not a fresh k=6 model and it is not simple redistribution of the old k8 weights.

The procedure keeps the original k8 archetype centers 1-6 fixed, removes centers 7 and 8, and recomputes every player-season's nonnegative weights against only centers 1-6. That means archetypes 1-6 keep their original k8 meanings, while players formerly explained by 7/8 are projected onto the closest mix of retained archetypes.

## Validation

Validation checks passed before pushing:

- 37,262 rows are present.
- Fixed 1-6 weights have no missing values.
- Fixed 1-6 weights sum to 1 within floating-point tolerance; max weight-sum error is `6.66e-16`.
- `k6_fixed_1to6_archetype_id` matches the dominant fixed 1-6 weight for every row.
- Dominant fixed 1-6 counts are 1: 4,111, 2: 4,319, 3: 7,091, 4: 7,147, 5: 5,669, 6: 8,925.

## Supporting Files

- `fixed_k6_reprojection_summary.json`: machine-readable summary of the fixed 1-6 reprojection run.
