# Finalized Player Dataset With Archetypes

This folder contains the all-player D1/D2 player-season archetype export used by the transfer modeling work.

## Main File

`all_d1_d2_player_seasons_with_k8_and_k6_drop7_8_archetypes.csv`

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

The file keeps both the original 8-archetype solution and the cleaned 6-archetype version that drops/redistributes archetypes 7 and 8.

Original raw k8 columns:

- `k8_archetype_id`
- `k8_archetype_confidence`
- `k8_archetype_1_weight` through `k8_archetype_8_weight`

Cleaned redistributed k6 columns:

- `k6_drop7_8_archetype_id`
- `k6_drop7_8_archetype_confidence`
- `k6_drop7_8_kept_mass_from_k8_1_to_6`
- `k6_drop7_8_removed_mass_from_k8_7_to_8`
- `k6_drop7_8_archetype_1_weight` through `k6_drop7_8_archetype_6_weight`

## Redistribution Logic

For rows with retained mass in original archetypes 1-6, the cleaned k6 weights are the original k8 weights 1-6 renormalized to sum to 1:

```text
k6_weight_i = k8_weight_i / sum(k8_weight_1 ... k8_weight_6)
```

The dropped mass is preserved in:

```text
k6_drop7_8_removed_mass_from_k8_7_to_8 = k8_weight_7 + k8_weight_8
```

Rows with zero retained mass in archetypes 1-6 cannot be renormalized. There are 25 such rows; those use a uniform fallback of `1/6` across the six cleaned archetypes.

## Fixed-Centroid 1-6 Reprojection

`all_d1_d2_player_seasons_with_k8_and_fixed_k6_reprojected_archetypes.csv` is the preferred drop-7/8 version when we want archetypes 7 and 8 removed without simply redistributing their original weights.

This file keeps the original k8 archetype centers 1-6 fixed, removes centers 7 and 8, and recomputes every player-season's nonnegative weights against only centers 1-6. It is not a fresh k=6 model, so archetypes 1-6 keep their original k8 meanings.

Added fixed-projection columns:

- `k6_fixed_1to6_archetype_id`
- `k6_fixed_1to6_archetype_confidence`
- `k6_fixed_1to6_projection_sse`
- `k6_fixed_1to6_kept_mass_from_k8_1_to_6`
- `k6_fixed_1to6_removed_mass_from_k8_7_to_8`
- `k6_fixed_1to6_archetype_1_weight` through `k6_fixed_1to6_archetype_6_weight`

Validation summary: 37,262 rows; no missing fixed weights; max weight-sum error `6.66e-16`; no dominant-ID mismatches; dominant counts are 1: 4,111, 2: 4,319, 3: 7,091, 4: 7,147, 5: 5,669, 6: 8,925. See `fixed_k6_reprojection_summary.json` for the machine-readable summary.

## Validation

Validation checks passed before pushing:

- Raw k8 weights sum to 1 within floating-point tolerance.
- `kept_mass_from_k8_1_to_6` equals the sum of raw k8 weights 1-6.
- `removed_mass_from_k8_7_to_8` equals the sum of raw k8 weights 7-8.
- Cleaned k6 weights sum to 1 within floating-point tolerance.
- `k6_drop7_8_archetype_id` matches the dominant cleaned k6 weight.
- All nonzero-retained-mass rows follow the k8-to-k6 renormalization formula above.
- The only renormalization exceptions are the 25 zero-retained-mass rows using the documented uniform fallback.

## Supporting Files

- `drop7_8_redistribution_matrix.csv`: crosswalk from original dominant k8 archetype to redistributed k6 archetype.
- `drop7_8_redistributed_k6_summary.csv`: summary of redistributed k6 archetype counts and removed-mass diagnostics.
- `drop7_8_redistributed_k6_actual_game_stats.csv`: basketball-stat profile summary for the cleaned k6 archetypes.
- `vector_summary.csv`: compact source summary for the all-player vector export.

