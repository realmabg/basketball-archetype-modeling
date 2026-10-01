# Transfer impact model results without D2 minutes

Generated from two reruns that remove `d2_MIN` and `d2_MPG` from the model inputs. The `no_minutes` regime still means no D1 minutes; the `with_minutes` regime adds D1 `d1_GP` and `d1_mins_per_game`.

## no_archetypes_no_d2_minutes

- Rows: 703
- Include archetypes: False
- Archetype mode: None
- D2 minute features removed: d2_MIN, d2_MPG
- D2 archetype vector matches: 0
- Results folder: `transfer_modeling/no_d2_minutes_no_archetypes`

## fixed_k6_archetypes_no_d2_minutes

- Rows: 703
- Include archetypes: True
- Archetype mode: fixed_1to6_reprojected
- D2 minute features removed: d2_MIN, d2_MPG
- D2 archetype vector matches: 696
- Results folder: `transfer_modeling/no_d2_minutes_fixed_k6_archetypes`

## Best models by target/regime

| run                               | target | regime       | model                   | test_n | r2    | pearson | spearman | mae   |
| --------------------------------- | ------ | ------------ | ----------------------- | ------ | ----- | ------- | -------- | ----- |
| fixed_k6_archetypes_no_d2_minutes | bpm    | no_minutes   | multi_gradient_boosting | 159    | 0.321 | 0.580   | 0.540    | 2.065 |
| fixed_k6_archetypes_no_d2_minutes | bpm    | with_minutes | multi_gradient_boosting | 159    | 0.397 | 0.642   | 0.609    | 2.012 |
| fixed_k6_archetypes_no_d2_minutes | bpr    | no_minutes   | gradient_boosting       | 162    | 0.325 | 0.594   | 0.601    | 1.333 |
| fixed_k6_archetypes_no_d2_minutes | bpr    | with_minutes | gradient_boosting       | 162    | 0.414 | 0.647   | 0.656    | 1.265 |
| fixed_k6_archetypes_no_d2_minutes | porpag | no_minutes   | multi_random_forest     | 159    | 0.132 | 0.422   | 0.430    | 0.822 |
| fixed_k6_archetypes_no_d2_minutes | porpag | with_minutes | ridge                   | 227    | 0.469 | 0.700   | 0.715    | 0.623 |
| fixed_k6_archetypes_no_d2_minutes | rapm   | no_minutes   | gradient_boosting       | 163    | 0.222 | 0.494   | 0.483    | 1.899 |
| fixed_k6_archetypes_no_d2_minutes | rapm   | with_minutes | multi_extra_trees       | 159    | 0.277 | 0.528   | 0.484    | 1.768 |
| no_archetypes_no_d2_minutes       | bpm    | no_minutes   | multi_gradient_boosting | 159    | 0.273 | 0.546   | 0.479    | 2.165 |
| no_archetypes_no_d2_minutes       | bpm    | with_minutes | multi_gradient_boosting | 159    | 0.413 | 0.661   | 0.642    | 1.966 |
| no_archetypes_no_d2_minutes       | bpr    | no_minutes   | gradient_boosting       | 162    | 0.309 | 0.589   | 0.605    | 1.339 |
| no_archetypes_no_d2_minutes       | bpr    | with_minutes | gradient_boosting       | 162    | 0.429 | 0.660   | 0.674    | 1.244 |
| no_archetypes_no_d2_minutes       | porpag | no_minutes   | multi_random_forest     | 159    | 0.127 | 0.418   | 0.412    | 0.830 |
| no_archetypes_no_d2_minutes       | porpag | with_minutes | ridge                   | 227    | 0.478 | 0.706   | 0.716    | 0.620 |
| no_archetypes_no_d2_minutes       | rapm   | no_minutes   | multi_gradient_boosting | 159    | 0.217 | 0.475   | 0.426    | 1.830 |
| no_archetypes_no_d2_minutes       | rapm   | with_minutes | multi_extra_trees       | 159    | 0.289 | 0.538   | 0.507    | 1.761 |

Full comparison table: `transfer_modeling/no_d2_minutes_comparison.csv`

## Inputs

Shared non-archetype inputs include D2 box-score/rate/efficiency features, D2/D1 team and conference context, season fields, height fields, and categorical team/conference/source fields. In these runs `d2_MIN` and `d2_MPG` are excluded from model features. Derived minute-normalized stats such as per-40 rates remain because they are performance rates, not raw minutes/MPG exposure variables.

The archetype run adds fixed-centroid 1-6 archetype confidence, kept/removed k8 mass metadata, six fixed 1-6 archetype weights, and a missing-vector flag. The fixed-centroid vectors keep the original k8 centers 1-6 and reproject every player-season onto those centers after removing centers 7 and 8; this is not a fresh k=6 archetype model.
