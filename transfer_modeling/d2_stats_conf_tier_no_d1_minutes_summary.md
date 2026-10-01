# Transfer impact models: D2 stats + conference tier, no D1 minutes

This is the `no_minutes` regime from the same baseline/archetype transfer model reruns. It excludes D1 `d1_GP` and `d1_mins_per_game` from both baseline and archetype models.

| Run | Target | Model | Test n | R2 | Pearson | Spearman | MAE |
|---|---|---|---:|---:|---:|---:|---:|
| archetypes_d2_stats_conf_tier_no_d1_minutes | bpm | multi_extra_trees | 159 | 0.227 | 0.506 | 0.437 | 2.253 |
| baseline_d2_stats_conf_tier_no_d1_minutes | bpm | multi_extra_trees | 159 | 0.218 | 0.495 | 0.421 | 2.253 |
| archetypes_d2_stats_conf_tier_no_d1_minutes | bpr | gradient_boosting | 162 | 0.248 | 0.517 | 0.527 | 1.404 |
| baseline_d2_stats_conf_tier_no_d1_minutes | bpr | extra_trees | 162 | 0.273 | 0.554 | 0.542 | 1.376 |
| archetypes_d2_stats_conf_tier_no_d1_minutes | porpag | multi_random_forest | 159 | 0.114 | 0.384 | 0.367 | 0.843 |
| baseline_d2_stats_conf_tier_no_d1_minutes | porpag | multi_random_forest | 159 | 0.139 | 0.435 | 0.415 | 0.827 |
| archetypes_d2_stats_conf_tier_no_d1_minutes | rapm | elasticnet | 163 | 0.100 | 0.393 | 0.372 | 2.022 |
| baseline_d2_stats_conf_tier_no_d1_minutes | rapm | elasticnet | 163 | 0.120 | 0.406 | 0.378 | 2.001 |

Full best-model comparison: `transfer_modeling/d2_stats_conf_tier_no_d1_minutes_comparison.csv`
