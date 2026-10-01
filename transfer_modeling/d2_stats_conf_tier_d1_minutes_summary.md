# Transfer impact models: D2 stats + conference tier + D1 minutes

Both runs use D2 stat inputs and destination `d1_conf_tier`. The `with_minutes` regime also includes D1 `d1_GP` and `d1_mins_per_game`. The archetype run adds the six no-minutes fixed drop-5/7 archetype vector features.

| Run | Target | Model | Test n | R2 | Pearson | Spearman | MAE |
|---|---|---|---:|---:|---:|---:|---:|
| archetypes_d2_stats_conf_tier_d1_minutes | bpm | multi_gradient_boosting | 159 | 0.367 | 0.619 | 0.605 | 1.998 |
| baseline_d2_stats_conf_tier_d1_minutes | bpm | multi_extra_trees | 159 | 0.369 | 0.633 | 0.593 | 2.027 |
| archetypes_d2_stats_conf_tier_d1_minutes | bpr | extra_trees | 162 | 0.397 | 0.649 | 0.635 | 1.275 |
| baseline_d2_stats_conf_tier_d1_minutes | bpr | extra_trees | 162 | 0.411 | 0.659 | 0.638 | 1.248 |
| archetypes_d2_stats_conf_tier_d1_minutes | porpag | ridge | 227 | 0.471 | 0.700 | 0.708 | 0.624 |
| baseline_d2_stats_conf_tier_d1_minutes | porpag | ridge | 227 | 0.473 | 0.701 | 0.709 | 0.620 |
| archetypes_d2_stats_conf_tier_d1_minutes | rapm | multi_extra_trees | 159 | 0.234 | 0.486 | 0.442 | 1.801 |
| baseline_d2_stats_conf_tier_d1_minutes | rapm | multi_extra_trees | 159 | 0.235 | 0.488 | 0.441 | 1.817 |

Full best-model comparison: `transfer_modeling/d2_stats_conf_tier_d1_minutes_comparison.csv`
