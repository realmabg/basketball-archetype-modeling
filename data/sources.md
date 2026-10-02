# Data Sources and Lineage

This file documents the curated files kept in `data/` and the local source files
they were copied or derived from during repo cleanup.

| Curated file | Source |
|---|---|
| `raw_d2_player_stats.csv` | `d2_database_export/d2_player_seasons_2021_22_to_2025_26_final_clean_d2_only_min5mpg_gp5.csv` |
| `archetype_model_features.csv` | `combined_d1_d2_exports/archetype_model_features_step43.csv` |
| `enriched_d1_d2_player_stats_with_archetypes.csv` | `combined_d1_d2_player_database_step30.csv` joined to `finalized_player_dataset_with_archetypes/all_d1_d2_player_seasons_with_k8_and_k6_drop7_8_archetypes.csv` |
| `transfer_model_ready_no_archetypes.csv` | `transfer_modeling/no_archetypes_conf_tier/baseline_transfer_impact_model_ready.csv` |
| `transfer_model_ready_with_archetypes.csv` | `transfer_modeling/with_archetypes_conf_tier/baseline_transfer_impact_model_ready.csv` |

The enriched D1/D2 file uses a composite join key:

```text
division + season + team + player_name
```

The source archetype export has unique values for this composite key. It matched
37,262 of 37,312 rows in the full D1/D2 stats file.
