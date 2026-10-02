#!/usr/bin/env python3
"""Promote the no-minutes k8 -> fixed k6 archetype vectors to active data files."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "player_archetypes"
VECTOR_FILE = MODEL_DIR / "k8_no_minutes_fixed_k6_drop5_7_vectors.csv"
LABEL_FILE = MODEL_DIR / "k6_no_minutes_drop5_7_labels.csv"
RUN_SUMMARY_FILE = MODEL_DIR / "k8_no_minutes_run_summary.json"
EXPERIMENT_CONFIG_FILE = MODEL_DIR / "experiment_config.json"
EXPERIMENT_SUMMARY_FILE = MODEL_DIR / "experiment_summary.csv"
PROFILE_FILE = MODEL_DIR / "k8_no_minutes_profiles.csv"
SCALING_FILE = MODEL_DIR / "k8_no_minutes_scaling.csv"

ENRICHED_FILE = ROOT / "data" / "enriched_d1_d2_player_stats_with_archetypes.csv"
TRANSFER_FILE = ROOT / "data" / "transfer_model_ready_with_archetypes.csv"
TRAIN_SCRIPT = ROOT / "models" / "transfer_with_archetypes" / "train_transfer_impact_models.py"

JOIN_KEYS = ["division", "season", "team", "player_name", "player_uid", "player_season_key"]
STAT_PROFILE_COLS = [
    "mins_per_game",
    "pts_per_game",
    "RPG",
    "ast_per_game",
    "TOPG",
    "stl_per_game",
    "blk_per_game",
    "usg",
    "FG_pct",
    "eFG",
    "TS_pct",
    "3P_pct",
    "FT_pct",
    "FTR",
    "three_share",
    "three_pa_per_100_team_poss",
    "TOV_pct",
    "AST_pct",
    "ORB_pct",
    "DRB_pct",
    "Blk_pct",
    "Stl_pct",
    "AST_TOV",
    "height_inches",
]


def vector_columns(vectors: pd.DataFrame) -> list[str]:
    prefixes = ("k8_no_minutes_", "k6_no_minutes_drop5_7_")
    return [col for col in vectors.columns if col.startswith(prefixes)]


def drop_old_archetype_columns(df: pd.DataFrame) -> pd.DataFrame:
    old_prefixes = (
        "k8_archetype_",
        "k6_drop7_8_",
        "k6_fixed_1to6_",
        "k8_no_minutes_",
        "k6_no_minutes_drop5_7_",
    )
    old_cols = [
        col
        for col in df.columns
        if col == "k8_archetype_id" or col.startswith(old_prefixes)
    ]
    return df.drop(columns=old_cols, errors="ignore")


def promote_player_vectors(vectors: pd.DataFrame) -> None:
    enriched = pd.read_csv(ENRICHED_FILE, low_memory=False)
    enriched = drop_old_archetype_columns(enriched)
    vector_subset = vectors[JOIN_KEYS + vector_columns(vectors)].copy()

    duplicate_keys = int(vector_subset.duplicated(JOIN_KEYS).sum())
    if duplicate_keys:
        raise ValueError(f"New vector export has {duplicate_keys} duplicate join keys")

    merged = enriched.merge(vector_subset, on=JOIN_KEYS, how="left", validate="m:1")
    matched = int(merged["k6_no_minutes_drop5_7_archetype_id"].notna().sum())
    if matched != len(vectors):
        raise ValueError(f"Expected {len(vectors)} matched vector rows, got {matched}")
    merged.to_csv(ENRICHED_FILE, index=False)


def load_transfer_module():
    spec = importlib.util.spec_from_file_location("transfer_with_archetypes", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rebuild_transfer_ready() -> bool:
    module = load_transfer_module()
    required_inputs = [
        Path(module.TRANSFER_PATH),
        Path(module.COMBINED_PATH),
        Path(module.ALT_D2_ADVANCED_PATH),
    ]
    missing_inputs = [path for path in required_inputs if not path.exists()]
    if missing_inputs:
        print(
            "Skipping transfer-ready rebuild because raw source exports are not present: "
            + ", ".join(str(path) for path in missing_inputs)
        )
        return False

    df = module.load_and_join(
        Path(module.TRANSFER_PATH),
        Path(module.COMBINED_PATH),
        Path(module.ALT_D2_ADVANCED_PATH),
        VECTOR_FILE,
        "no_minutes_drop5_7_fixed",
    )
    df.to_csv(TRANSFER_FILE, index=False)
    summary = {
        "rows": int(len(df)),
        "d2_archetype_vector_matches": int(df.attrs.get("d2_archetype_vector_matches", 0)),
        "archetype_vector_source": str(VECTOR_FILE.relative_to(ROOT)),
        "archetype_mode": "no_minutes_drop5_7_fixed",
    }
    (ROOT / "data" / "transfer_model_ready_with_archetypes.summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return True


def refresh_model_artifacts(vectors: pd.DataFrame) -> None:
    def copy_if_different(src: Path, dst: Path) -> None:
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    copy_if_different(LABEL_FILE, MODEL_DIR / "k6_no_minutes_drop5_7_labels.csv")
    copy_if_different(PROFILE_FILE, MODEL_DIR / "k8_no_minutes_profiles.csv")
    copy_if_different(PROFILE_FILE, MODEL_DIR / "k8_profiles.csv")
    copy_if_different(SCALING_FILE, MODEL_DIR / "k8_no_minutes_scaling.csv")
    copy_if_different(SCALING_FILE, MODEL_DIR / "k8_scaling.csv")
    copy_if_different(RUN_SUMMARY_FILE, MODEL_DIR / "k8_no_minutes_run_summary.json")
    copy_if_different(EXPERIMENT_CONFIG_FILE, MODEL_DIR / "experiment_config.json")
    copy_if_different(EXPERIMENT_SUMMARY_FILE, MODEL_DIR / "experiment_summary.csv")

    labels = pd.read_csv(LABEL_FILE)
    summary = labels[
        [
            "k6_archetype_id",
            "archetype_name",
            "rows_after_reprojection",
            "share",
            "d1_rows",
            "d2_rows",
            "avg_conf",
            "avg_removed_mass",
        ]
    ].rename(
        columns={
            "k6_archetype_id": "archetype_id",
            "archetype_name": "label",
            "rows_after_reprojection": "players",
            "avg_conf": "avg_confidence",
        }
    )
    summary.to_csv(MODEL_DIR / "k6_summary.csv", index=False)
    labels[
        [
            "source_k8_archetype_id",
            "k6_archetype_id",
            "archetype_name",
            "source_k8_rows_before_drop",
            "rows_after_reprojection",
            "avg_removed_mass",
        ]
    ].to_csv(MODEL_DIR / "k6_redistribution_matrix.csv", index=False)

    enriched = pd.read_csv(ENRICHED_FILE, low_memory=False)
    profile_rows = []
    for _, label_row in labels.iterrows():
        archetype_id = int(label_row["k6_archetype_id"])
        mask = enriched["k6_no_minutes_drop5_7_archetype_id"].eq(archetype_id)
        row = {
            "archetype_id": archetype_id,
            "label": label_row["archetype_name"],
            "n": int(mask.sum()),
        }
        for col in STAT_PROFILE_COLS:
            if col in enriched.columns:
                row[col] = pd.to_numeric(enriched.loc[mask, col], errors="coerce").mean()
        profile_rows.append(row)
    pd.DataFrame(profile_rows).to_csv(MODEL_DIR / "k6_actual_game_stats.csv", index=False)

    k8 = vectors["k8_no_minutes_archetype_id"].value_counts(dropna=True).sort_index()
    k6 = vectors["k6_no_minutes_drop5_7_archetype_id"].value_counts(dropna=True).sort_index()
    lines = ["k8 no-minutes labels on all D1/D2 player-seasons", "archetype_id,players,share"]
    total = len(vectors)
    lines.extend(f"{int(idx)},{int(count)},{count / total:.10f}" for idx, count in k8.items())
    lines.extend(["", "k6 no-minutes drop5_7 fixed labels on all D1/D2 player-seasons", "archetype_id,label,players,share"])
    label_lookup = labels.set_index("k6_archetype_id")["archetype_name"].to_dict()
    lines.extend(
        f"{int(idx)},{label_lookup[int(idx)]},{int(count)},{count / total:.10f}"
        for idx, count in k6.items()
    )
    (MODEL_DIR / "vector_summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    vectors = pd.read_csv(VECTOR_FILE, low_memory=False)
    promote_player_vectors(vectors)
    rebuilt_transfer = rebuild_transfer_ready()
    refresh_model_artifacts(vectors)
    print(f"Promoted no-minutes archetypes from {VECTOR_FILE.relative_to(ROOT)}")
    print(f"Updated {ENRICHED_FILE.relative_to(ROOT)}")
    if rebuilt_transfer:
        print(f"Updated {TRANSFER_FILE.relative_to(ROOT)}")
    else:
        print(f"Kept existing {TRANSFER_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
