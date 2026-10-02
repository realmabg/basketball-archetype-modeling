#!/usr/bin/env python3
"""Train baseline D2-to-D1 impact models.

The script builds a model-ready transfer table, joins D1 PORPAG and D2
efficiency/rate features from the combined player database, and evaluates
baseline models for RAPM, BPR, PORPAG, and BPM under two feature regimes:
pre-D1 only and pre-D1 plus known D1 minutes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


TRANSFER_PATH = ""
COMBINED_PATH = ""
ALT_D2_ADVANCED_PATH = ""
ARCHETYPE_VECTOR_PATH = "data/enriched_d1_d2_player_stats_with_archetypes.csv"
SUPPLEMENTAL_D2_ADVANCED_PATH = ""
WORKING_D2_ADVANCED_PATH = ""
WEBSITE_PRIORITY_D2_ADVANCED_PATH = ""
WEBSITE_MIN5_HEIGHTS_D2_ADVANCED_PATH = ""
MODEL_READY_PATH = "data/transfer_model_ready_no_archetypes.csv"
OUT_DIR = "models/transfer_baseline/runs"

TARGETS = {
    "rapm": "hoop_rapm_net",
    "bpr": "evanmiya_bpr",
    "porpag": "d1_PORPAG",
    "bpm": "d1_bpm",
}

D2_NUMERIC_FEATURES = [
    "d2_GP",
    "d2_GS",
    "d2_MIN",
    "d2_MPG",
    "d2_PTS",
    "d2_PPG",
    "d2_FGM",
    "d2_FGA",
    "d2_3PM",
    "d2_3PA",
    "d2_FTM",
    "d2_FTA",
    "d2_TRB",
    "d2_AST",
    "d2_TOV",
    "d2_STL",
    "d2_BLK",
    "d2_height_inches",
    "d2_eFG",
    "d2_TS_pct",
    "d2_FTR",
    "d2_three_share",
    "d2_AST_TOV",
    "d2_TOV_pct",
    "d2_AST_pct",
    "d2_usg",
    "d2_ORB_pct",
    "d2_DRB_pct",
    "d2_Blk_pct",
    "d2_Stl_pct",
    "d2_PTS_per_40",
    "d2_ORB_per_40",
    "d2_DRB_per_40",
    "d2_TRB_per_40",
    "d2_AST_per_40",
    "d2_TOV_per_40",
    "d2_STL_per_40",
    "d2_BLK_per_40",
    "d2_three_pct",
    "d2_ft_pct",
    "d2_three_pa_per_100_team_poss_model",
]

CONTEXT_NUMERIC_FEATURES = [
    "d2_season_end_year",
    "next_d1_season_end_year",
    "d1_height_inches",
]

CATEGORICAL_FEATURES = [
    "d2_source_family",
    "d2_conf",
    "d2_team",
    "d1_conf",
    "d1_conf_tier",
    "d1_team",
    "next_d1_season",
]
CONFERENCE_TIER_CATEGORICAL_FEATURES = ["d1_conf_tier"]

WITH_MINUTES_FEATURES = ["d1_GP", "d1_mins_per_game"]

ARCHETYPE_NUMERIC_FEATURES = [
    "d2_k6_drop7_8_archetype_confidence",
    "d2_k6_drop7_8_kept_mass_from_k8_1_to_6",
    "d2_k6_drop7_8_removed_mass_from_k8_7_to_8",
    "d2_k6_drop7_8_archetype_1_weight",
    "d2_k6_drop7_8_archetype_2_weight",
    "d2_k6_drop7_8_archetype_3_weight",
    "d2_k6_drop7_8_archetype_4_weight",
    "d2_k6_drop7_8_archetype_5_weight",
    "d2_k6_drop7_8_archetype_6_weight",
    "d2_archetype_vector_missing",
]

D2_MINUTE_FEATURES = {"d2_MIN", "d2_MPG"}

ARCHETYPE_COLUMN_MAPS = {
    "drop7_8_renormalized": {
        "k6_drop7_8_archetype_confidence": "k6_drop7_8_archetype_confidence",
        "k6_drop7_8_kept_mass_from_k8_1_to_6": "k6_drop7_8_kept_mass_from_k8_1_to_6",
        "k6_drop7_8_removed_mass_from_k8_7_to_8": "k6_drop7_8_removed_mass_from_k8_7_to_8",
        **{
            f"k6_drop7_8_archetype_{idx}_weight": f"k6_drop7_8_archetype_{idx}_weight"
            for idx in range(1, 7)
        },
    },
    "fixed_1to6_reprojected": {
        "k6_drop7_8_archetype_confidence": "k6_fixed_1to6_archetype_confidence",
        "k6_drop7_8_kept_mass_from_k8_1_to_6": "k6_fixed_1to6_kept_mass_from_k8_1_to_6",
        "k6_drop7_8_removed_mass_from_k8_7_to_8": "k6_fixed_1to6_removed_mass_from_k8_7_to_8",
        **{
            f"k6_drop7_8_archetype_{idx}_weight": f"k6_fixed_1to6_archetype_{idx}_weight"
            for idx in range(1, 7)
        },
    },
    "no_minutes_drop5_7_fixed": {
        "k6_drop7_8_archetype_confidence": "k6_no_minutes_drop5_7_archetype_confidence",
        "k6_drop7_8_kept_mass_from_k8_1_to_6": "k6_no_minutes_drop5_7_kept_mass_from_k8_nonjunk",
        "k6_drop7_8_removed_mass_from_k8_7_to_8": "k6_no_minutes_drop5_7_removed_mass_from_k8_5_7",
        "k6_drop7_8_archetype_1_weight": "k6_no_minutes_drop5_7_archetype_1_source_k8_1_low_usage_connector_weight",
        "k6_drop7_8_archetype_2_weight": "k6_no_minutes_drop5_7_archetype_2_source_k8_2_rim_protecting_big_weight",
        "k6_drop7_8_archetype_3_weight": "k6_no_minutes_drop5_7_archetype_3_source_k8_3_lead_guard_weight",
        "k6_drop7_8_archetype_4_weight": "k6_no_minutes_drop5_7_archetype_4_source_k8_4_defensive_spacer_weight",
        "k6_drop7_8_archetype_5_weight": "k6_no_minutes_drop5_7_archetype_5_source_k8_6_scoring_big_weight",
        "k6_drop7_8_archetype_6_weight": "k6_no_minutes_drop5_7_archetype_6_source_k8_8_pure_shooter_weight",
    },
}


def norm_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


TEAM_ALIASES = {
    "ark monticello": "arkansas monticello",
    "cal st east bay": "cal state east bay",
    "cal st la": "cal state la",
    "cal st monterey bay": "cal state monterey bay",
    "chadron": "chadron state",
    "clayton st": "clayton state",
    "dist columbia": "district of columbia",
    "lee university": "lee",
    "millersville university": "millersville",
    "newman university": "newman",
    "northern st": "northern state",
    "palm beach atl": "palm beach atlantic",
    "saint rose": "st rose",
    "texas tyler": "ut tyler",
    "university of mary": "mary",
    "west ala": "west alabama",
}

NAME_ALIASES = {
    "jae slack": {"jaedaun slack"},
    "jaedaun slack": {"jae slack"},
    "josh ward": {"joshua ward"},
    "joshua ward": {"josh ward"},
    "king ijeoma": {"kinglsey ijeoma", "kingsley ijeoma"},
    "kinglsey ijeoma": {"king ijeoma", "kingsley ijeoma"},
    "kingsley ijeoma": {"king ijeoma", "kinglsey ijeoma"},
    "matthew spears": {"matt spears"},
    "matt spears": {"matthew spears"},
    "cam mcdowell": {"camron mcdowell"},
    "camron mcdowell": {"cam mcdowell"},
}

POWER_4_CONFERENCES = {"ACC", "B10", "B12", "SEC"}


def d1_conf_tier(value: object) -> str:
    conf = "" if pd.isna(value) else str(value).strip()
    return "power_4" if conf in POWER_4_CONFERENCES else "mid_major"


def team_key(value: object) -> str:
    key = norm_text(value)
    key = re.sub(r"\buniv\b", "university", key)
    key = re.sub(r"\bst\b", "state", key)
    key = key.replace("saint", "st")
    key = re.sub(r"\s+", " ", key).strip()
    return TEAM_ALIASES.get(key, key)


def text_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def display_to_first_last(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if "," in text:
        last, first = [part.strip() for part in text.split(",", 1)]
        return f"{first} {last}".strip()
    return text


def name_keys(value: object) -> set[str]:
    text = "" if pd.isna(value) else str(value).strip()
    candidates = {text, display_to_first_last(text)}
    out = set()
    for candidate in candidates:
        key = norm_text(candidate)
        parts = [p for p in key.split() if p not in {"jr", "sr", "ii", "iii", "iv", "v"}]
        if parts:
            out.add(" ".join(parts))
        if len(parts) > 2:
            out.add(f"{parts[0]} {parts[-1]}")
    expanded = set(out)
    for key in out:
        expanded.update(NAME_ALIASES.get(key, set()))
    out = expanded
    return out


def season_end_year(season: object) -> float:
    text = "" if pd.isna(season) else str(season)
    match = re.search(r"20(\d{2})\s*-\s*(\d{2})", text)
    if match:
        return float(f"20{match.group(2)}")
    match = re.search(r"(\d{4})", text)
    return float(match.group(1)) if match else np.nan


def to_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.where(den.ne(0))
    return num / den


def fill_derived_d2_efficiency(df: pd.DataFrame) -> None:
    """Fill D2 efficiency basics from raw D2 box counts when joins miss."""
    raw_cols = [
        "d2_FGM",
        "d2_FGA",
        "d2_3PM",
        "d2_3PA",
        "d2_FTM",
        "d2_FTA",
        "d2_PTS",
        "d2_TRB",
        "d2_AST",
        "d2_TOV",
        "d2_STL",
        "d2_BLK",
        "d2_MIN",
    ]
    to_numeric(df, raw_cols)
    derived = {
        "d2_eFG": safe_div(df["d2_FGM"] + 0.5 * df["d2_3PM"], df["d2_FGA"]),
        "d2_TS_pct": safe_div(df["d2_PTS"], 2 * (df["d2_FGA"] + 0.44 * df["d2_FTA"])),
        "d2_FTR": safe_div(df["d2_FTA"], df["d2_FGA"]),
        "d2_three_share": safe_div(df["d2_3PA"], df["d2_FGA"]),
        "d2_AST_TOV": safe_div(df["d2_AST"], df["d2_TOV"]),
        "d2_three_pct": safe_div(df["d2_3PM"], df["d2_3PA"]),
        "d2_ft_pct": safe_div(df["d2_FTM"], df["d2_FTA"]),
        "d2_PTS_per_40": safe_div(40 * df["d2_PTS"], df["d2_MIN"]),
        "d2_TRB_per_40": safe_div(40 * df["d2_TRB"], df["d2_MIN"]),
        "d2_AST_per_40": safe_div(40 * df["d2_AST"], df["d2_MIN"]),
        "d2_TOV_per_40": safe_div(40 * df["d2_TOV"], df["d2_MIN"]),
        "d2_STL_per_40": safe_div(40 * df["d2_STL"], df["d2_MIN"]),
        "d2_BLK_per_40": safe_div(40 * df["d2_BLK"], df["d2_MIN"]),
    }
    for col, values in derived.items():
        if col in df.columns:
            df[col] = df[col].fillna(values)
        else:
            df[col] = values


ALT_ADVANCED_COL_MAP = {
    "usg": "d2_usg",
    "ORB_pct": "d2_ORB_pct",
    "DRB_pct": "d2_DRB_pct",
    "AST_pct": "d2_AST_pct",
    "TOV_pct": "d2_TOV_pct",
    "Stl_pct": "d2_Stl_pct",
    "Blk_pct": "d2_Blk_pct",
    "eFG": "d2_eFG",
    "TS_pct": "d2_TS_pct",
    "FTR": "d2_FTR",
    "three_share": "d2_three_share",
    "AST_TOV": "d2_AST_TOV",
    "3P_pct": "d2_three_pct",
    "FT_pct": "d2_ft_pct",
}

ALT_ADVANCED_RENAME_MAP = {
    "player_name": "player",
    "eFG_pct": "eFG",
    "USG_pct": "usg",
    "BLK_pct": "Blk_pct",
    "STL_pct": "Stl_pct",
}

ARCHETYPE_VECTOR_COLS = [
    "k6_drop7_8_archetype_confidence",
    "k6_drop7_8_kept_mass_from_k8_1_to_6",
    "k6_drop7_8_removed_mass_from_k8_7_to_8",
    "k6_drop7_8_archetype_1_weight",
    "k6_drop7_8_archetype_2_weight",
    "k6_drop7_8_archetype_3_weight",
    "k6_drop7_8_archetype_4_weight",
    "k6_drop7_8_archetype_5_weight",
    "k6_drop7_8_archetype_6_weight",
]


def add_archetype_vectors(
    out: pd.DataFrame,
    vector_path: Path,
    archetype_mode: str = "drop7_8_renormalized",
) -> pd.DataFrame:
    column_map = ARCHETYPE_COLUMN_MAPS[archetype_mode]
    out["d2_archetype_match_method"] = "unmatched"
    out["d2_archetype_vector_missing"] = 1.0
    for col in ARCHETYPE_VECTOR_COLS:
        out[f"d2_{col}"] = np.nan
    if not vector_path.exists():
        out.attrs["d2_archetype_vector_matches"] = 0
        return out

    vectors = pd.read_csv(vector_path, low_memory=False)
    vectors = vectors.loc[vectors["division"].eq("D2")].copy()
    vectors["season_end_join"] = vectors["season"].map(season_end_year)
    vectors["team_key_join"] = vectors["team"].map(team_key)
    vectors["_name_keys"] = vectors["player_name"].map(name_keys)
    to_numeric(vectors, list(column_map.values()))

    by_name_team_season: dict[tuple[str, str, float], list[int]] = {}
    by_name_season: dict[tuple[str, float], list[int]] = {}
    by_name_team: dict[tuple[str, str], list[int]] = {}
    for idx, row in vectors.iterrows():
        season = row["season_end_join"]
        if pd.isna(season):
            continue
        team = row["team_key_join"]
        for name_key in row["_name_keys"]:
            by_name_team_season.setdefault((name_key, team, season), []).append(idx)
            by_name_season.setdefault((name_key, season), []).append(idx)
            by_name_team.setdefault((name_key, team), []).append(idx)

    matched = 0
    for out_idx, row in out.iterrows():
        season = row.get("d2_season_end_join")
        if pd.isna(season):
            continue
        transfer_team = team_key(row.get("d2_team"))
        candidates = []
        for name_key in name_keys(row.get("d2_player_name")):
            candidates.extend(by_name_team_season.get((name_key, transfer_team, season), []))
        candidates = sorted(set(candidates))
        match_method = "archetype_exact_name_team_season"

        if len(candidates) != 1:
            fallback = []
            for name_key in name_keys(row.get("d2_player_name")):
                fallback.extend(by_name_season.get((name_key, season), []))
            fallback = sorted(set(fallback))
            fallback_teams = {vectors.at[idx, "team_key_join"] for idx in fallback}
            if len(fallback) == 1 or len(fallback_teams) == 1:
                candidates = fallback
                match_method = "archetype_unique_name_season"

        if len(candidates) != 1:
            same_team = []
            for name_key in name_keys(row.get("d2_player_name")):
                same_team.extend(by_name_team.get((name_key, transfer_team), []))
            same_team = sorted(set(same_team))
            if same_team and row.get("d2_source_family") == "older_school_realgm":
                candidates = sorted(
                    same_team,
                    key=lambda idx: (
                        abs(vectors.at[idx, "season_end_join"] - season)
                        if pd.notna(vectors.at[idx, "season_end_join"])
                        else 99
                    ),
                )[:1]
                match_method = "archetype_same_player_team_any_season"

        if len(candidates) != 1:
            continue

        vector_row = vectors.loc[candidates[0]]
        for col in ARCHETYPE_VECTOR_COLS:
            out.at[out_idx, f"d2_{col}"] = pd.to_numeric(vector_row[column_map[col]], errors="coerce")
        out.at[out_idx, "d2_archetype_vector_missing"] = 0.0
        out.at[out_idx, "d2_archetype_match_method"] = match_method
        matched += 1

    out.attrs["d2_archetype_vector_matches"] = matched
    return out


def fill_alt_d2_advanced(out: pd.DataFrame, alt_path: Path) -> pd.DataFrame:
    source_paths = [
        (Path(SUPPLEMENTAL_D2_ADVANCED_PATH), "step26_height_class_advanced_export"),
        (Path(WORKING_D2_ADVANCED_PATH), "step18_working_advanced_export"),
        (Path(WEBSITE_PRIORITY_D2_ADVANCED_PATH), "website_priority_advanced_export"),
        (Path(WEBSITE_MIN5_HEIGHTS_D2_ADVANCED_PATH), "website_priority_min5_heights_advanced_export"),
        (alt_path, "d2_player_seasons_possession_export"),
    ]
    existing_source_paths = [(path, label) for path, label in source_paths if path.exists()]
    if not existing_source_paths:
        out["d2_advanced_match_method"] = np.where(out["d2_usg"].notna(), "combined_exact", "unmatched")
        out["d2_advanced_source"] = np.where(out["d2_usg"].notna(), "combined_player_database", "")
        return out

    frames = []
    for priority, (path, label) in enumerate(existing_source_paths):
        alt_file = pd.read_csv(path, low_memory=False).rename(columns=ALT_ADVANCED_RENAME_MAP)
        if "player" not in alt_file.columns or "team" not in alt_file.columns or "season" not in alt_file.columns:
            continue
        alt_file["_advanced_source_label"] = label
        alt_file["_advanced_source_priority"] = priority
        frames.append(alt_file)
    if not frames:
        out["d2_advanced_match_method"] = np.where(out["d2_usg"].notna(), "combined_exact", "unmatched")
        out["d2_advanced_source"] = np.where(out["d2_usg"].notna(), "combined_player_database", "")
        return out
    alt = pd.concat(frames, ignore_index=True, sort=False)
    alt["season_end_join"] = alt["season"].map(season_end_year)
    alt["team_key_join"] = alt["team"].map(team_key)
    alt["_name_keys"] = alt["player"].map(name_keys)

    by_name_team_season: dict[tuple[str, str, float], list[int]] = {}
    by_name_season: dict[tuple[str, float], list[int]] = {}
    by_name_team: dict[tuple[str, str], list[int]] = {}
    for idx, row in alt.iterrows():
        season = row["season_end_join"]
        if pd.isna(season):
            continue
        team = row["team_key_join"]
        for name_key in row["_name_keys"]:
            by_name_team_season.setdefault((name_key, team, season), []).append(idx)
            by_name_season.setdefault((name_key, season), []).append(idx)
            by_name_team.setdefault((name_key, team), []).append(idx)

    source = pd.Series("", index=out.index, dtype=object)
    method = pd.Series("unmatched", index=out.index, dtype=object)
    source.loc[out["d2_usg"].notna()] = "combined_player_database"
    method.loc[out["d2_usg"].notna()] = "combined_exact"
    matched_alt = 0
    matched_fallback = 0

    for out_idx, row in out.iterrows():
        needs = any(pd.isna(row.get(dst)) for dst in ALT_ADVANCED_COL_MAP.values() if dst in out.columns)
        if not needs:
            continue
        season = row.get("d2_season_end_join")
        if pd.isna(season):
            continue
        transfer_team = team_key(row.get("d2_team"))
        candidates = []
        for name_key in name_keys(row.get("d2_player_name")):
            candidates.extend(by_name_team_season.get((name_key, transfer_team, season), []))
        candidates = sorted(set(candidates))
        match_method = "alt_exact_name_team_season"
        if len(candidates) >= 1:
            candidates = [sorted(candidates, key=lambda idx: alt.at[idx, "_advanced_source_priority"])[0]]
        else:
            fallback = []
            for name_key in name_keys(row.get("d2_player_name")):
                fallback.extend(by_name_season.get((name_key, season), []))
            fallback = sorted(set(fallback))
            fallback_teams = {alt.at[idx, "team_key_join"] for idx in fallback}
            if len(fallback) == 1 or len(fallback_teams) == 1:
                candidates = [sorted(fallback, key=lambda idx: alt.at[idx, "_advanced_source_priority"])[0]]
                match_method = "alt_unique_name_season"
        if len(candidates) != 1:
            transfer_names = name_keys(row.get("d2_player_name"))
            scored = []
            season_rows = alt.index[alt["season_end_join"].eq(season)].tolist()
            for idx in season_rows:
                alt_row = alt.loc[idx]
                name_score = max(
                    (text_ratio(name_key, alt_name_key) for name_key in transfer_names for alt_name_key in alt_row["_name_keys"]),
                    default=0.0,
                )
                team_score = text_ratio(transfer_team, alt_row["team_key_join"])
                combo = 0.72 * name_score + 0.28 * team_score
                scored.append((combo, name_score, team_score, idx))
            scored.sort(reverse=True)
            if scored:
                top = scored[0]
                second_combo = scored[1][0] if len(scored) > 1 else 0.0
                if (
                    top[1] >= 0.80
                    and top[2] >= 0.78
                    and top[0] - second_combo >= 0.025
                ):
                    candidates = [top[3]]
                    match_method = "alt_fuzzy_name_team_season"
        if len(candidates) != 1:
            same_team = []
            for name_key in name_keys(row.get("d2_player_name")):
                same_team.extend(by_name_team.get((name_key, transfer_team), []))
            same_team = [
                idx
                for idx in sorted(set(same_team))
                if "usg" in alt.columns and pd.notna(pd.to_numeric(alt.at[idx, "usg"], errors="coerce"))
            ]
            if same_team:
                same_team = sorted(
                    same_team,
                    key=lambda idx: (
                        abs(alt.at[idx, "season_end_join"] - season)
                        if pd.notna(alt.at[idx, "season_end_join"])
                        else 99,
                        alt.at[idx, "_advanced_source_priority"],
                    ),
                )
                advanced_signature_cols = [src for src in ALT_ADVANCED_COL_MAP if src in alt.columns]
                signatures = {
                    tuple(
                        round(float(pd.to_numeric(alt.at[idx, col], errors="coerce")), 6)
                        if pd.notna(pd.to_numeric(alt.at[idx, col], errors="coerce"))
                        else np.nan
                        for col in advanced_signature_cols
                    )
                    for idx in same_team
                }
                if len(signatures) == 1 or row.get("d2_source_family") == "older_school_realgm":
                    candidates = [same_team[0]]
                    match_method = "alt_same_player_team_any_season"
        if len(candidates) != 1:
            continue

        alt_row = alt.loc[candidates[0]]
        for src, dst in ALT_ADVANCED_COL_MAP.items():
            if dst in out.columns and src in alt.columns and pd.isna(out.at[out_idx, dst]):
                out.at[out_idx, dst] = pd.to_numeric(alt_row[src], errors="coerce")
        source.at[out_idx] = alt_row.get("_advanced_source_label", "d2_advanced_supplement")
        method.at[out_idx] = match_method
        if match_method == "alt_unique_name_season":
            matched_fallback += 1
        else:
            matched_alt += 1

    out["d2_advanced_match_method"] = method
    out["d2_advanced_source"] = source
    out.attrs["alt_advanced_exact_matches"] = matched_alt
    out.attrs["alt_advanced_unique_name_matches"] = matched_fallback
    return out


def load_and_join(
    transfers_path: Path,
    combined_path: Path,
    alt_d2_advanced_path: Path,
    archetype_vector_path: Path | None = None,
    archetype_mode: str = "drop7_8_renormalized",
) -> pd.DataFrame:
    transfers = pd.read_csv(transfers_path, low_memory=False)
    combined = pd.read_csv(combined_path, low_memory=False)

    transfers["d2_name_key_join"] = transfers["d2_player_name"].map(norm_text)
    transfers["d2_team_key_join"] = transfers["d2_team"].map(team_key)
    transfers["d2_season_end_join"] = transfers["d2_season"].map(season_end_year)
    transfers["d1_pid_join"] = pd.to_numeric(transfers["d1_pid"], errors="coerce")
    transfers["d1_season_end_join"] = transfers["next_d1_season"].map(season_end_year)

    combined["name_key_join"] = combined["player_name"].map(norm_text)
    combined["team_key_join"] = combined["team"].map(team_key)
    combined["season_end_join"] = combined["season"].map(season_end_year)
    combined["pid_join"] = pd.to_numeric(combined.get("pid"), errors="coerce")

    d1_cols = [
        "pid_join",
        "season_end_join",
        "PORPAG",
        "D_PORPAG",
        "bpm",
        "obpm",
        "dbpm",
        "gbpm",
        "ogbpm",
        "dgbpm",
    ]
    d1 = (
        combined.loc[combined["division"].eq("D1"), d1_cols]
        .dropna(subset=["pid_join", "season_end_join"])
        .drop_duplicates(["pid_join", "season_end_join"], keep="first")
        .rename(columns={col: f"d1_{col}" for col in d1_cols if col not in {"pid_join", "season_end_join"}})
    )
    out = transfers.merge(
        d1,
        left_on=["d1_pid_join", "d1_season_end_join"],
        right_on=["pid_join", "season_end_join"],
        how="left",
    ).drop(columns=["pid_join", "season_end_join"], errors="ignore")
    if "d1_bpm_x" in out.columns:
        out = out.rename(columns={"d1_bpm_x": "d1_bpm", "d1_bpm_y": "d1_bart_bpm"})
    out["d1_conf_tier"] = out["d1_conf"].map(d1_conf_tier)

    d2_cols = [
        "name_key_join",
        "team_key_join",
        "season_end_join",
        "height_inches",
        "eFG",
        "TS_pct",
        "FTR",
        "three_share",
        "AST_TOV",
        "TOV_pct",
        "AST_pct",
        "usg",
        "ORB_pct",
        "DRB_pct",
        "Blk_pct",
        "Stl_pct",
        "PTS_per_40",
        "ORB_per_40",
        "DRB_per_40",
        "TRB_per_40",
        "AST_per_40",
        "TOV_per_40",
        "STL_per_40",
        "BLK_per_40",
        "three_pct",
        "ft_pct",
        "three_pa_per_100_team_poss_model",
    ]
    d2 = (
        combined.loc[combined["division"].eq("D2"), d2_cols]
        .dropna(subset=["name_key_join", "team_key_join", "season_end_join"])
        .drop_duplicates(["name_key_join", "team_key_join", "season_end_join"], keep="first")
        .rename(
            columns={
                col: f"d2_{col}"
                for col in d2_cols
                if col not in {"name_key_join", "team_key_join", "season_end_join"}
            }
        )
    )
    out = out.merge(
        d2,
        left_on=["d2_name_key_join", "d2_team_key_join", "d2_season_end_join"],
        right_on=["name_key_join", "team_key_join", "season_end_join"],
        how="left",
    ).drop(columns=["name_key_join", "team_key_join", "season_end_join"], errors="ignore")

    numeric_cols = (
        D2_NUMERIC_FEATURES
        + CONTEXT_NUMERIC_FEATURES
        + WITH_MINUTES_FEATURES
        + list(TARGETS.values())
        + ["d1_D_PORPAG", "d1_obpm", "d1_dbpm", "d1_gbpm", "d1_ogbpm", "d1_dgbpm"]
    )
    to_numeric(out, numeric_cols)
    out = fill_alt_d2_advanced(out, alt_d2_advanced_path)
    fill_derived_d2_efficiency(out)
    if archetype_vector_path is not None:
        out = add_archetype_vectors(out, archetype_vector_path, archetype_mode)
    return out


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=3, sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def models(random_state: int) -> dict[str, object]:
    return {
        "mean": None,
        "ridge": Ridge(alpha=10.0),
        "elasticnet": ElasticNet(alpha=0.03, l1_ratio=0.25, max_iter=20000, random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=random_state, l2_regularization=0.1),
        "knn": KNeighborsRegressor(n_neighbors=15, weights="distance"),
        "svr_rbf": SVR(C=2.0, epsilon=0.2, gamma="scale"),
    }


def d2_numeric_features(drop_d2_minutes: bool = False) -> list[str]:
    if not drop_d2_minutes:
        return D2_NUMERIC_FEATURES
    return [col for col in D2_NUMERIC_FEATURES if col not in D2_MINUTE_FEATURES]


def feature_sets(
    df: pd.DataFrame,
    drop_d2_minutes: bool = False,
    context_mode: str = "full",
) -> dict[str, dict[str, list[str]]]:
    if context_mode == "conf_tier_only":
        numeric_context = []
        categorical_candidates = CONFERENCE_TIER_CATEGORICAL_FEATURES
    else:
        numeric_context = CONTEXT_NUMERIC_FEATURES
        categorical_candidates = CATEGORICAL_FEATURES

    numeric_base = [
        col
        for col in d2_numeric_features(drop_d2_minutes) + numeric_context + ARCHETYPE_NUMERIC_FEATURES
        if col in df.columns
    ]
    categorical = [col for col in categorical_candidates if col in df.columns and df[col].notna().any()]
    return {
        "no_minutes": {"numeric": numeric_base, "categorical": categorical},
        "with_minutes": {
            "numeric": numeric_base + [col for col in WITH_MINUTES_FEATURES if col in df.columns],
            "categorical": categorical,
        },
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) == 0:
        return {key: np.nan for key in ["n", "mae", "rmse", "r2", "pearson", "spearman"]}
    pearson = stats.pearsonr(y_true, y_pred).statistic if len(y_true) >= 2 and np.std(y_pred) > 0 else np.nan
    spearman = stats.spearmanr(y_true, y_pred).statistic if len(y_true) >= 2 and np.std(y_pred) > 0 else np.nan
    return {
        "n": float(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 and np.std(y_true) > 0 else np.nan,
        "pearson": float(pearson) if np.isfinite(pearson) else np.nan,
        "spearman": float(spearman) if np.isfinite(spearman) else np.nan,
    }


def calibration_rows(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, context: dict) -> list[dict]:
    work = meta.copy()
    work["actual"] = y_true
    work["pred"] = y_pred
    work = work[np.isfinite(work["actual"]) & np.isfinite(work["pred"])].copy()
    if len(work) < 5:
        return []
    bins = min(10, max(2, len(work) // 8))
    work["pred_bucket"] = pd.qcut(work["pred"].rank(method="first"), q=bins, labels=False, duplicates="drop") + 1
    rows = []
    for bucket, group in work.groupby("pred_bucket", dropna=True):
        row = dict(context)
        row.update(
            {
                "bucket": int(bucket),
                "n": len(group),
                "pred_min": group["pred"].min(),
                "pred_max": group["pred"].max(),
                "pred_mean": group["pred"].mean(),
                "actual_mean": group["actual"].mean(),
                "actual_median": group["actual"].median(),
                "bias_actual_minus_pred": (group["actual"] - group["pred"]).mean(),
            }
        )
        rows.append(row)
    return rows


def split_masks(df: pd.DataFrame, target_col: str) -> tuple[pd.Series, pd.Series, str]:
    available = df[target_col].notna()
    holdout = df["next_d1_season"].eq("2025-26")
    if (available & holdout).sum() >= 15 and (available & ~holdout).sum() >= 40:
        return available & ~holdout, available & holdout, "season_holdout_2025_26"
    seasons = sorted(df.loc[available, "next_d1_season"].dropna().unique())
    test_season = seasons[-1]
    return available & df["next_d1_season"].ne(test_season), available & df["next_d1_season"].eq(test_season), f"season_holdout_{test_season}"


def train_single_target(
    df: pd.DataFrame,
    out_dir: Path,
    random_state: int,
    drop_d2_minutes: bool = False,
    context_mode: str = "full",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = []
    predictions = []
    calibration = []
    feature_def = feature_sets(df, drop_d2_minutes, context_mode)
    model_defs = models(random_state)

    for target_name, target_col in TARGETS.items():
        train_mask, test_mask, split_name = split_masks(df, target_col)
        for regime, defs in feature_def.items():
            features = defs["numeric"] + defs["categorical"]
            train = df.loc[train_mask].copy()
            test = df.loc[test_mask].copy()
            X_train = train[features]
            y_train = train[target_col].to_numpy(dtype=float)
            X_test = test[features]
            y_test = test[target_col].to_numpy(dtype=float)

            for model_name, estimator in model_defs.items():
                if model_name == "mean":
                    pred = np.full_like(y_test, float(np.nanmean(y_train)), dtype=float)
                else:
                    pipe = Pipeline(
                        [
                            ("preprocess", make_preprocessor(defs["numeric"], defs["categorical"])),
                            ("model", estimator),
                        ]
                    )
                    pipe.fit(X_train, y_train)
                    pred = pipe.predict(X_test)

                score = metrics(y_test, pred)
                result = {
                    "target": target_name,
                    "target_col": target_col,
                    "regime": regime,
                    "model": model_name,
                    "split": split_name,
                    "train_n": int(len(train)),
                    "test_n": int(len(test)),
                    "numeric_feature_count": len(defs["numeric"]),
                    "categorical_feature_count": len(defs["categorical"]),
                }
                result.update(score)
                results.append(result)

                pred_frame = test[
                    [
                        "d2_player_name",
                        "d2_team",
                        "d2_season",
                        "d1_player_name",
                        "d1_team",
                        "next_d1_season",
                    ]
                ].copy()
                pred_frame["target"] = target_name
                pred_frame["regime"] = regime
                pred_frame["model"] = model_name
                pred_frame["actual"] = y_test
                pred_frame["predicted"] = pred
                pred_frame["residual_actual_minus_pred"] = y_test - pred
                predictions.append(pred_frame)

                calibration.extend(
                    calibration_rows(
                        test[["d1_player_name", "d1_team", "next_d1_season"]],
                        y_test,
                        pred,
                        {"target": target_name, "regime": regime, "model": model_name, "split": split_name},
                    )
                )

    return pd.DataFrame(results), pd.concat(predictions, ignore_index=True), pd.DataFrame(calibration)


def train_multi_output(
    df: pd.DataFrame,
    random_state: int,
    drop_d2_minutes: bool = False,
    context_mode: str = "full",
) -> pd.DataFrame:
    all_targets = list(TARGETS.values())
    available = df[all_targets].notna().all(axis=1)
    holdout = df["next_d1_season"].eq("2025-26")
    if (available & holdout).sum() < 15 or (available & ~holdout).sum() < 40:
        return pd.DataFrame()

    rows = []
    feature_def = feature_sets(df, drop_d2_minutes, context_mode)
    multi_models = {
        "multi_ridge": Ridge(alpha=10.0),
        "multi_random_forest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=-1,
        ),
        "multi_extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=-1,
        ),
        "multi_gradient_boosting": MultiOutputRegressor(GradientBoostingRegressor(random_state=random_state)),
    }
    for regime, defs in feature_def.items():
        features = defs["numeric"] + defs["categorical"]
        train = df.loc[available & ~holdout]
        test = df.loc[available & holdout]
        X_train = train[features]
        Y_train = train[all_targets].to_numpy(dtype=float)
        X_test = test[features]
        Y_test = test[all_targets].to_numpy(dtype=float)
        for model_name, estimator in multi_models.items():
            pipe = Pipeline(
                [
                    ("preprocess", make_preprocessor(defs["numeric"], defs["categorical"])),
                    ("model", estimator),
                ]
            )
            pipe.fit(X_train, Y_train)
            pred = pipe.predict(X_test)
            for idx, (target_name, target_col) in enumerate(TARGETS.items()):
                row = {
                    "target": target_name,
                    "target_col": target_col,
                    "regime": regime,
                    "model": model_name,
                    "split": "multi_output_season_holdout_2025_26",
                    "train_n": len(train),
                    "test_n": len(test),
                    "numeric_feature_count": len(defs["numeric"]),
                    "categorical_feature_count": len(defs["categorical"]),
                }
                row.update(metrics(Y_test[:, idx], pred[:, idx]))
                rows.append(row)
    return pd.DataFrame(rows)


def feature_coverage(
    df: pd.DataFrame,
    out_dir: Path,
    drop_d2_minutes: bool = False,
    context_mode: str = "full",
) -> None:
    numeric_context = [] if context_mode == "conf_tier_only" else CONTEXT_NUMERIC_FEATURES
    cols = [
        col
        for col in d2_numeric_features(drop_d2_minutes)
        + numeric_context
        + ARCHETYPE_NUMERIC_FEATURES
        + WITH_MINUTES_FEATURES
        + list(TARGETS.values())
        if col in df.columns
    ]
    rows = []
    for col in cols:
        rows.append(
            {
                "column": col,
                "non_null": int(df[col].notna().sum()),
                "missing": int(df[col].isna().sum()),
                "missing_pct": float(df[col].isna().mean()),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "feature_target_coverage.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-ready", default=MODEL_READY_PATH, help="Use an existing curated model-ready table instead of rebuilding joins.")
    parser.add_argument("--transfers", default=TRANSFER_PATH)
    parser.add_argument("--combined", default=COMBINED_PATH)
    parser.add_argument("--alt-d2-advanced", default=ALT_D2_ADVANCED_PATH)
    parser.add_argument("--archetype-vectors", default=ARCHETYPE_VECTOR_PATH)
    parser.add_argument("--include-archetypes", action="store_true")
    parser.add_argument(
        "--archetype-mode",
        choices=sorted(ARCHETYPE_COLUMN_MAPS),
        default="drop7_8_renormalized",
        help="Which archetype columns to pull from --archetype-vectors.",
    )
    parser.add_argument(
        "--drop-d2-minutes",
        action="store_true",
        help="Remove raw D2 minute features d2_MIN and d2_MPG from model inputs.",
    )
    parser.add_argument(
        "--context-mode",
        choices=["full", "conf_tier_only"],
        default="full",
        help="Use all historical context categoricals, or only destination d1_conf_tier.",
    )
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model_ready:
        df = pd.read_csv(args.model_ready, low_memory=False)
    else:
        df = load_and_join(
            Path(args.transfers),
            Path(args.combined),
            Path(args.alt_d2_advanced),
            Path(args.archetype_vectors) if args.include_archetypes else None,
            args.archetype_mode,
        )
    model_ready = out_dir / "baseline_transfer_impact_model_ready.csv"
    df.to_csv(model_ready, index=False)
    feature_coverage(df, out_dir, args.drop_d2_minutes, args.context_mode)

    results, predictions, calibration = train_single_target(
        df,
        out_dir,
        args.random_state,
        args.drop_d2_minutes,
        args.context_mode,
    )
    multi_results = train_multi_output(df, args.random_state, args.drop_d2_minutes, args.context_mode)
    if not multi_results.empty:
        all_results = pd.concat([results, multi_results], ignore_index=True)
    else:
        all_results = results

    all_results.to_csv(out_dir / "baseline_impact_model_results.csv", index=False)
    predictions.to_csv(out_dir / "baseline_impact_model_predictions.csv", index=False)
    calibration.to_csv(out_dir / "baseline_impact_model_calibration_deciles.csv", index=False)

    summary = {
        "rows": int(len(df)),
        "targets_non_null": {name: int(df[col].notna().sum()) for name, col in TARGETS.items()},
        "model_ready": str(model_ready),
        "results": str(out_dir / "baseline_impact_model_results.csv"),
        "predictions": str(out_dir / "baseline_impact_model_predictions.csv"),
        "calibration": str(out_dir / "baseline_impact_model_calibration_deciles.csv"),
        "alt_d2_advanced_exact_matches": int(df.attrs.get("alt_advanced_exact_matches", 0)),
        "alt_d2_advanced_unique_name_matches": int(df.attrs.get("alt_advanced_unique_name_matches", 0)),
        "include_archetypes": bool(args.include_archetypes),
        "archetype_mode": args.archetype_mode if args.include_archetypes else None,
        "drop_d2_minutes": bool(args.drop_d2_minutes),
        "context_mode": args.context_mode,
        "d2_minute_features_removed": sorted(D2_MINUTE_FEATURES) if args.drop_d2_minutes else [],
        "d2_archetype_vector_matches": int(df.attrs.get("d2_archetype_vector_matches", 0)),
    }
    (out_dir / "baseline_impact_model_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nBest by target/regime using Spearman then MAE:")
    display = all_results.sort_values(["target", "regime", "spearman", "mae"], ascending=[True, True, False, True])
    best = display.groupby(["target", "regime"], as_index=False).head(1)
    print(best[["target", "regime", "model", "test_n", "mae", "rmse", "r2", "pearson", "spearman"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
