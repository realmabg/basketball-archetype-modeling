#!/usr/bin/env python3
"""Reproject no-minutes k8 archetypes after removing junk archetypes 5 and 7.

This keeps the original no-minutes k8 centers for archetypes 1, 2, 3, 4, 6,
and 8 fixed, removes centers 5 and 7, and recomputes each player-season's
nonnegative simplex weights using only the retained six centers.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


RUN_ID = "k8__pooled_zscore__no_height_no_minutes_no_efficiency__method-nnls__init-uniform__ninit-1"
ROOT = Path("combined_d1_d2_exports/archetype_experiments_step43_no_minutes_full_k8_nnls_vectors")
INPUT = ROOT / f"{RUN_ID}__all_d1_d2_player_seasons_k8_no_minutes_vectors.csv"
PROFILES = ROOT / "profiles" / f"{RUN_ID}__profiles.csv"
SCALING = ROOT / "scaling" / f"{RUN_ID}__scaling.csv"
OUTPUT = ROOT / f"{RUN_ID}__all_d1_d2_player_seasons_k8_no_minutes_and_fixed_k6_drop5_7_vectors.csv"
SUMMARY = ROOT / f"{RUN_ID}__fixed_k6_drop5_7_reprojection_summary.json"

FEATURES = [
    "FTR",
    "three_share",
    "three_pa_per_100_team_poss",
    "usg",
    "tov_pct",
    "ast_pct",
    "orb_pct",
    "drb_pct",
    "blk_pct",
    "stl_pct",
    "ast_tov",
]

RETAINED_SOURCE_IDS = [1, 2, 3, 4, 6, 8]
REMOVED_SOURCE_IDS = [5, 7]
RETAINED_LABELS = {
    1: "low_usage_connector",
    2: "rim_protecting_big",
    3: "lead_guard",
    4: "defensive_spacer",
    6: "scoring_big",
    8: "pure_shooter",
}


def scaled_feature_matrix(df: pd.DataFrame, scaling: pd.DataFrame) -> np.ndarray:
    scale = scaling.set_index("feature").loc[FEATURES]
    raw = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if raw.isna().any().any():
        missing = raw.isna().sum()
        raise ValueError(f"Missing/non-numeric features:\n{missing[missing > 0]}")

    clipped = raw.clip(
        lower=scale["clip_p01"],
        upper=scale["clip_p99"],
        axis="columns",
    )
    z = (clipped - scale["mean_after_clip"]) / scale["std_after_clip"].replace(0, 1)
    return z.to_numpy(dtype=np.float64)


def retained_centers(profiles: pd.DataFrame) -> np.ndarray:
    centers = []
    for source_id in RETAINED_SOURCE_IDS:
        row = profiles.loc[profiles["archetype_id"].eq(source_id)]
        if row.empty:
            raise ValueError(f"Missing retained archetype {source_id}")
        centers.append([float(row.iloc[0][f"{feature}_archetype_z"]) for feature in FEATURES])
    return np.asarray(centers, dtype=np.float64)


def precompute_subset_solvers(centers: np.ndarray) -> list[tuple[tuple[int, ...], np.ndarray, np.ndarray]]:
    solvers = []
    for size in range(1, centers.shape[0] + 1):
        for subset in combinations(range(centers.shape[0]), size):
            active_centers = centers[list(subset)]
            gram = 2.0 * (active_centers @ active_centers.T)
            kkt = np.empty((size + 1, size + 1), dtype=np.float64)
            kkt[:size, :size] = gram
            kkt[:size, size] = 1.0
            kkt[size, :size] = 1.0
            kkt[size, size] = 0.0
            solvers.append((subset, active_centers, np.linalg.pinv(kkt)))
    return solvers


def project_to_fixed_centers(x: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    best_sse = np.full(n, np.inf, dtype=np.float64)
    best_weights = np.zeros((n, centers.shape[0]), dtype=np.float64)
    solvers = precompute_subset_solvers(centers)
    ones = np.ones((1, n), dtype=np.float64)

    for subset, active_centers, inv in solvers:
        size = len(subset)
        rhs = np.vstack([2.0 * active_centers @ x.T, ones])
        active_weights = (inv @ rhs)[:size].T
        valid = (active_weights >= -1e-9).all(axis=1)
        if not valid.any():
            continue
        active_weights = np.clip(active_weights, 0.0, None)
        active_weights = active_weights / active_weights.sum(axis=1, keepdims=True)
        recon = active_weights @ active_centers
        sse = ((x - recon) ** 2).sum(axis=1)
        improved = valid & (sse < best_sse)
        if improved.any():
            full = np.zeros((improved.sum(), centers.shape[0]), dtype=np.float64)
            full[:, list(subset)] = active_weights[improved]
            best_weights[improved] = full
            best_sse[improved] = sse[improved]

    if not np.isfinite(best_sse).all():
        raise RuntimeError("Projection failed for at least one row.")
    return best_weights, best_sse


def main() -> int:
    df = pd.read_csv(INPUT, low_memory=False)
    profiles = pd.read_csv(PROFILES)
    scaling = pd.read_csv(SCALING)

    x = scaled_feature_matrix(df, scaling)
    centers = retained_centers(profiles)
    weights, sse = project_to_fixed_centers(x, centers)
    dominant_idx = weights.argmax(axis=1)
    dominant_source_ids = np.asarray(RETAINED_SOURCE_IDS)[dominant_idx]

    out = df.copy()
    prefix = "k6_no_minutes_drop5_7"
    out[f"{prefix}_archetype_id"] = dominant_idx + 1
    out[f"{prefix}_source_k8_archetype_id"] = dominant_source_ids
    out[f"{prefix}_archetype_label"] = [RETAINED_LABELS[int(source_id)] for source_id in dominant_source_ids]
    out[f"{prefix}_archetype_confidence"] = weights.max(axis=1)
    out[f"{prefix}_projection_sse"] = sse

    retained_weight_cols = [f"k8_no_minutes_archetype_{idx}_weight" for idx in RETAINED_SOURCE_IDS]
    removed_weight_cols = [f"k8_no_minutes_archetype_{idx}_weight" for idx in REMOVED_SOURCE_IDS]
    out[f"{prefix}_kept_mass_from_k8_nonjunk"] = out[retained_weight_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    out[f"{prefix}_removed_mass_from_k8_5_7"] = out[removed_weight_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    for new_idx, source_id in enumerate(RETAINED_SOURCE_IDS, start=1):
        label = RETAINED_LABELS[source_id]
        out[f"{prefix}_archetype_{new_idx}_source_k8_{source_id}_{label}_weight"] = weights[:, new_idx - 1]

    out.to_csv(OUTPUT, index=False)

    summary = {
        "input": str(INPUT),
        "profiles": str(PROFILES),
        "scaling": str(SCALING),
        "output": str(OUTPUT),
        "rows": int(len(out)),
        "features": FEATURES,
        "retained_source_k8_archetypes": RETAINED_SOURCE_IDS,
        "removed_source_k8_archetypes": REMOVED_SOURCE_IDS,
        "labels": {str(key): value for key, value in RETAINED_LABELS.items()},
        "max_weight_sum_error": float(np.abs(weights.sum(axis=1) - 1.0).max()),
        "min_weight": float(weights.min()),
        "mean_projection_sse": float(sse.mean()),
        "median_projection_sse": float(np.median(sse)),
        "dominant_counts_by_new_id": {
            str(idx): int((out[f"{prefix}_archetype_id"] == idx).sum()) for idx in range(1, 7)
        },
        "dominant_counts_by_source_k8_id": {
            str(source_id): int((out[f"{prefix}_source_k8_archetype_id"] == source_id).sum())
            for source_id in RETAINED_SOURCE_IDS
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
