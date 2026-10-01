#!/usr/bin/env python3
"""Reproject player-seasons onto retained k8 archetypes 1-6.

This keeps the original k8 archetype centers 1-6 fixed, removes centers 7 and
8, and recomputes each player-season's nonnegative simplex weights using only
the retained six centers.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


RUN_ID = "k8__pooled_zscore__no_height_no_efficiency__method-pgd__init-uniform__ninit-1"
ROOT = Path("combined_d1_d2_exports/archetype_experiments_step43_sample3000_grid")
INPUT = ROOT / f"{RUN_ID}__all_d1_d2_player_seasons_k8_and_drop7_8_k6_vectors.csv"
PROFILES = ROOT / "profiles" / f"{RUN_ID}__profiles.csv"
SCALING = ROOT / "scaling" / f"{RUN_ID}__scaling.csv"
OUTPUT = ROOT / f"{RUN_ID}__all_d1_d2_player_seasons_k8_and_fixed_k6_reprojected_vectors.csv"
SUMMARY = ROOT / f"{RUN_ID}__fixed_k6_reprojection_summary.json"

FEATURES = [
    "mins_per_game",
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
    profiles = profiles.sort_values("archetype_id")
    centers = []
    for archetype_id in range(1, 7):
        row = profiles.loc[profiles["archetype_id"].eq(archetype_id)]
        if row.empty:
            raise ValueError(f"Missing retained archetype {archetype_id}")
        centers.append([float(row.iloc[0][f"{feature}_archetype_z"]) for feature in FEATURES])
    return np.asarray(centers, dtype=np.float64)


def precompute_subset_solvers(centers: np.ndarray) -> list[tuple[tuple[int, ...], np.ndarray, np.ndarray]]:
    """Return affine solvers for every nonempty active set."""
    solvers = []
    for size in range(1, centers.shape[0] + 1):
        for subset in combinations(range(centers.shape[0]), size):
            a = centers[list(subset)]
            gram = 2.0 * (a @ a.T)
            kkt = np.empty((size + 1, size + 1), dtype=np.float64)
            kkt[:size, :size] = gram
            kkt[:size, size] = 1.0
            kkt[size, :size] = 1.0
            kkt[size, size] = 0.0
            inv = np.linalg.pinv(kkt)
            solvers.append((subset, a, inv))
    return solvers


def project_to_fixed_centers(x: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project rows of x onto the convex hull of fixed centers."""
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
            best_sse[improved] = sse[improved]
            full = np.zeros((improved.sum(), centers.shape[0]), dtype=np.float64)
            full[:, list(subset)] = active_weights[improved]
            best_weights[improved] = full

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

    out = df.copy()
    out["k6_fixed_1to6_archetype_id"] = weights.argmax(axis=1) + 1
    out["k6_fixed_1to6_archetype_confidence"] = weights.max(axis=1)
    out["k6_fixed_1to6_projection_sse"] = sse
    out["k6_fixed_1to6_kept_mass_from_k8_1_to_6"] = out[
        [f"k8_archetype_{idx}_weight" for idx in range(1, 7)]
    ].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    out["k6_fixed_1to6_removed_mass_from_k8_7_to_8"] = out[
        ["k8_archetype_7_weight", "k8_archetype_8_weight"]
    ].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    for idx in range(1, 7):
        out[f"k6_fixed_1to6_archetype_{idx}_weight"] = weights[:, idx - 1]

    out.to_csv(OUTPUT, index=False)

    summary = {
        "input": str(INPUT),
        "profiles": str(PROFILES),
        "scaling": str(SCALING),
        "output": str(OUTPUT),
        "rows": int(len(out)),
        "features": FEATURES,
        "retained_archetypes": [1, 2, 3, 4, 5, 6],
        "removed_archetypes": [7, 8],
        "max_weight_sum_error": float(np.abs(weights.sum(axis=1) - 1.0).max()),
        "min_weight": float(weights.min()),
        "mean_projection_sse": float(sse.mean()),
        "median_projection_sse": float(np.median(sse)),
        "dominant_counts": {
            str(idx): int((out["k6_fixed_1to6_archetype_id"] == idx).sum()) for idx in range(1, 7)
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

