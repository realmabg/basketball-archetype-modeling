#!/usr/bin/env python3
"""Run archetypes-package experiments on the step43 feature table."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from archetypes import AA


INPUT_FILE = "data/archetype_model_features.csv"
OUTPUT_DIR = "models/player_archetypes/runs"

BASE_FEATURES = [
    "height_inches",
    "mins_per_game",
    "eFG",
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
    "three_pct",
    "ft_pct",
]

RESIDUAL_FEATURES = [
    "resid_three_pct_shot_mix",
    "resid_orb_pct_height",
    "resid_three_share_height",
    "resid_drb_pct_height",
]

EFFICIENCY_FEATURES = ["eFG", "three_pct", "ft_pct"]

FEATURE_SETS = {
    "full": BASE_FEATURES,
    "no_height": [feature for feature in BASE_FEATURES if feature != "height_inches"],
    "no_height_no_minutes": [
        feature for feature in BASE_FEATURES if feature not in {"height_inches", "mins_per_game"}
    ],
    "role_only_no_height": [
        "usg",
        "ast_pct",
        "tov_pct",
        "ast_tov",
        "FTR",
        "three_share",
        "three_pa_per_100_team_poss",
        "orb_pct",
        "drb_pct",
        "stl_pct",
        "blk_pct",
    ],
    "role_only_height_025": [
        "usg",
        "ast_pct",
        "tov_pct",
        "ast_tov",
        "FTR",
        "three_share",
        "three_pa_per_100_team_poss",
        "orb_pct",
        "drb_pct",
        "stl_pct",
        "blk_pct",
        "height_inches",
    ],
    "quality_lite_no_height": [
        "usg",
        "ast_pct",
        "tov_pct",
        "ast_tov",
        "FTR",
        "three_share",
        "three_pa_per_100_team_poss",
        "orb_pct",
        "drb_pct",
        "stl_pct",
        "blk_pct",
        "eFG",
        "three_pct",
        "ft_pct",
    ],
    "quality_lite_height_025": [
        "usg",
        "ast_pct",
        "tov_pct",
        "ast_tov",
        "FTR",
        "three_share",
        "three_pa_per_100_team_poss",
        "orb_pct",
        "drb_pct",
        "stl_pct",
        "blk_pct",
        "eFG",
        "three_pct",
        "ft_pct",
        "height_inches",
    ],
    "role_residual_height_025": [
        "usg",
        "ast_pct",
        "tov_pct",
        "ast_tov",
        "FTR",
        "three_share",
        "three_pa_per_100_team_poss",
        "orb_pct",
        "drb_pct",
        "stl_pct",
        "blk_pct",
        "height_inches",
        "resid_three_pct_shot_mix",
        "resid_orb_pct_height",
        "resid_three_share_height",
        "resid_drb_pct_height",
    ],
    "role_residual_context_only_height_025": [
        "usg",
        "ast_pct",
        "tov_pct",
        "ast_tov",
        "FTR",
        "three_pa_per_100_team_poss",
        "stl_pct",
        "blk_pct",
        "height_inches",
        "resid_three_pct_shot_mix",
        "resid_orb_pct_height",
        "resid_three_share_height",
        "resid_drb_pct_height",
    ],
    "quality_lite_residual_height_025": [
        "usg",
        "ast_pct",
        "tov_pct",
        "ast_tov",
        "FTR",
        "three_share",
        "three_pa_per_100_team_poss",
        "orb_pct",
        "drb_pct",
        "stl_pct",
        "blk_pct",
        "eFG",
        "three_pct",
        "ft_pct",
        "height_inches",
        "resid_three_pct_shot_mix",
        "resid_orb_pct_height",
        "resid_three_share_height",
        "resid_drb_pct_height",
    ],
    "quality_lite_residual_bucketed_height_025": [
        "usg",
        "ast_pct",
        "tov_pct",
        "ast_tov",
        "FTR",
        "three_share",
        "three_pa_per_100_team_poss",
        "resid_three_share_height",
        "orb_pct",
        "drb_pct",
        "resid_orb_pct_height",
        "resid_drb_pct_height",
        "stl_pct",
        "blk_pct",
        "eFG",
        "three_pct",
        "resid_three_pct_shot_mix",
        "ft_pct",
        "height_inches",
    ],
    "role_residual_bucketed_height_025": [
        "usg",
        "ast_pct",
        "tov_pct",
        "ast_tov",
        "FTR",
        "three_share",
        "three_pa_per_100_team_poss",
        "resid_three_share_height",
        "orb_pct",
        "drb_pct",
        "resid_orb_pct_height",
        "resid_drb_pct_height",
        "stl_pct",
        "blk_pct",
        "height_inches",
    ],
    "no_efficiency": [feature for feature in BASE_FEATURES if feature not in EFFICIENCY_FEATURES],
    "no_height_no_efficiency": [
        feature
        for feature in BASE_FEATURES
        if feature != "height_inches" and feature not in EFFICIENCY_FEATURES
    ],
    "no_height_no_minutes_no_efficiency": [
        feature
        for feature in BASE_FEATURES
        if feature not in {"height_inches", "mins_per_game"} and feature not in EFFICIENCY_FEATURES
    ],
}

ID_COLUMNS = [
    "division",
    "season",
    "team",
    "player_name",
    "first_name",
    "last_name",
    "player_uid",
    "player_season_key",
    "class_year",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--k-values", default="6,7,8,9,10")
    parser.add_argument("--scaling-modes", default="pooled_zscore,division_zscore")
    parser.add_argument(
        "--feature-sets",
        default="full,no_height,no_efficiency,no_height_no_efficiency",
    )
    parser.add_argument("--methods", default="nnls,pgd")
    parser.add_argument("--inits", default="uniform,furthest_sum")
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument(
        "--weights-file",
        default=None,
        help="Optional CSV with feature,z_multiplier columns. Model fits on z * z_multiplier.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional stratified sample size for search runs. 0 uses all rows.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--write-assignments",
        action="store_true",
        help="Write row-level assignment CSVs for every run. Profiles and diagnostics are always written.",
    )
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_manifest(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for k, scaling_mode, feature_set, method, init in itertools.product(
        [int(value) for value in split_csv(args.k_values)],
        split_csv(args.scaling_modes),
        split_csv(args.feature_sets),
        split_csv(args.methods),
        split_csv(args.inits),
    ):
        features = FEATURE_SETS[feature_set]
        run_id = (
            f"k{k}__{scaling_mode}__{feature_set}"
            f"__method-{method}__init-{init}__ninit-{args.n_init}"
        )
        rows.append(
            {
                "run_id": run_id,
                "k": k,
                "scaling_mode": scaling_mode,
                "feature_set": feature_set,
                "method": method,
                "init": init,
                "n_init": args.n_init,
                "max_iter": args.max_iter,
                "tol": args.tol,
                "features": "|".join(features),
                "excluded_features": "|".join(
                    feature for feature in BASE_FEATURES + RESIDUAL_FEATURES if feature not in features
                ),
            }
        )
    return pd.DataFrame(rows)


def zscore_features(df: pd.DataFrame, features: list[str], scaling_mode: str) -> tuple[np.ndarray, pd.DataFrame]:
    raw = df[features].apply(pd.to_numeric, errors="coerce")
    if raw.isna().any().any():
        missing = raw.isna().sum()
        raise ValueError(f"Missing/non-numeric feature values:\n{missing[missing > 0]}")

    rows = []
    scaled = pd.DataFrame(index=df.index, columns=features, dtype=float)
    groups = [("pooled", df.index)] if scaling_mode == "pooled_zscore" else df.groupby("division").groups.items()

    for group_name, group_index in groups:
        group_raw = raw.loc[group_index]
        lower = group_raw.quantile(0.01)
        upper = group_raw.quantile(0.99)
        clipped = group_raw.clip(lower=lower, upper=upper, axis="columns")
        mean = clipped.mean()
        std = clipped.std(ddof=0).replace(0, 1)
        scaled.loc[group_index, features] = (clipped - mean) / std
        for feature in features:
            rows.append(
                {
                    "scaling_group": group_name,
                    "feature": feature,
                    "clip_p01": lower[feature],
                    "clip_p99": upper[feature],
                    "mean_after_clip": mean[feature],
                    "std_after_clip": std[feature],
                }
            )

    return scaled.to_numpy(dtype=np.float64), pd.DataFrame(rows)


def load_feature_weights(path: str | None, features: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    if not path:
        weights = pd.DataFrame({"feature": features, "z_multiplier": np.ones(len(features))})
        return np.ones(len(features), dtype=np.float64), weights

    weights = pd.read_csv(path)
    required = {"feature", "z_multiplier"}
    missing_cols = required - set(weights.columns)
    if missing_cols:
        raise ValueError(f"Weights file missing required columns: {sorted(missing_cols)}")

    weights = weights[["feature", "z_multiplier"]].copy()
    missing_features = [feature for feature in features if feature not in set(weights["feature"])]
    if missing_features:
        raise ValueError(f"Weights file missing model features: {missing_features}")

    extra_features = sorted(set(weights["feature"]) - set(features))
    if extra_features:
        raise ValueError(f"Weights file contains features not in this model run: {extra_features}")

    weights = weights.set_index("feature").loc[features].reset_index()
    return weights["z_multiplier"].to_numpy(dtype=np.float64), weights


def feature_profiles(
    df: pd.DataFrame,
    features: list[str],
    labels: np.ndarray,
    archetypes: np.ndarray,
    coefficients: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for label in sorted(np.unique(labels)):
        mask = labels == label
        center = pd.Series(archetypes[label], index=features)
        top = center.sort_values(ascending=False).head(5)
        bottom = center.sort_values().head(5)
        row = {
            "archetype_id": int(label) + 1,
            "rows": int(mask.sum()),
            "share": float(mask.mean()),
            "d1_rows": int(((df["division"] == "D1") & mask).sum()),
            "d2_rows": int(((df["division"] == "D2") & mask).sum()),
            "mean_membership_weight": float(coefficients[mask, label].mean()) if mask.any() else np.nan,
            "top_z_features": "; ".join(f"{name}={value:.2f}" for name, value in top.items()),
            "bottom_z_features": "; ".join(f"{name}={value:.2f}" for name, value in bottom.items()),
        }
        for feature in features:
            row[f"{feature}_mean"] = float(df.loc[mask, feature].mean())
            row[f"{feature}_median"] = float(df.loc[mask, feature].median())
            row[f"{feature}_archetype_z"] = float(center[feature])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "profiles").mkdir(exist_ok=True)
    (output_dir / "assignments").mkdir(exist_ok=True)
    (output_dir / "scaling").mkdir(exist_ok=True)

    manifest = build_manifest(args)
    manifest.to_csv(output_dir / "experiment_manifest.csv", index=False)
    config = {
        "input": args.input,
        "k_values": split_csv(args.k_values),
        "scaling_modes": split_csv(args.scaling_modes),
        "feature_sets": {name: FEATURE_SETS[name] for name in split_csv(args.feature_sets)},
        "efficiency_features_excluded": EFFICIENCY_FEATURES,
        "methods": split_csv(args.methods),
        "inits": split_csv(args.inits),
        "n_init": args.n_init,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "seed": args.seed,
        "weights_file": args.weights_file,
        "sample_size": args.sample_size,
    }
    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(f"Planned runs: {len(manifest)}")
    print(f"Output directory: {output_dir}")
    if args.dry_run:
        print("Dry run only; wrote manifest and config.")
        return

    df = pd.read_csv(args.input, low_memory=False)
    if args.sample_size and args.sample_size < len(df):
        rng = np.random.default_rng(args.seed)
        sample_parts = []
        strata = df.groupby(["division", "season"], sort=True)
        for _, group in strata:
            take = max(1, round(args.sample_size * len(group) / len(df)))
            take = min(take, len(group))
            sample_parts.append(group.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
        sampled = pd.concat(sample_parts)
        if len(sampled) > args.sample_size:
            sampled = sampled.sample(n=args.sample_size, random_state=args.seed)
        elif len(sampled) < args.sample_size:
            remainder = df.drop(sampled.index, errors="ignore")
            extra = remainder.sample(
                n=min(args.sample_size - len(sampled), len(remainder)),
                random_state=args.seed,
            )
            sampled = pd.concat([sampled, extra])
        df = sampled.reset_index(drop=True)
        print(f"Using stratified sample rows: {len(df):,}", flush=True)
    summary_rows = []

    for idx, row in manifest.iterrows():
        start = time.time()
        features = row["features"].split("|")
        print(f"[{idx + 1}/{len(manifest)}] {row['run_id']}", flush=True)

        z, scaling = zscore_features(df, features, row["scaling_mode"])
        multipliers, feature_weights = load_feature_weights(args.weights_file, features)
        z_weighted = z * multipliers
        scaling.insert(0, "run_id", row["run_id"])
        scaling = scaling.merge(feature_weights, on="feature", how="left")
        scaling.to_csv(output_dir / "scaling" / f"{row['run_id']}__scaling.csv", index=False)

        model = AA(
            int(row["k"]),
            max_iter=int(row["max_iter"]),
            tol=float(row["tol"]),
            init=row["init"],
            n_init=int(row["n_init"]),
            init_params={"params": {}},
            method=row["method"],
            random_state=args.seed + idx,
        )
        model.fit(z_weighted)
        labels = model.labels_.astype(int)
        coefficients = model.coefficients_
        archetypes_unweighted_z = model.archetypes_ / multipliers
        profiles = feature_profiles(df, features, labels, archetypes_unweighted_z, coefficients)
        profiles.insert(0, "run_id", row["run_id"])
        profiles.to_csv(output_dir / "profiles" / f"{row['run_id']}__profiles.csv", index=False)

        elapsed = time.time() - start
        counts = pd.Series(labels).value_counts()
        summary_rows.append(
            {
                **row.to_dict(),
                "rss": float(model.rss_),
                "rss_per_row": float(model.rss_ / len(df)),
                "n_iter": int(model.n_iter_) if np.isscalar(model.n_iter_) else str(model.n_iter_),
                "min_archetype_rows": int(counts.min()),
                "median_archetype_rows": float(counts.median()),
                "max_archetype_rows": int(counts.max()),
                "elapsed_seconds": elapsed,
            }
        )
        pd.DataFrame(summary_rows).to_csv(output_dir / "experiment_summary.csv", index=False)

        if args.write_assignments:
            assignments = df[ID_COLUMNS].copy()
            assignments.insert(0, "run_id", row["run_id"])
            assignments.insert(1, "archetype_id", labels + 1)
            assignments.insert(2, "archetype_weight", coefficients[np.arange(len(df)), labels])
            for archetype_idx in range(coefficients.shape[1]):
                assignments[f"archetype_{archetype_idx + 1}_weight"] = coefficients[:, archetype_idx]
            assignments.to_csv(output_dir / "assignments" / f"{row['run_id']}__assignments.csv", index=False)

    print(f"Completed runs: {len(summary_rows)}")
    print(f"Summary: {output_dir / 'experiment_summary.csv'}")


if __name__ == "__main__":
    main()
