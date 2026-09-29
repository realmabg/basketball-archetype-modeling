#!/usr/bin/env python3
"""Add residual role/context features for archetype experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = "combined_d1_d2_exports/archetype_model_features_step43.csv"
DEFAULT_OUTPUT = "combined_d1_d2_exports/archetype_model_features_step43_residuals.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def qcut_codes(series: pd.Series, bins: int) -> pd.Series:
    """Quantile-bin a series, reducing bins automatically if ties are heavy."""
    ranked = series.rank(method="first")
    return pd.qcut(ranked, q=min(bins, series.notna().sum()), labels=False, duplicates="drop")


def residual_by_group_mean(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    min_group_size: int,
    fallback_cols: list[str],
) -> tuple[pd.Series, pd.Series]:
    expected = pd.Series(np.nan, index=df.index, dtype=float)
    group_sizes = df.groupby(group_cols, dropna=False)[value_col].transform("size")
    group_means = df.groupby(group_cols, dropna=False)[value_col].transform("mean")
    expected.loc[group_sizes >= min_group_size] = group_means.loc[group_sizes >= min_group_size]

    missing = expected.isna()
    if missing.any():
        fallback_means = df.groupby(fallback_cols, dropna=False)[value_col].transform("mean")
        expected.loc[missing] = fallback_means.loc[missing]

    missing = expected.isna()
    if missing.any():
        expected.loc[missing] = df[value_col].mean()

    return df[value_col] - expected, expected


def add_shot_mix_residual(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["three_share_bin"] = -1
    out["three_pa_rate_bin"] = -1

    for division, idx in out.groupby("division").groups.items():
        div = out.loc[idx]
        out.loc[idx, "three_share_bin"] = qcut_codes(div["three_share"], 10).astype(int)
        out.loc[idx, "three_pa_rate_bin"] = qcut_codes(div["three_pa_per_100_team_poss"], 10).astype(int)

    resid, expected = residual_by_group_mean(
        out,
        value_col="three_pct",
        group_cols=["division", "three_share_bin", "three_pa_rate_bin"],
        min_group_size=20,
        fallback_cols=["division", "three_share_bin"],
    )
    out["resid_three_pct_shot_mix"] = resid
    out["expected_three_pct_shot_mix"] = expected
    return out.drop(columns=["three_share_bin", "three_pa_rate_bin"])


def add_height_residuals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["height_bin_inches"] = pd.to_numeric(out["height_inches"], errors="coerce").round().astype("Int64")

    residual_specs = {
        "resid_orb_pct_height": "orb_pct",
        "resid_three_share_height": "three_share",
        "resid_drb_pct_height": "drb_pct",
    }
    for residual_col, value_col in residual_specs.items():
        resid, expected = residual_by_group_mean(
            out,
            value_col=value_col,
            group_cols=["division", "height_bin_inches"],
            min_group_size=20,
            fallback_cols=["division"],
        )
        out[residual_col] = resid
        out[f"expected_{value_col}_height"] = expected

    return out.drop(columns=["height_bin_inches"])


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, low_memory=False)
    required = [
        "division",
        "three_pct",
        "three_share",
        "three_pa_per_100_team_poss",
        "height_inches",
        "orb_pct",
        "drb_pct",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    out = add_shot_mix_residual(df)
    out = add_height_residuals(out)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    residual_cols = [
        "resid_three_pct_shot_mix",
        "resid_orb_pct_height",
        "resid_three_share_height",
        "resid_drb_pct_height",
    ]
    summary = out.groupby("division")[residual_cols].agg(["mean", "std", "min", "max"])
    print(f"Wrote {output} with {len(out):,} rows")
    print(summary.to_string())


if __name__ == "__main__":
    main()
