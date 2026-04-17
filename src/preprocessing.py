"""
src/preprocessing.py
--------------------
Loads the raw CSV, validates schema, handles missing values,
and returns a clean DataFrame ready for feature engineering.
"""

import os
import pandas as pd
import numpy as np

REQUIRED_COLS = {
    "date", "year", "month", "quarter", "season",
    "region", "country",
    "temperature", "rainfall", "humidity", "co2", "sea_level",
}

NUMERIC_COLS = ["temperature", "rainfall", "humidity", "co2", "sea_level"]


# ─── Loader ────────────────────────────────────────────────────────────────────
def load_raw(path: str) -> pd.DataFrame:
    """Load CSV and parse dates."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Run  python data/generate_dataset.py  first."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    return df


# ─── Cleaning ──────────────────────────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Interpolate numeric gaps with linear interpolation per region,
    then forward-fill any remaining NaNs.
    Returns cleaned DataFrame and a quality report dict.
    """
    report = {}
    total_cells = len(df) * len(NUMERIC_COLS)
    missing_before = df[NUMERIC_COLS].isna().sum().sum()

    df = df.copy()
    for col in NUMERIC_COLS:
        # interpolate per region time series
        df[col] = (
            df.groupby("region")[col]
            .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
        )

    # fallback: fill any remaining NaN with global median
    for col in NUMERIC_COLS:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    missing_after = df[NUMERIC_COLS].isna().sum().sum()
    report["total_cells"]    = total_cells
    report["missing_before"] = int(missing_before)
    report["missing_after"]  = int(missing_after)
    report["pct_affected"]   = round(missing_before / total_cells * 100, 2)
    report["method"]         = "Linear interpolation per region + median fill"

    return df, report


def clip_outliers(df: pd.DataFrame, sigma: float = 5.0) -> pd.DataFrame:
    """
    Clip extreme values beyond ±sigma standard deviations per region.
    This removes data entry errors without removing genuine extremes.
    """
    df = df.copy()
    for col in NUMERIC_COLS:
        grp = df.groupby("region")[col]
        mean = grp.transform("mean")
        std  = grp.transform("std")
        lo   = mean - sigma * std
        hi   = mean + sigma * std
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def enforce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct dtypes."""
    df = df.copy()
    df["date"]    = pd.to_datetime(df["date"])
    df["year"]    = df["year"].astype(int)
    df["month"]   = df["month"].astype(int)
    df["quarter"] = df["quarter"].astype(int)
    for col in NUMERIC_COLS:
        df[col] = df[col].astype(float)
    for col in ["region", "country", "season"]:
        df[col] = df[col].astype("category")
    return df


# ─── Master function ───────────────────────────────────────────────────────────
def load_and_clean(path: str) -> tuple[pd.DataFrame, dict]:
    """
    Full pipeline: load → validate → clean → type-fix.
    Returns (clean_df, quality_report).
    """
    df = load_raw(path)
    df, report = handle_missing(df)
    df = clip_outliers(df)
    df = enforce_types(df)
    df = df.sort_values(["region", "date"]).reset_index(drop=True)
    return df, report


# ─── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE = os.path.join(os.path.dirname(__file__), "..", "data", "climate_raw.csv")
    df, report = load_and_clean(BASE)
    print("✅  Clean dataset shape:", df.shape)
    print("   Quality report:", report)
    print(df.dtypes)
    print(df.head(3))
