"""
src/feature_engineering.py
--------------------------
Computes derived climate features on top of the clean dataset.

Added columns
─────────────
  temp_anomaly      – deviation from baseline mean temperature
  rain_anomaly      – deviation from baseline mean rainfall
  temp_roll12       – 12-month rolling mean of temperature
  rain_roll12       – 12-month rolling mean of rainfall
  temp_zscore       – z-score of temperature per region
  rain_zscore       – z-score of rainfall per region
  temp_pct_change   – % change vs prior year same month
  rain_pct_change   – % change vs prior year same month
  trend_label       – decade-level warming label
  decade            – decade string, e.g. "1990s"
"""

import pandas as pd
import numpy as np
from typing import Tuple


# ─── Baseline anomaly ─────────────────────────────────────────────────────────
def compute_anomalies(
    df: pd.DataFrame,
    baseline: Tuple[int, int] = (1991, 2020),
) -> pd.DataFrame:
    """
    Compute temperature and rainfall anomalies relative to a baseline period.
    Anomaly = value - baseline_mean  (per region, per calendar month).
    """
    df = df.copy()
    b_start, b_end = baseline

    base_mask = (df["year"] >= b_start) & (df["year"] <= b_end)
    base_stats = (
        df[base_mask]
        .groupby(["region", "month"])[["temperature", "rainfall"]]
        .mean()
        .rename(columns={"temperature": "base_temp", "rainfall": "base_rain"})
        .reset_index()
    )

    df = df.merge(base_stats, on=["region", "month"], how="left")
    df["temp_anomaly"] = (df["temperature"] - df["base_temp"]).round(3)
    df["rain_anomaly"] = (df["rainfall"] - df["base_rain"]).round(3)
    df.drop(columns=["base_temp", "base_rain"], inplace=True)
    return df


# ─── Rolling means ────────────────────────────────────────────────────────────
def compute_rolling_means(
    df: pd.DataFrame,
    window: int = 12,
) -> pd.DataFrame:
    """12-month centred rolling mean per region."""
    df = df.copy()
    df = df.sort_values(["region", "date"])

    df["temp_roll12"] = (
        df.groupby("region")["temperature"]
        .transform(lambda s: s.rolling(window, min_periods=6, center=True).mean())
        .round(3)
    )
    df["rain_roll12"] = (
        df.groupby("region")["rainfall"]
        .transform(lambda s: s.rolling(window, min_periods=6, center=True).mean())
        .round(3)
    )
    return df


# ─── Z-scores ────────────────────────────────────────────────────────────────
def compute_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """Per-region z-scores for temperature and rainfall."""
    df = df.copy()

    for col, out in [("temperature", "temp_zscore"), ("rainfall", "rain_zscore")]:
        mu  = df.groupby("region")[col].transform("mean")
        sig = df.groupby("region")[col].transform("std")
        df[out] = ((df[col] - mu) / sig.replace(0, np.nan)).round(3)

    return df


# ─── Year-over-year % change ──────────────────────────────────────────────────
def compute_yoy_change(df: pd.DataFrame) -> pd.DataFrame:
    """Percentage change vs same month of prior year, per region."""
    df = df.copy().sort_values(["region", "date"])

    def _pct(s: pd.Series) -> pd.Series:
        return s.pct_change(12).mul(100).round(2)

    df["temp_pct_change"] = df.groupby("region")["temperature"].transform(_pct)
    df["rain_pct_change"] = df.groupby("region")["rainfall"].transform(_pct)
    return df


# ─── Decade labels ───────────────────────────────────────────────────────────
def add_decade(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["decade"] = (df["year"] // 10 * 10).astype(str) + "s"
    return df


# ─── Trend label per region ──────────────────────────────────────────────────
def add_trend_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each row with a warming trend category based on the
    rolling-12 temperature trend over recent years.
    """
    df = df.copy()
    df["trend_label"] = "Stable"

    for region, grp in df.groupby("region"):
        idx = grp.index
        anomaly = grp["temp_anomaly"].values
        if len(anomaly) < 24:
            continue
        recent_mean = anomaly[-24:].mean()
        if recent_mean > 1.5:
            label = "Strong Warming"
        elif recent_mean > 0.7:
            label = "Moderate Warming"
        elif recent_mean > 0.2:
            label = "Slight Warming"
        elif recent_mean < -0.7:
            label = "Cooling"
        else:
            label = "Stable"
        df.loc[idx, "trend_label"] = label

    return df


# ─── Master pipeline ─────────────────────────────────────────────────────────
def add_all_features(
    df: pd.DataFrame,
    baseline: Tuple[int, int] = (1991, 2020),
) -> pd.DataFrame:
    """
    Apply the complete feature engineering pipeline.
    Returns DataFrame enriched with all derived columns.
    """
    df = compute_anomalies(df, baseline)
    df = compute_rolling_means(df)
    df = compute_zscores(df)
    df = compute_yoy_change(df)
    df = add_decade(df)
    df = add_trend_labels(df)
    return df


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from src.preprocessing import load_and_clean

    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "climate_raw.csv")
    df, _ = load_and_clean(DATA_PATH)
    df = add_all_features(df)
    print("✅  Feature-engineered shape:", df.shape)
    print("   New columns:", [c for c in df.columns if c not in
          ["date","year","month","quarter","season","region","country",
           "temperature","rainfall","humidity","co2","sea_level"]])
    print(df[["region","date","temperature","temp_anomaly","temp_zscore","trend_label"]].head(6))
