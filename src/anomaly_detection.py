"""
src/anomaly_detection.py
------------------------
Detects extreme climate events using multiple statistical methods:
  - Z-score thresholding
  - IQR-based outlier detection
  - Percentile-based thresholding

Severity classification:
  Normal | Mild Anomaly | Severe Anomaly | Critical Anomaly

Event type labelling:
  Heatwave | Cold Snap | Extreme Rainfall | Drought-like | Normal
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ─── Severity mapping ─────────────────────────────────────────────────────────
def _severity_from_zscore(z: float) -> str:
    az = abs(z)
    if az < 1.5:
        return "Normal"
    if az < 2.0:
        return "Mild Anomaly"
    if az < 3.0:
        return "Severe Anomaly"
    return "Critical Anomaly"


def _event_type(row: pd.Series) -> str:
    tz = row.get("temp_zscore", 0)
    rz = row.get("rain_zscore", 0)

    if tz > 2.5:
        return "Heatwave"
    if tz < -2.5:
        return "Cold Snap"
    if rz > 2.5:
        return "Extreme Rainfall"
    if rz < -2.5:
        return "Drought-like"
    if abs(tz) > 1.5:
        return "Temperature Anomaly"
    if abs(rz) > 1.5:
        return "Rainfall Anomaly"
    return "Normal"


# ─── Main detector ───────────────────────────────────────────────────────────
def detect_extreme_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns:
      extreme_event   – bool, True when either temp or rain z-score |z| > 1.5
      event_type      – string label
      temp_severity   – Normal / Mild / Severe / Critical
      rain_severity   – Normal / Mild / Severe / Critical
      anomaly_score   – composite 0–100 anomaly intensity
    """
    df = df.copy()

    # ensure z-scores are present
    if "temp_zscore" not in df.columns:
        raise ValueError("Run feature_engineering.add_all_features() first.")

    df["temp_severity"] = df["temp_zscore"].apply(_severity_from_zscore)
    df["rain_severity"] = df["rain_zscore"].apply(_severity_from_zscore)
    df["event_type"]    = df.apply(_event_type, axis=1)
    df["extreme_event"] = df["event_type"] != "Normal"

    # composite 0-100 anomaly score (max of both z-scores, normalised)
    max_z = df[["temp_zscore", "rain_zscore"]].abs().max(axis=1)
    df["anomaly_score"] = (max_z.clip(0, 4) / 4 * 100).round(1)

    return df


# ─── IQR flag (alternative method) ──────────────────────────────────────────
def flag_iqr_outliers(
    df: pd.DataFrame,
    variable: str = "temperature",
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """Flag values outside [Q1 - m*IQR, Q3 + m*IQR] per region."""
    df = df.copy()
    col_flag = f"{variable}_iqr_flag"

    def _flag(s: pd.Series) -> pd.Series:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return (s < q1 - multiplier * iqr) | (s > q3 + multiplier * iqr)

    df[col_flag] = df.groupby("region")[variable].transform(_flag)
    return df


# ─── Summary table ───────────────────────────────────────────────────────────
def extreme_event_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a summary DataFrame with extreme-event counts per region.
    """
    if "extreme_event" not in df.columns:
        df = detect_extreme_events(df)

    events = df[df["extreme_event"]].copy()
    summary = (
        events.groupby(["region", "event_type"])
        .size()
        .reset_index(name="count")
        .sort_values(["region", "count"], ascending=[True, False])
    )
    return summary


def top_extreme_events(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return the n rows with the highest anomaly_score."""
    if "anomaly_score" not in df.columns:
        df = detect_extreme_events(df)
    return (
        df.nlargest(n, "anomaly_score")[
            ["date", "region", "country", "temperature", "rainfall",
             "temp_zscore", "rain_zscore", "event_type",
             "temp_severity", "anomaly_score"]
        ]
        .reset_index(drop=True)
    )


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from src.preprocessing import load_and_clean
    from src.feature_engineering import add_all_features

    DATA = os.path.join(os.path.dirname(__file__), "..", "data", "climate_raw.csv")
    df, _ = load_and_clean(DATA)
    df = add_all_features(df)
    df = detect_extreme_events(df)

    print("✅  Extreme events detected:", df["extreme_event"].sum())
    print("\n   Event type counts:")
    print(df["event_type"].value_counts())
    print("\n   Top 5 critical events:")
    print(top_extreme_events(df, 5)[["date","region","event_type","anomaly_score"]])
