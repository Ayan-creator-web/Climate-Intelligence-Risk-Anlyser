"""
src/trend_analysis.py
---------------------
Linear-regression based trend engine.

Functions
─────────
  slope_per_decade(series, dates)   → float (°C / 10 yr)
  region_trend_table(df)            → DataFrame with slope columns
  decade_mean_table(df)             → pivot of decade means per region
  trend_direction(slope)            → "Increasing" | "Decreasing" | "Stable"
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ─── Core slope helper ────────────────────────────────────────────────────────
def slope_per_decade(
    values: np.ndarray,
    dates: pd.Series,
) -> float:
    """
    Fit OLS to (time_in_years, values) and return slope × 10
    = change per decade.  Returns NaN if fewer than 12 data points.
    """
    mask = ~np.isnan(values)
    if mask.sum() < 12:
        return np.nan

    x = ((dates - dates.iloc[0]).dt.days.values[mask] / 365.25).reshape(-1, 1)
    y = values[mask]
    model = LinearRegression().fit(x, y)
    return round(float(model.coef_[0]) * 10, 4)   # per decade


# ─── Region-level trend table ─────────────────────────────────────────────────
def region_trend_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every region, compute:
      - temp_slope_per_decade
      - rain_slope_per_decade
      - anomaly_slope_per_decade
      - first_year_temp / last_year_temp (raw means)
      - trend_direction
    """
    rows = []
    for region, grp in df.groupby("region"):
        grp = grp.sort_values("date")
        t_slope  = slope_per_decade(grp["temperature"].values, grp["date"])
        r_slope  = slope_per_decade(grp["rainfall"].values, grp["date"])
        a_slope  = slope_per_decade(grp["temp_anomaly"].values, grp["date"])

        first10 = grp[grp["year"] <= grp["year"].min() + 9]["temperature"].mean()
        last10  = grp[grp["year"] >= grp["year"].max() - 9]["temperature"].mean()

        rows.append(
            dict(
                region=region,
                temp_slope_per_decade=t_slope,
                rain_slope_per_decade=r_slope,
                anomaly_slope_per_decade=a_slope,
                first_decade_avg_temp=round(first10, 2),
                last_decade_avg_temp=round(last10, 2),
                total_temp_change=round(last10 - first10, 2),
                trend_direction=trend_direction(t_slope),
            )
        )
    return pd.DataFrame(rows).set_index("region")


# ─── Decade mean pivot ────────────────────────────────────────────────────────
def decade_mean_table(
    df: pd.DataFrame,
    variable: str = "temperature",
) -> pd.DataFrame:
    """
    Returns a pivot table: regions × decades with mean of `variable`.
    Useful for heatmap visualisations.
    """
    tbl = (
        df.groupby(["region", "decade"])[variable]
        .mean()
        .round(2)
        .reset_index()
        .pivot(index="region", columns="decade", values=variable)
    )
    return tbl


# ─── Rolling trend (30-year window) ──────────────────────────────────────────
def rolling_trend_series(
    series: pd.Series,
    dates: pd.Series,
    window: int = 360,   # months  ≈ 30 years
) -> pd.Series:
    """
    For each month t, fit a linear regression over the preceding `window`
    months and return the slope per decade.
    """
    slopes = np.full(len(series), np.nan)
    vals   = series.values
    for i in range(window, len(vals)):
        s_window = vals[i - window: i]
        d_window = dates.iloc[i - window: i]
        slopes[i] = slope_per_decade(s_window, d_window)
    return pd.Series(slopes, index=series.index)


# ─── Helper ───────────────────────────────────────────────────────────────────
def trend_direction(slope: float) -> str:
    if np.isnan(slope):
        return "Unknown"
    if slope > 0.05:
        return "Increasing"
    if slope < -0.05:
        return "Decreasing"
    return "Stable"


# ─── Annual mean series (convenience) ────────────────────────────────────────
def annual_means(
    df: pd.DataFrame,
    region: str,
    variable: str = "temperature",
) -> pd.Series:
    """Return annual mean for a given region."""
    grp = df[df["region"] == region]
    return grp.groupby("year")[variable].mean()


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from src.preprocessing import load_and_clean
    from src.feature_engineering import add_all_features

    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "climate_raw.csv")
    df, _ = load_and_clean(DATA_PATH)
    df = add_all_features(df)

    tbl = region_trend_table(df)
    print("✅  Region Trend Table:")
    print(tbl[["temp_slope_per_decade", "total_temp_change", "trend_direction"]])

    dec = decade_mean_table(df)
    print("\n✅  Decade mean temperatures:")
    print(dec)
