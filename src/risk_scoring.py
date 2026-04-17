"""
src/risk_scoring.py
-------------------
Computes a composite Climate Risk Score (0–100) for every region.

Score components
────────────────
  1. Anomaly Frequency   – % of months flagged as extreme
  2. Warming Trend       – temperature slope per decade (normalised)
  3. Rainfall Volatility – CV of monthly rainfall
  4. Extreme Event Count – extreme events per year
  5. Recent Acceleration – trend in last 10 years vs full period

Risk categories
───────────────
  Low    [0 – 33]
  Medium [34 – 66]
  High   [67 – 100]
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ─── Component scorers ────────────────────────────────────────────────────────
def _anomaly_frequency(grp: pd.DataFrame) -> float:
    """Fraction of months with |temp_zscore| > 1.5."""
    if "temp_zscore" not in grp.columns:
        return 0.0
    return float((grp["temp_zscore"].abs() > 1.5).mean())


def _warming_trend(grp: pd.DataFrame) -> float:
    """Temperature slope per decade (raw)."""
    from src.trend_analysis import slope_per_decade
    vals  = grp["temperature"].values
    dates = grp["date"]
    s = slope_per_decade(vals, dates)
    return 0.0 if np.isnan(s) else max(0.0, s)


def _rainfall_volatility(grp: pd.DataFrame) -> float:
    """Coefficient of variation of monthly rainfall."""
    rain = grp["rainfall"].dropna()
    if rain.mean() == 0:
        return 0.0
    return float(rain.std() / rain.mean())


def _extreme_event_rate(grp: pd.DataFrame) -> float:
    """Extreme events per year."""
    if "extreme_event" not in grp.columns:
        return 0.0
    n_years = grp["year"].nunique()
    return float(grp["extreme_event"].sum() / max(1, n_years))


def _recent_acceleration(grp: pd.DataFrame) -> float:
    """
    Ratio of slope(last 10 yr) / slope(full period).
    Values > 1 indicate acceleration.
    """
    from src.trend_analysis import slope_per_decade
    grp = grp.sort_values("date")

    full_slope = slope_per_decade(grp["temperature"].values, grp["date"])
    recent = grp[grp["year"] >= grp["year"].max() - 10]
    recent_slope = slope_per_decade(recent["temperature"].values, recent["date"])

    if np.isnan(full_slope) or full_slope == 0:
        return 1.0
    ratio = recent_slope / full_slope
    return float(np.clip(ratio, 0, 5))


# ─── Master scorer ───────────────────────────────────────────────────────────
def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame (index = region) with columns:
      anomaly_freq, warming_trend, rain_volatility, extreme_rate,
      acceleration, raw_score, risk_score (0-100), risk_level,
      risk_breakdown (dict)
    """
    rows = []
    for region, grp in df.groupby("region"):
        grp = grp.sort_values("date")
        af   = _anomaly_frequency(grp)
        wt   = _warming_trend(grp)
        rv   = _rainfall_volatility(grp)
        er   = _extreme_event_rate(grp)
        acc  = _recent_acceleration(grp)

        rows.append(
            dict(
                region=region,
                anomaly_freq=round(af, 4),
                warming_trend=round(wt, 4),
                rain_volatility=round(rv, 4),
                extreme_rate=round(er, 4),
                acceleration=round(acc, 4),
            )
        )

    risk_df = pd.DataFrame(rows).set_index("region")

    # Normalise each component to [0, 1]
    components = ["anomaly_freq", "warming_trend", "rain_volatility",
                  "extreme_rate", "acceleration"]
    weights    = [0.25, 0.30, 0.20, 0.15, 0.10]  # sum = 1.0

    scaler = MinMaxScaler()
    norm = pd.DataFrame(
        scaler.fit_transform(risk_df[components]),
        columns=components,
        index=risk_df.index,
    )

    risk_df["risk_score"] = (
        (norm * weights).sum(axis=1) * 100
    ).round(1)

    risk_df["risk_level"] = risk_df["risk_score"].apply(_risk_level)

    # Human-readable breakdown
    risk_df["primary_driver"] = norm.idxmax(axis=1).map({
        "anomaly_freq":   "High anomaly frequency",
        "warming_trend":  "Strong warming trend",
        "rain_volatility":"High rainfall volatility",
        "extreme_rate":   "Frequent extreme events",
        "acceleration":   "Accelerating recent trend",
    })

    return risk_df.reset_index()


def _risk_level(score: float) -> str:
    if score >= 67:
        return "🔴 High"
    if score >= 34:
        return "🟡 Medium"
    return "🟢 Low"


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from src.preprocessing import load_and_clean
    from src.feature_engineering import add_all_features
    from src.anomaly_detection import detect_extreme_events

    DATA = os.path.join(os.path.dirname(__file__), "..", "data", "climate_raw.csv")
    df, _ = load_and_clean(DATA)
    df = add_all_features(df)
    df = detect_extreme_events(df)

    scores = compute_risk_scores(df)
    print("✅  Climate Risk Scores:")
    print(scores[["region","risk_score","risk_level","primary_driver"]].to_string())
