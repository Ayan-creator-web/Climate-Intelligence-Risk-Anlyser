"""
src/insights_generator.py
--------------------------
Automatically generates plain-English climate insights from the
enriched dataset and risk scores.

Functions
─────────
  generate_insights(df, risk_df) → list[str]
  key_findings(df, risk_df)      → list[str]  (top 3)
"""

import numpy as np
import pandas as pd


def _fmt(val: float, decimals: int = 2) -> str:
    return f"{val:+.{decimals}f}" if val >= 0 else f"{val:.{decimals}f}"


def generate_insights(
    df: pd.DataFrame,
    risk_df: pd.DataFrame | None = None,
) -> list[str]:
    """
    Returns a list of auto-generated insight strings.
    """
    insights: list[str] = []

    # ── 1. Global temperature trend ──────────────────────────────────────────
    annual = df.groupby("year")["temperature"].mean()
    if len(annual) >= 20:
        early = annual.iloc[:10].mean()
        late  = annual.iloc[-10:].mean()
        delta = late - early
        direction = "risen" if delta > 0 else "fallen"
        insights.append(
            f"🌡️  Global average temperature has {direction} by "
            f"**{abs(delta):.2f}°C** from the 1950s to the 2020s."
        )

    # ── 2. Fastest warming region ─────────────────────────────────────────────
    from src.trend_analysis import slope_per_decade
    slopes = {}
    for region, grp in df.groupby("region"):
        grp = grp.sort_values("date")
        slopes[region] = slope_per_decade(grp["temperature"].values, grp["date"])
    if slopes:
        fastest = max(slopes, key=lambda r: slopes[r] if not np.isnan(slopes[r]) else -999)
        insights.append(
            f"🔥  **{fastest}** is the fastest-warming region, with a trend of "
            f"**{slopes[fastest]:+.2f}°C per decade**."
        )

    # ── 3. Year of highest anomaly ───────────────────────────────────────────
    if "temp_anomaly" in df.columns:
        peak_row = df.loc[df["temp_anomaly"].idxmax()]
        insights.append(
            f"📈  The highest single-month temperature anomaly occurred in "
            f"**{peak_row['region']}** during "
            f"**{pd.Timestamp(peak_row['date']).strftime('%b %Y')}** "
            f"({peak_row['temp_anomaly']:+.2f}°C above baseline)."
        )

    # ── 4. Most extreme rainfall region ──────────────────────────────────────
    rain_cv = df.groupby("region")["rainfall"].apply(
        lambda s: s.std() / s.mean() if s.mean() > 0 else 0
    )
    wettest = rain_cv.idxmax()
    insights.append(
        f"🌧️  **{wettest}** has the highest rainfall variability "
        f"(CV = {rain_cv[wettest]:.2f}), indicating unpredictable precipitation patterns."
    )

    # ── 5. Extreme events ────────────────────────────────────────────────────
    if "extreme_event" in df.columns:
        total_ext = df["extreme_event"].sum()
        pct_ext   = df["extreme_event"].mean() * 100
        insights.append(
            f"⚠️  **{int(total_ext):,}** extreme climate events detected "
            f"across all regions ({pct_ext:.1f}% of all monthly records)."
        )

        # Trend in extreme events over time
        ext_annual = df[df["extreme_event"]].groupby("year").size()
        if len(ext_annual) >= 20:
            early_ext = ext_annual.iloc[:10].mean()
            late_ext  = ext_annual.iloc[-10:].mean()
            if late_ext > early_ext * 1.2:
                insights.append(
                    f"📊  Extreme events are **increasing** — the annual count has grown "
                    f"from ~{early_ext:.0f} in earlier decades to ~{late_ext:.0f} recently."
                )

    # ── 6. CO₂ trend ─────────────────────────────────────────────────────────
    co2_early = df[df["year"] <= df["year"].min() + 5]["co2"].mean()
    co2_late  = df[df["year"] >= df["year"].max() - 5]["co2"].mean()
    insights.append(
        f"🏭  Atmospheric CO₂ has increased from "
        f"**{co2_early:.0f} ppm** to **{co2_late:.0f} ppm** "
        f"({co2_late - co2_early:+.0f} ppm over the study period)."
    )

    # ── 7. Sea level ─────────────────────────────────────────────────────────
    sl_late = df[df["year"] >= df["year"].max() - 5]["sea_level"].mean()
    insights.append(
        f"🌊  Sea levels are currently approximately "
        f"**{sl_late:.0f} mm** above the 1950 baseline."
    )

    # ── 8. Risk scores ────────────────────────────────────────────────────────
    if risk_df is not None and "risk_score" in risk_df.columns:
        top_risk = risk_df.sort_values("risk_score", ascending=False).iloc[0]
        low_risk = risk_df.sort_values("risk_score").iloc[0]
        insights.append(
            f"🔴  **{top_risk['region']}** carries the highest climate risk score "
            f"({top_risk['risk_score']:.0f}/100) — driven by {top_risk['primary_driver'].lower()}."
        )
        insights.append(
            f"🟢  **{low_risk['region']}** has the lowest climate risk score "
            f"({low_risk['risk_score']:.0f}/100)."
        )

    # ── 9. Seasonal shift ─────────────────────────────────────────────────────
    if "season" in df.columns:
        summer_trend = (
            df[df["season"] == "Summer"]
            .groupby("year")["temperature"]
            .mean()
        )
        if len(summer_trend) >= 20:
            delta_summer = summer_trend.iloc[-10:].mean() - summer_trend.iloc[:10].mean()
            insights.append(
                f"☀️  Summer temperatures have shifted by "
                f"**{delta_summer:+.2f}°C** on average since the 1950s, "
                f"suggesting measurable seasonal intensification."
            )

    # ── 10. Data quality note ─────────────────────────────────────────────────
    insights.append(
        "ℹ️  *Note: Recent data may be preliminary. "
        "Forecasts shown in the dashboard are illustrative trend extrapolations "
        "and are not equivalent to official climate model projections.*"
    )

    return insights


def key_findings(
    df: pd.DataFrame,
    risk_df: pd.DataFrame | None = None,
) -> list[str]:
    """Return the top 3 most impactful insights."""
    all_insights = generate_insights(df, risk_df)
    # Return items 1, 2, 3 (indices 1, 2, 4 — warming, fastest region, extremes)
    picks = []
    priority_keywords = ["fastest-warming", "highest single-month", "highest climate risk"]
    for keyword in priority_keywords:
        for ins in all_insights:
            if keyword in ins:
                picks.append(ins)
                break
    # fill to 3 if needed
    for ins in all_insights:
        if ins not in picks:
            picks.append(ins)
        if len(picks) == 3:
            break
    return picks[:3]


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from src.preprocessing import load_and_clean
    from src.feature_engineering import add_all_features
    from src.anomaly_detection import detect_extreme_events
    from src.risk_scoring import compute_risk_scores

    DATA = os.path.join(os.path.dirname(__file__), "..", "data", "climate_raw.csv")
    df, _ = load_and_clean(DATA)
    df = add_all_features(df)
    df = detect_extreme_events(df)
    risk_df = compute_risk_scores(df)

    print("✅  AUTO-GENERATED INSIGHTS\n")
    for ins in generate_insights(df, risk_df):
        print(" •", ins)

    print("\n✅  KEY FINDINGS\n")
    for f in key_findings(df, risk_df):
        print(" ★", f)
