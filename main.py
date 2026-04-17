"""
main.py
=======
CLI entry-point for the Climate Intelligence & Risk Analyzer.

Usage
─────
  python main.py          # full pipeline + save outputs
  python main.py --help
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from data.generate_dataset   import generate_climate_dataset
from src.preprocessing       import load_and_clean
from src.feature_engineering import add_all_features
from src.trend_analysis      import region_trend_table, decade_mean_table
from src.anomaly_detection   import detect_extreme_events, top_extreme_events
from src.forecasting         import get_forecast
from src.risk_scoring        import compute_risk_scores
from src.insights_generator  import generate_insights, key_findings

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "outputs")
IMAGE_DIR   = os.path.join(os.path.dirname(__file__), "images")
REPORT_DIR  = os.path.join(os.path.dirname(__file__), "reports")
RAW_CSV     = os.path.join(DATA_DIR,   "climate_raw.csv")
CLEAN_CSV   = os.path.join(OUTPUT_DIR, "climate_clean.csv")
RISK_CSV    = os.path.join(OUTPUT_DIR, "risk_scores.csv")
ANOMALY_CSV = os.path.join(OUTPUT_DIR, "extreme_events.csv")
TREND_CSV   = os.path.join(OUTPUT_DIR, "region_trends.csv")
INSIGHT_TXT = os.path.join(REPORT_DIR, "auto_insights.txt")


def ensure_dirs():
    for d in [DATA_DIR, OUTPUT_DIR, IMAGE_DIR, REPORT_DIR]:
        os.makedirs(d, exist_ok=True)


def step_generate():
    if not os.path.exists(RAW_CSV):
        print("📊 Step 1: Generating synthetic climate dataset …")
        df = generate_climate_dataset()
        df.to_csv(RAW_CSV, index=False)
        print(f"   ✅ Saved → {RAW_CSV}  shape={df.shape}")
    else:
        print(f"📊 Step 1: Dataset already exists → {RAW_CSV}")


def step_clean(baseline=(1991, 2020)):
    print("\n🧹 Step 2: Loading & cleaning data …")
    df, report = load_and_clean(RAW_CSV)
    print(f"   Missing before: {report['missing_before']}  |  after: {report['missing_after']}")

    print("\n⚙️  Step 3: Feature engineering …")
    df = add_all_features(df, baseline)

    print("\n🔍 Step 4: Anomaly detection …")
    df = detect_extreme_events(df)
    print(f"   Extreme events flagged: {df['extreme_event'].sum():,}")

    df.to_csv(CLEAN_CSV, index=False)
    print(f"   ✅ Clean dataset saved → {CLEAN_CSV}")
    return df


def step_analysis(df):
    print("\n📐 Step 5: Trend analysis …")
    trends = region_trend_table(df).reset_index()
    trends.to_csv(TREND_CSV, index=False)
    print("   Region Trend Table:")
    print(trends[["region","temp_slope_per_decade","trend_direction"]].to_string(index=False))

    print("\n🛡️  Step 6: Risk scoring …")
    risk_df = compute_risk_scores(df)
    risk_df.to_csv(RISK_CSV, index=False)
    print("   Risk Scores:")
    print(risk_df[["region","risk_score","risk_level"]].to_string(index=False))

    print("\n⚠️  Step 7: Saving extreme events table …")
    top = top_extreme_events(df, 50)
    top.to_csv(ANOMALY_CSV, index=False)
    print(f"   Top events saved → {ANOMALY_CSV}")

    return risk_df


def step_insights(df, risk_df):
    print("\n💡 Step 8: Auto-generating insights …")
    insights = generate_insights(df, risk_df)
    findings = key_findings(df, risk_df)

    with open(INSIGHT_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("   CLIMATE INTELLIGENCE & RISK ANALYZER — AUTO INSIGHTS\n")
        f.write("=" * 70 + "\n\n")
        f.write("★  KEY FINDINGS\n\n")
        for i, fin in enumerate(findings, 1):
            clean = fin.replace("**", "").replace("*", "")
            f.write(f"  {i}. {clean}\n")
        f.write("\n" + "-" * 70 + "\n")
        f.write("\n📋  ALL INSIGHTS\n\n")
        for ins in insights:
            clean = ins.replace("**", "").replace("*", "")
            f.write(f"  • {clean}\n")

    print(f"   ✅ Insights saved → {INSIGHT_TXT}")
    for fin in findings:
        clean = fin.replace("**", "").replace("*", "")
        print(f"     ★ {clean[:90]}")


def step_charts(df, risk_df):
    print("\n📊 Step 9: Generating static charts …")
    import numpy as np

    # 1. Global temperature anomaly
    ann = df.groupby("year")["temp_anomaly"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#e74c3c" if v >= 0 else "#3498db" for v in ann["temp_anomaly"]]
    ax.bar(ann["year"], ann["temp_anomaly"], color=colors, width=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Year"); ax.set_ylabel("Temperature Anomaly (°C)")
    ax.set_title("Global Mean Temperature Anomaly (1950–2023)", fontsize=14, fontweight="bold")
    ax.text(0.01, 0.01,
            "Baseline: 1991–2020 | Red = warmer, Blue = cooler",
            transform=ax.transAxes, fontsize=9, color="gray")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "01_global_temp_anomaly.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ 01_global_temp_anomaly.png")

    # 2. Region temperature trends
    ann_region = df.groupby(["year","region"])["temperature"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(13, 6))
    for region, grp in ann_region.groupby("region"):
        ax.plot(grp["year"], grp["temperature"], label=region, linewidth=1.5)
    ax.set_xlabel("Year"); ax.set_ylabel("Temperature (°C)")
    ax.set_title("Regional Annual Mean Temperature Trends", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "02_region_temp_trends.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ 02_region_temp_trends.png")

    # 3. Extreme events per year
    ext_yr = df[df["extreme_event"]].groupby("year").size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(ext_yr["year"], ext_yr["count"], alpha=0.6, color="#e67e22")
    ax.plot(ext_yr["year"], ext_yr["count"], color="#d35400", linewidth=2)
    ax.set_xlabel("Year"); ax.set_ylabel("Number of Extreme Events")
    ax.set_title("Extreme Climate Events per Year (All Regions)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "03_extreme_events_timeline.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ 03_extreme_events_timeline.png")

    # 4. Risk score bar chart
    r = risk_df.sort_values("risk_score", ascending=True)
    risk_colors = [
        "#e74c3c" if "High" in lv else "#f39c12" if "Medium" in lv else "#2ecc71"
        for lv in r["risk_level"]
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(r["region"], r["risk_score"], color=risk_colors)
    ax.axvline(33, color="#2ecc71", linestyle="--", linewidth=1, label="Low/Medium")
    ax.axvline(67, color="#e74c3c", linestyle="--", linewidth=1, label="Medium/High")
    ax.set_xlabel("Climate Risk Score (0–100)")
    ax.set_title("Climate Risk Score by Region", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    for bar, val in zip(bars, r["risk_score"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{val:.0f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "04_risk_scores.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ 04_risk_scores.png")

    # 5. Decade heatmap
    import seaborn as sns
    pivot = decade_mean_table(df)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="RdYlBu_r",
        ax=ax, linewidths=0.5, cbar_kws={"label": "Avg Temp (°C)"},
    )
    ax.set_title("Decade Mean Temperature by Region", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "05_decade_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ 05_decade_heatmap.png")

    # 6. Seasonal pattern for one region
    region = "South Asia"
    rdf = df[df["region"] == region]
    monthly = rdf.groupby("month")["temperature"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(monthly.index, monthly.values, alpha=0.4, color="#e74c3c")
    ax.plot(monthly.index, monthly.values, "o-", color="#c0392b", linewidth=2)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_xlabel("Month"); ax.set_ylabel("Avg Temperature (°C)")
    ax.set_title(f"Seasonal Temperature Pattern — {region}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "06_seasonal_pattern.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ 06_seasonal_pattern.png")

    # 7. Forecast chart
    hist, fc = get_forecast(df, "South Asia", "temperature", 24)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(hist["date"], hist["value"], color="#7f8c8d", linewidth=1, label="Historical", alpha=0.7)
    roll12 = hist["value"].rolling(12, min_periods=6).mean()
    ax.plot(hist["date"], roll12, color="#e74c3c", linewidth=2, label="12-Month Avg")
    ax.plot(fc["date"], fc["forecast"], "--", color="#2980b9", linewidth=2.5, label="Forecast")
    ax.fill_between(fc["date"], fc["lower_ci"], fc["upper_ci"],
                    alpha=0.2, color="#2980b9", label="90% CI")
    ax.axvline(hist["date"].iloc[-1], color="orange", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Date"); ax.set_ylabel("Temperature (°C)")
    ax.set_title("Temperature Forecast — South Asia (24 months)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "07_forecast.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ 07_forecast.png")

    print(f"\n   📁 All charts saved to: {IMAGE_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Climate Intelligence & Risk Analyzer — CLI")
    parser.add_argument("--baseline-start", type=int, default=1991, help="Baseline start year")
    parser.add_argument("--baseline-end",   type=int, default=2020, help="Baseline end year")
    parser.add_argument("--skip-charts",    action="store_true",    help="Skip chart generation")
    args = parser.parse_args()

    baseline = (args.baseline_start, args.baseline_end)

    print("=" * 65)
    print("  🌍 CLIMATE INTELLIGENCE & RISK ANALYZER")
    print(f"  Baseline: {baseline[0]}–{baseline[1]}")
    print("=" * 65)

    ensure_dirs()
    step_generate()
    df = step_clean(baseline)
    risk_df = step_analysis(df)
    step_insights(df, risk_df)

    if not args.skip_charts:
        step_charts(df, risk_df)

    print("\n" + "=" * 65)
    print("  ✅  PIPELINE COMPLETE")
    print(f"  Outputs  → {OUTPUT_DIR}")
    print(f"  Images   → {IMAGE_DIR}")
    print(f"  Reports  → {REPORT_DIR}")
    print("  Dashboard → run:  streamlit run app/dashboard.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
