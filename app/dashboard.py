"""
app/dashboard.py
================
Climate Intelligence & Risk Analyzer — Streamlit Dashboard

Pages
─────
  🏠  Overview          — KPIs, key findings, data quality
  📈  Trend Analysis    — temperature/rainfall trends, rolling avg, decade slopes
  🚨  Anomaly Detector  — extreme events, severity, timeline
  🗺️  Regional Compare  — side-by-side region comparisons
  🔭  Forecast          — 12/24/60-month projections with CI bands
  🛡️  Risk Intelligence — composite risk scores, rankings, recommendations

Run
───
  streamlit run app/dashboard.py
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── src imports ───────────────────────────────────────────────────────────────
from src.preprocessing       import load_and_clean
from src.feature_engineering import add_all_features
from src.trend_analysis      import region_trend_table, decade_mean_table, annual_means
from src.anomaly_detection   import detect_extreme_events, extreme_event_summary, top_extreme_events
from src.forecasting         import get_forecast, FORECAST_CAVEAT
from src.risk_scoring        import compute_risk_scores
from src.insights_generator  import generate_insights, key_findings

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Climate Intelligence & Risk Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "climate_raw.csv")
BASELINE_OPTIONS = {
    "1951–1980 (pre-industrial reference)": (1951, 1980),
    "1981–2010 (standard WMO 30-yr)":       (1981, 2010),
    "1991–2020 (current WMO standard)":     (1991, 2020),
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers / caching
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Loading & cleaning dataset …")
def load_data(baseline: tuple) -> tuple:
    df, qreport = load_and_clean(DATA_PATH)
    df = add_all_features(df, baseline)
    df = detect_extreme_events(df)
    return df, qreport


@st.cache_data(show_spinner="⏳ Computing risk scores …")
def get_risk(df_json: str) -> pd.DataFrame:
    df = pd.read_json(df_json)
    df["date"] = pd.to_datetime(df["date"])
    return compute_risk_scores(df)


def df_to_json(df: pd.DataFrame) -> str:
    return df.to_json()


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
REGION_COLORS = px.colors.qualitative.Bold
RISK_COLORS   = {"🔴 High": "#e74c3c", "🟡 Medium": "#f39c12", "🟢 Low": "#2ecc71"}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def sidebar(df: pd.DataFrame) -> dict:
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/"
        "WMO_logo.svg/240px-WMO_logo.svg.png",
        width=60,
    )
    st.sidebar.title("🌍 Climate Intelligence\n& Risk Analyzer")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Overview", "📈 Trend Analysis", "🚨 Anomaly Detector",
         "🗺️ Regional Compare", "🔭 Forecast", "🛡️ Risk Intelligence",
         "💡 Insights"],
        index=0,
    )
    st.sidebar.markdown("---")

    baseline_label = st.sidebar.selectbox(
        "📏 Baseline Period",
        list(BASELINE_OPTIONS.keys()),
        index=2,
    )
    baseline = BASELINE_OPTIONS[baseline_label]

    all_regions = sorted(df["region"].unique().tolist())
    selected_regions = st.sidebar.multiselect(
        "🗺️ Filter Regions",
        all_regions,
        default=all_regions,
    )

    year_range = st.sidebar.slider(
        "📅 Year Range",
        int(df["year"].min()),
        int(df["year"].max()),
        (int(df["year"].min()), int(df["year"].max())),
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Data: Synthetic climate dataset (1950–2023)\n"
        "Methodology aligned with NOAA / NASA / IPCC standards.\n\n"
        "⚠️ For portfolio demonstration only."
    )

    return dict(
        page=page,
        baseline=baseline,
        baseline_label=baseline_label,
        selected_regions=selected_regions,
        year_range=year_range,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Overview
# ─────────────────────────────────────────────────────────────────────────────
def page_overview(df: pd.DataFrame, risk_df: pd.DataFrame, qreport: dict, opts: dict):
    st.title("🏠 Climate Overview Dashboard")
    st.markdown(
        "> *What is happening overall?* — A high-level summary of the dataset, "
        "key climate indicators, and top findings."
    )
    st.markdown("---")

    fdf = df[
        (df["region"].isin(opts["selected_regions"])) &
        (df["year"].between(*opts["year_range"]))
    ]

    # ── KPI cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("📋 Records",       f"{len(fdf):,}")
    c2.metric("🗺️ Regions",       fdf["region"].nunique())
    c3.metric("🌡️ Avg Temp",      f"{fdf['temperature'].mean():.1f}°C")
    c4.metric("🌧️ Avg Rainfall",  f"{fdf['rainfall'].mean():.1f} mm")
    c5.metric("⚠️ Extremes",      f"{fdf['extreme_event'].sum():,}")
    hi_risk = risk_df[risk_df["risk_level"] == "🔴 High"]["region"].tolist()
    c6.metric("🔴 High-Risk",     len(hi_risk))

    st.markdown("---")

    # ── Global temp trend ──────────────────────────────────────────────────
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.subheader("🌡️ Global Temperature Trend")
        ann = fdf.groupby("year")[["temperature", "temp_anomaly"]].mean().reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=ann["year"], y=ann["temperature"],
                       name="Avg Temperature (°C)", line=dict(color="#e74c3c", width=2)),
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(x=ann["year"], y=ann["temp_anomaly"],
                   name="Temp Anomaly (°C)",
                   marker_color=np.where(ann["temp_anomaly"] >= 0, "#e74c3c", "#3498db"),
                   opacity=0.6),
            secondary_y=True,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", secondary_y=True)
        fig.update_layout(height=340, margin=dict(t=20, b=20),
                          legend=dict(orientation="h", y=-0.15))
        fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
        fig.update_yaxes(title_text="Anomaly (°C)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("📝 Temperature has increased steadily since the 1980s, with anomalies accelerating post-2000.")

    with col_b:
        st.subheader("🛡️ Risk Distribution")
        risk_counts = risk_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig2 = px.pie(
            risk_counts, names="Risk Level", values="Count",
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
            hole=0.45,
        )
        fig2.update_layout(height=340, margin=dict(t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("📝 Risk distribution across all monitored regions.")

    # ── CO₂ & Sea Level ────────────────────────────────────────────────────
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("🏭 CO₂ Concentration")
        co2_ann = fdf.groupby("year")["co2"].mean().reset_index()
        fig3 = px.area(co2_ann, x="year", y="co2",
                       labels={"co2": "CO₂ (ppm)", "year": "Year"},
                       color_discrete_sequence=["#8e44ad"])
        fig3.update_layout(height=260, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("📝 CO₂ has risen from ~315 ppm in 1950 to over 420 ppm today.")

    with col_d:
        st.subheader("🌊 Sea Level Rise")
        sl_ann = fdf.groupby("year")["sea_level"].mean().reset_index()
        fig4 = px.area(sl_ann, x="year", y="sea_level",
                       labels={"sea_level": "Sea Level Anomaly (mm)", "year": "Year"},
                       color_discrete_sequence=["#2980b9"])
        fig4.update_layout(height=260, margin=dict(t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("📝 Sea levels are rising with increasing acceleration after 1990.")

    # ── Data quality ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Data Quality Summary")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Total Cells",      f"{qreport['total_cells']:,}")
    q2.metric("Missing Before",   qreport['missing_before'])
    q3.metric("Missing After",    qreport['missing_after'])
    q4.metric("% Data Affected",  f"{qreport['pct_affected']}%")
    st.info(f"**Fill Method:** {qreport['method']}")


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — Trend Analysis
# ─────────────────────────────────────────────────────────────────────────────
def page_trends(df: pd.DataFrame, opts: dict):
    st.title("📈 Climate Trend Analysis")
    st.markdown(
        "> *Where is it happening and how fast?* — Temperature & rainfall trends, "
        "rolling averages, anomaly charts, and decade-level slope analysis."
    )
    st.markdown("---")

    fdf = df[
        (df["region"].isin(opts["selected_regions"])) &
        (df["year"].between(*opts["year_range"]))
    ]

    region = st.selectbox("Select Region", sorted(fdf["region"].unique()), key="trend_region")
    rdf = fdf[fdf["region"] == region].sort_values("date")

    # ── Temperature trend with rolling avg + anomaly ───────────────────────
    st.subheader(f"🌡️ Temperature Analysis — {region}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rdf["date"], y=rdf["temperature"],
        name="Monthly Temp", mode="lines",
        line=dict(color="rgba(231,76,60,0.35)", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=rdf["date"], y=rdf["temp_roll12"],
        name="12-Month Rolling Avg", mode="lines",
        line=dict(color="#e74c3c", width=2.5),
    ))
    # Trend line
    if len(rdf) > 24:
        from sklearn.linear_model import LinearRegression
        x_num = ((rdf["date"] - rdf["date"].iloc[0]).dt.days.values / 365.25).reshape(-1, 1)
        y_num = rdf["temperature"].values
        mask = ~np.isnan(y_num)
        lr = LinearRegression().fit(x_num[mask], y_num[mask])
        trend_y = lr.predict(x_num)
        fig.add_trace(go.Scatter(
            x=rdf["date"], y=trend_y,
            name=f"Trend Line", mode="lines",
            line=dict(color="#8e44ad", width=2, dash="dash"),
        ))
    fig.update_layout(height=380, xaxis_title="Date",
                      yaxis_title="Temperature (°C)",
                      legend=dict(orientation="h", y=-0.18),
                      margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"📝 The dashed purple line shows the long-term trend for {region}. The red line is the 12-month rolling average smoothing out seasonal noise.")

    # ── Anomaly bar chart ──────────────────────────────────────────────────
    st.subheader(f"📊 Temperature Anomaly — {region} (Baseline: {opts['baseline_label']})")
    ann_anom = rdf.groupby("year")["temp_anomaly"].mean().reset_index()
    colors = ["#e74c3c" if v >= 0 else "#3498db" for v in ann_anom["temp_anomaly"]]
    fig2 = go.Figure(go.Bar(
        x=ann_anom["year"], y=ann_anom["temp_anomaly"],
        marker_color=colors, name="Annual Temp Anomaly",
    ))
    fig2.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    fig2.update_layout(height=300, xaxis_title="Year",
                       yaxis_title="Anomaly (°C)", margin=dict(t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("📝 Red bars = warmer than baseline; Blue bars = cooler. Notice the shift from blue-dominant to red-dominant after the 1980s.")

    # ── Rainfall trend ────────────────────────────────────────────────────
    st.subheader(f"🌧️ Rainfall Trend — {region}")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=rdf["date"], y=rdf["rainfall"],
        name="Monthly Rainfall", mode="lines",
        line=dict(color="rgba(52,152,219,0.3)", width=1),
    ))
    fig3.add_trace(go.Scatter(
        x=rdf["date"], y=rdf["rain_roll12"],
        name="12-Month Rolling Avg", mode="lines",
        line=dict(color="#2980b9", width=2.5),
    ))
    fig3.update_layout(height=300, xaxis_title="Date",
                       yaxis_title="Rainfall (mm)",
                       legend=dict(orientation="h", y=-0.18),
                       margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    # ── Decade slope table ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📐 Trend Slope per Decade — All Regions")
    trend_tbl = region_trend_table(df[df["region"].isin(opts["selected_regions"])])
    st.dataframe(
        trend_tbl[["temp_slope_per_decade", "rain_slope_per_decade",
                   "total_temp_change", "trend_direction"]]
        .style.background_gradient(subset=["temp_slope_per_decade"], cmap="RdYlGn_r")
        .format({"temp_slope_per_decade": "{:+.3f}°C/decade",
                 "rain_slope_per_decade": "{:+.2f} mm/decade",
                 "total_temp_change": "{:+.2f}°C"}),
        use_container_width=True,
    )
    st.caption("📝 Slope = linear regression coefficient × 10. Positive = warming trend. All regions show warming over the study period.")

    # ── Decade heatmap ─────────────────────────────────────────────────────
    st.subheader("🗓️ Decade Mean Temperature Heatmap")
    pivot = decade_mean_table(df[df["region"].isin(opts["selected_regions"])])
    fig4 = px.imshow(
        pivot, text_auto=".1f",
        color_continuous_scale="RdYlBu_r",
        labels=dict(x="Decade", y="Region", color="Temp (°C)"),
        aspect="auto",
    )
    fig4.update_layout(height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("📝 Each cell = mean temperature for that region-decade. Darker red = hotter. The rightward shift to deeper reds is unmistakable.")


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — Anomaly Detector
# ─────────────────────────────────────────────────────────────────────────────
def page_anomalies(df: pd.DataFrame, opts: dict):
    st.title("🚨 Extreme Event & Anomaly Detector")
    st.markdown(
        "> *Is it abnormal? How severe?* — Z-score based detection, severity "
        "classification, event timeline, and worst-case table."
    )
    st.markdown("---")

    fdf = df[
        (df["region"].isin(opts["selected_regions"])) &
        (df["year"].between(*opts["year_range"]))
    ]

    # ── Summary KPIs ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚠️ Total Extremes",    int(fdf["extreme_event"].sum()))
    c2.metric("🔥 Heatwaves",         int((fdf["event_type"] == "Heatwave").sum()))
    c3.metric("🌧️ Extreme Rainfall",  int((fdf["event_type"] == "Extreme Rainfall").sum()))
    c4.metric("💧 Drought-like",       int((fdf["event_type"] == "Drought-like").sum()))
    st.markdown("---")

    # ── Anomaly scatter timeline ────────────────────────────────────────────
    st.subheader("📍 Extreme Event Timeline — Temperature Z-Score")
    colors_map = {
        "Normal": "#bdc3c7", "Heatwave": "#e74c3c",
        "Cold Snap": "#3498db", "Extreme Rainfall": "#9b59b6",
        "Drought-like": "#e67e22", "Temperature Anomaly": "#f1c40f",
        "Rainfall Anomaly": "#1abc9c",
    }
    fig = px.scatter(
        fdf, x="date", y="temp_zscore",
        color="event_type", size="anomaly_score",
        size_max=14,
        color_discrete_map=colors_map,
        hover_data=["region", "temperature", "rainfall", "event_type"],
        labels={"temp_zscore": "Temperature Z-Score", "date": "Date"},
    )
    fig.add_hline(y=2,  line_dash="dash", line_color="#e74c3c",  annotation_text="Severe threshold (+2σ)")
    fig.add_hline(y=-2, line_dash="dash", line_color="#3498db",  annotation_text="Severe threshold (−2σ)")
    fig.add_hline(y=0,  line_dash="dot",  line_color="gray",     line_width=1)
    fig.update_layout(height=400, margin=dict(t=20, b=20),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("📝 Each dot is one region-month. Size = anomaly intensity. Dots above +2σ or below −2σ are classified as Severe/Critical.")

    # ── Event type breakdown ────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📊 Events per Region")
        summary = extreme_event_summary(fdf)
        summary_pivot = summary.pivot(index="region", columns="event_type", values="count").fillna(0)
        fig2 = px.bar(
            summary_pivot.reset_index().melt(id_vars="region"),
            x="region", y="value", color="event_type",
            barmode="stack",
            color_discrete_map=colors_map,
            labels={"value": "Count", "region": "Region", "event_type": "Event Type"},
        )
        fig2.update_layout(height=360, margin=dict(t=10, b=10),
                           legend=dict(orientation="h", y=-0.3),
                           xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.subheader("🗓️ Extreme Events per Year")
        ext_annual = (
            fdf[fdf["extreme_event"]]
            .groupby(["year", "event_type"])
            .size()
            .reset_index(name="count")
        )
        fig3 = px.area(
            ext_annual, x="year", y="count", color="event_type",
            color_discrete_map=colors_map,
            labels={"count": "Count", "year": "Year"},
        )
        fig3.update_layout(height=360, margin=dict(t=10, b=10),
                           legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Top critical events table ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔴 Top 20 Most Critical Events")
    top = top_extreme_events(fdf, 20)
    top["date"] = top["date"].dt.strftime("%b %Y")
    st.dataframe(
        top.style.background_gradient(subset=["anomaly_score"], cmap="Reds")
                 .format({"temperature": "{:.1f}°C",
                          "rainfall": "{:.1f} mm",
                          "temp_zscore": "{:+.2f}σ",
                          "rain_zscore": "{:+.2f}σ",
                          "anomaly_score": "{:.0f}/100"}),
        use_container_width=True,
    )
    st.caption("📝 Sorted by composite anomaly score. Scores approaching 100 indicate multi-standard-deviation extremes.")


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 — Regional Comparison
# ─────────────────────────────────────────────────────────────────────────────
def page_regional(df: pd.DataFrame, opts: dict):
    st.title("🗺️ Regional Climate Comparison")
    st.markdown(
        "> *Where is it happening?* — Side-by-side trend, anomaly, and "
        "volatility comparisons across selected regions."
    )
    st.markdown("---")

    all_regions = sorted(df["region"].unique().tolist())
    compare_regions = st.multiselect(
        "Select 2–5 regions to compare",
        all_regions,
        default=all_regions[:4],
        key="compare_select",
    )
    if len(compare_regions) < 2:
        st.warning("Please select at least 2 regions.")
        return

    variable = st.selectbox("Variable", ["temperature", "rainfall", "temp_anomaly"], key="comp_var")
    fdf = df[
        (df["region"].isin(compare_regions)) &
        (df["year"].between(*opts["year_range"]))
    ]

    # ── Annual mean comparison ─────────────────────────────────────────────
    st.subheader(f"📈 Annual Mean — {variable.replace('_', ' ').title()}")
    ann = fdf.groupby(["year", "region"])[variable].mean().reset_index()
    fig = px.line(
        ann, x="year", y=variable, color="region",
        color_discrete_sequence=REGION_COLORS,
        labels={"year": "Year", variable: variable.replace("_", " ").title()},
    )
    fig.update_layout(height=380, margin=dict(t=10, b=10),
                      legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("📝 Direct comparison of annual mean values. Diverging trajectories after 1980 are clearly visible.")

    # ── Trend slope bar ────────────────────────────────────────────────────
    st.subheader("📐 Warming Trend Slope per Decade")
    tbl = region_trend_table(fdf).reset_index()
    tbl = tbl[tbl["region"].isin(compare_regions)]
    fig2 = px.bar(
        tbl.sort_values("temp_slope_per_decade", ascending=False),
        x="region", y="temp_slope_per_decade",
        color="temp_slope_per_decade",
        color_continuous_scale="RdYlBu_r",
        text="temp_slope_per_decade",
        labels={"temp_slope_per_decade": "°C / Decade"},
    )
    fig2.update_traces(texttemplate="%{text:+.2f}", textposition="outside")
    fig2.update_layout(height=350, showlegend=False,
                       coloraxis_showscale=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("📝 Higher bars = faster warming. All bars positive = consistent warming across all regions.")

    # ── Rainfall volatility ranking ────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🌧️ Rainfall Volatility (CV)")
        rain_cv = (
            fdf.groupby("region")["rainfall"]
            .apply(lambda s: round(s.std() / s.mean(), 3) if s.mean() > 0 else 0)
            .reset_index(name="CV")
            .sort_values("CV", ascending=False)
        )
        fig3 = px.bar(rain_cv, x="CV", y="region",
                      orientation="h",
                      color="CV",
                      color_continuous_scale="Blues",
                      labels={"CV": "Coeff. of Variation"})
        fig3.update_layout(height=300, coloraxis_showscale=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("📝 Higher CV = more unpredictable rainfall.")

    with col_b:
        st.subheader("🌡️ Hottest vs Coolest Region (Avg Temp)")
        avg_temp = (
            fdf.groupby("region")["temperature"]
            .mean()
            .reset_index(name="avg_temp")
            .sort_values("avg_temp", ascending=False)
        )
        fig4 = px.bar(avg_temp, x="avg_temp", y="region",
                      orientation="h",
                      color="avg_temp",
                      color_continuous_scale="RdYlBu_r",
                      labels={"avg_temp": "Avg Temperature (°C)"})
        fig4.update_layout(height=300, coloraxis_showscale=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("📝 Sub-Saharan Africa and South Asia lead with highest baseline temperatures.")

    # ── Seasonal box plots ─────────────────────────────────────────────────
    st.subheader("📦 Temperature Distribution by Season & Region")
    fig5 = px.box(
        fdf, x="region", y="temperature",
        color="season",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        labels={"temperature": "Temperature (°C)"},
        notched=True,
    )
    fig5.update_layout(height=420, margin=dict(t=10, b=10),
                       xaxis_tickangle=-25,
                       legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("📝 Box plots show median, IQR, and outliers. Wider seasonal spread = more pronounced seasonality.")

    # ── Anomaly ranking ────────────────────────────────────────────────────
    st.subheader("🏅 Anomaly Ranking — Mean Absolute Temperature Anomaly")
    anom_rank = (
        fdf.groupby("region")["temp_anomaly"]
        .apply(lambda s: round(s.abs().mean(), 3))
        .reset_index(name="Mean |Anomaly| (°C)")
        .sort_values("Mean |Anomaly| (°C)", ascending=False)
    )
    st.dataframe(
        anom_rank.style.background_gradient(subset=["Mean |Anomaly| (°C)"], cmap="Oranges"),
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page 5 — Forecast
# ─────────────────────────────────────────────────────────────────────────────
def page_forecast(df: pd.DataFrame, opts: dict):
    st.title("🔭 Climate Forecast")
    st.markdown(
        "> *What might happen next?* — Temperature and rainfall projections "
        "with confidence intervals."
    )
    st.info(FORECAST_CAVEAT)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    region   = col1.selectbox("Region",   sorted(df["region"].unique()), key="fc_region")
    variable = col2.selectbox("Variable", ["temperature", "rainfall"], key="fc_var")
    periods  = col3.selectbox("Forecast Horizon", [12, 24, 60], index=1, key="fc_periods")

    # ── Get historical + forecast ──────────────────────────────────────────
    hist, fc = get_forecast(df, region, variable, periods)

    # ── Chart ──────────────────────────────────────────────────────────────
    st.subheader(f"📈 {variable.title()} Forecast — {region} ({periods} months)")
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["value"],
        name=f"Historical {variable.title()}",
        mode="lines",
        line=dict(color="#7f8c8d", width=1.5),
    ))

    # 12-month rolling
    roll = hist["value"].rolling(12, min_periods=6, center=True).mean()
    fig.add_trace(go.Scatter(
        x=hist["date"], y=roll,
        name="12-Month Avg",
        mode="lines",
        line=dict(color="#2ecc71" if variable == "rainfall" else "#e74c3c", width=2),
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([fc["date"], fc["date"].iloc[::-1]]),
        y=pd.concat([fc["upper_ci"], fc["lower_ci"].iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(41,128,185,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="90% Confidence Band",
        showlegend=True,
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["forecast"],
        name=f"Forecast ({fc['method'].iloc[0]})",
        mode="lines",
        line=dict(color="#2980b9", width=2.5, dash="dash"),
    ))

    fig.add_vline(
        x=hist["date"].iloc[-1].timestamp() * 1000,
        line_dash="dot", line_color="orange",
        annotation_text="Forecast Start",
    )

    unit = "°C" if variable == "temperature" else "mm"
    fig.update_layout(
        height=440,
        xaxis_title="Date",
        yaxis_title=f"{variable.title()} ({unit})",
        legend=dict(orientation="h", y=-0.18),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Forecast table ─────────────────────────────────────────────────────
    st.subheader("📋 Forecast Values")
    fc_disp = fc.copy()
    fc_disp["date"] = fc_disp["date"].dt.strftime("%b %Y")
    fc_disp.columns = ["Date", "Forecast", "Lower CI (90%)", "Upper CI (90%)", "Method"]
    unit_fmt = "{:.2f}°C" if variable == "temperature" else "{:.1f} mm"
    st.dataframe(
        fc_disp.style.format({"Forecast": unit_fmt,
                               "Lower CI (90%)": unit_fmt,
                               "Upper CI (90%)": unit_fmt}),
        use_container_width=True,
    )
    st.caption(
        f"📝 Method: Linear trend extrapolation. "
        f"The shaded band represents 90% confidence interval based on historical residuals."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page 6 — Risk Intelligence
# ─────────────────────────────────────────────────────────────────────────────
def page_risk(df: pd.DataFrame, risk_df: pd.DataFrame, opts: dict):
    st.title("🛡️ Climate Risk Intelligence")
    st.markdown(
        "> *What does it mean?* — Composite climate risk scores, "
        "regional rankings, score breakdown, and recommendations."
    )
    st.markdown("---")

    r = risk_df[risk_df["region"].isin(opts["selected_regions"])].copy()
    r = r.sort_values("risk_score", ascending=False)

    # ── Risk ranking chart ─────────────────────────────────────────────────
    st.subheader("🏅 Climate Risk Ranking — All Regions")
    color_list = [RISK_COLORS.get(lv, "#95a5a6") for lv in r["risk_level"]]
    fig = go.Figure(go.Bar(
        x=r["risk_score"], y=r["region"],
        orientation="h",
        marker_color=color_list,
        text=[f"{s:.0f}/100" for s in r["risk_score"]],
        textposition="outside",
        hovertext=r["primary_driver"],
    ))
    fig.add_vline(x=33, line_dash="dash", line_color="#2ecc71",  annotation_text="Low/Med")
    fig.add_vline(x=67, line_dash="dash", line_color="#e74c3c",  annotation_text="Med/High")
    fig.update_layout(
        height=400, xaxis_range=[0, 110],
        xaxis_title="Risk Score (0–100)",
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("📝 Scores > 67 = High Risk. Driven by a weighted combination of 5 climate indicators.")

    # ── Score components radar ──────────────────────────────────────────────
    st.subheader("🕸️ Risk Score Components")
    components = ["anomaly_freq", "warming_trend", "rain_volatility", "extreme_rate", "acceleration"]
    labels = ["Anomaly\nFrequency", "Warming\nTrend", "Rainfall\nVolatility", "Extreme\nEvent Rate", "Recent\nAcceleration"]

    fig2 = go.Figure()
    for _, row in r.iterrows():
        vals = [row[c] for c in components]
        max_vals = [r[c].max() for c in components]
        norm_vals = [v / m if m > 0 else 0 for v, m in zip(vals, max_vals)]
        fig2.add_trace(go.Scatterpolar(
            r=norm_vals + [norm_vals[0]],
            theta=labels + [labels[0]],
            name=row["region"],
            fill="toself",
            opacity=0.5,
        ))
    fig2.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=450,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("📝 Larger polygon = higher risk across more dimensions. Helps identify which factor is driving risk for each region.")

    # ── Detailed table ─────────────────────────────────────────────────────
    st.subheader("📋 Detailed Risk Table")
    disp = r[["region", "risk_score", "risk_level", "primary_driver",
               "anomaly_freq", "warming_trend", "rain_volatility",
               "extreme_rate", "acceleration"]].copy()
    st.dataframe(
        disp.style
            .background_gradient(subset=["risk_score"], cmap="RdYlGn_r")
            .format({
                "risk_score": "{:.1f}",
                "anomaly_freq": "{:.2%}",
                "warming_trend": "{:+.3f}",
                "rain_volatility": "{:.3f}",
                "extreme_rate": "{:.2f}",
                "acceleration": "{:.2f}x",
            }),
        use_container_width=True,
    )

    # ── Recommendations ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💬 Regional Recommendations")
    for _, row in r.iterrows():
        level = row["risk_level"]
        with st.expander(f"{level}  {row['region']}  —  Score: {row['risk_score']:.0f}/100"):
            if "High" in level:
                st.error(f"**Priority Region.** {row['primary_driver']} is the leading risk driver. "
                         f"Immediate adaptation planning and early-warning systems are recommended.")
            elif "Medium" in level:
                st.warning(f"**Monitoring Recommended.** {row['primary_driver']} is notable. "
                           f"Medium-term resilience investments would be beneficial.")
            else:
                st.success(f"**Currently Low Risk.** {row['primary_driver']} is relatively contained. "
                           f"Continue long-term monitoring.")


# ─────────────────────────────────────────────────────────────────────────────
# Page 7 — Insights
# ─────────────────────────────────────────────────────────────────────────────
def page_insights(df: pd.DataFrame, risk_df: pd.DataFrame, opts: dict):
    st.title("💡 Auto-Generated Climate Insights")
    st.markdown(
        "> Instead of reading 20 charts, here are the key findings "
        "automatically extracted from the data."
    )
    st.markdown("---")

    fdf = df[df["region"].isin(opts["selected_regions"])]
    r   = risk_df[risk_df["region"].isin(opts["selected_regions"])]

    st.subheader("⭐ Key Findings")
    for finding in key_findings(fdf, r):
        st.info(finding)

    st.markdown("---")
    st.subheader("📋 All Insights")
    for ins in generate_insights(fdf, r):
        st.markdown(f"- {ins}")

    # ── Export ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📥 Export Data")
    col1, col2 = st.columns(2)

    with col1:
        csv = fdf.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️  Download Full Dataset (CSV)",
            data=csv,
            file_name="climate_data_export.csv",
            mime="text/csv",
        )

    with col2:
        risk_csv = r.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️  Download Risk Scores (CSV)",
            data=risk_csv,
            file_name="climate_risk_scores.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Generate data if it doesn't exist
    if not os.path.exists(DATA_PATH):
        with st.spinner("Generating synthetic dataset for the first time …"):
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from data.generate_dataset import generate_climate_dataset
            os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
            df_gen = generate_climate_dataset()
            df_gen.to_csv(DATA_PATH, index=False)

    # Sidebar options (baseline not re-applied after initial load for speed)
    # We load data once with default baseline; sidebar baseline used for display label
    opts_prelim = {}
    baseline_label = st.sidebar.selectbox(
        "📏 Baseline Period",
        list(BASELINE_OPTIONS.keys()),
        index=2,
        key="_bl",
    ) if False else None  # will be handled inside sidebar()

    df, qreport = load_data((1991, 2020))  # cached load

    opts = sidebar(df)

    # Re-compute anomalies if baseline changed
    if opts["baseline"] != (1991, 2020):
        df_new = df.copy()
        from src.feature_engineering import compute_anomalies
        df_new = compute_anomalies(df_new, opts["baseline"])
        df = df_new

    risk_df = get_risk(df_to_json(df))

    page = opts["page"]

    if   page == "🏠 Overview":          page_overview(df, risk_df, qreport, opts)
    elif page == "📈 Trend Analysis":    page_trends(df, opts)
    elif page == "🚨 Anomaly Detector":  page_anomalies(df, opts)
    elif page == "🗺️ Regional Compare":  page_regional(df, opts)
    elif page == "🔭 Forecast":          page_forecast(df, opts)
    elif page == "🛡️ Risk Intelligence": page_risk(df, risk_df, opts)
    elif page == "💡 Insights":          page_insights(df, risk_df, opts)


if __name__ == "__main__":
    main()
