"""
notebooks/eda_script.py
========================
Exploratory Data Analysis — standalone Python script.
Mirrors what you would put in a Jupyter notebook.

Run:  python notebooks/eda_script.py
Outputs go to images/ folder.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import warnings; warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing       import load_and_clean
from src.feature_engineering import add_all_features
from src.anomaly_detection   import detect_extreme_events

DATA  = os.path.join(os.path.dirname(__file__), "..", "data", "climate_raw.csv")
OUT   = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(OUT, exist_ok=True)

print("Loading data …")
df, report = load_and_clean(DATA)
df = add_all_features(df)
df = detect_extreme_events(df)
print(f"Shape: {df.shape}  |  Columns: {list(df.columns)}\n")

# ── 1. Dataset snapshot ───────────────────────────────────────────────────────
print("="*50)
print("DATASET SNAPSHOT")
print("="*50)
print(df.describe().T[["count","mean","std","min","max"]].round(2))

# ── 2. Missing value summary ──────────────────────────────────────────────────
print("\nMissing values after cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# ── 3. Temperature distribution per region ────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=False)
axes = axes.flatten()
for i, (region, grp) in enumerate(df.groupby("region")):
    axes[i].hist(grp["temperature"], bins=30, color="#e74c3c", alpha=0.7, edgecolor="white")
    axes[i].set_title(region, fontsize=9, fontweight="bold")
    axes[i].set_xlabel("Temp (°C)", fontsize=8)
    axes[i].set_ylabel("Frequency", fontsize=8)
    axes[i].axvline(grp["temperature"].mean(), color="black", linestyle="--", linewidth=1)
plt.suptitle("Temperature Distribution per Region", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "eda_01_temp_distribution.png"), dpi=120)
plt.close()
print("\n✅  eda_01_temp_distribution.png")

# ── 4. Correlation heatmap ────────────────────────────────────────────────────
num_cols = ["temperature","rainfall","humidity","co2","sea_level","temp_anomaly","rain_anomaly"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            ax=ax, linewidths=0.5, vmin=-1, vmax=1)
ax.set_title("Correlation Matrix — Climate Variables", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "eda_02_correlation_heatmap.png"), dpi=120)
plt.close()
print("✅  eda_02_correlation_heatmap.png")

# ── 5. Anomaly over time ──────────────────────────────────────────────────────
ann_anom = df.groupby("year")["temp_anomaly"].mean().reset_index()
fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.bar(ann_anom["year"], ann_anom["temp_anomaly"],
              color=["#e74c3c" if v >= 0 else "#3498db" for v in ann_anom["temp_anomaly"]])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Year"); ax.set_ylabel("Anomaly (°C)")
ax.set_title("Global Mean Temperature Anomaly (Baseline: 1991–2020)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "eda_03_anomaly_time.png"), dpi=120)
plt.close()
print("✅  eda_03_anomaly_time.png")

# ── 6. Seasonal box plots ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
season_order = ["Spring", "Summer", "Autumn", "Winter"]
season_colors = {"Spring": "#2ecc71", "Summer": "#e74c3c",
                 "Autumn": "#e67e22", "Winter": "#3498db"}
for season in season_order:
    s_data = [
        df[(df["region"] == r) & (df["season"] == season)]["temperature"].values
        for r in sorted(df["region"].unique())
    ]
box = ax.boxplot(
    [df[df["season"] == s]["temperature"].values for s in season_order],
    labels=season_order, patch_artist=True,
    boxprops=dict(facecolor="lightblue", color="navy"),
    medianprops=dict(color="red", linewidth=2),
)
ax.set_ylabel("Temperature (°C)"); ax.set_title("Temperature by Season (All Regions)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "eda_04_seasonal_box.png"), dpi=120)
plt.close()
print("✅  eda_04_seasonal_box.png")

# ── 7. Event type distribution ────────────────────────────────────────────────
event_counts = df["event_type"].value_counts()
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#e74c3c","#3498db","#9b59b6","#e67e22","#f1c40f","#1abc9c","#bdc3c7"]
ax.bar(event_counts.index, event_counts.values, color=colors[:len(event_counts)])
ax.set_xlabel("Event Type"); ax.set_ylabel("Count")
ax.set_title("Distribution of Climate Event Types", fontweight="bold")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "eda_05_event_types.png"), dpi=120)
plt.close()
print("✅  eda_05_event_types.png")

# ── 8. CO2 vs Temperature scatter ─────────────────────────────────────────────
ann2 = df.groupby("year")[["co2","temperature"]].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(ann2["co2"], ann2["temperature"],
                c=ann2["year"], cmap="plasma", s=50, alpha=0.8)
plt.colorbar(sc, ax=ax, label="Year")
ax.set_xlabel("CO₂ (ppm)"); ax.set_ylabel("Global Avg Temp (°C)")
ax.set_title("CO₂ vs Global Temperature (Annual Means)", fontweight="bold")
# trend line
z = np.polyfit(ann2["co2"], ann2["temperature"], 1)
p = np.poly1d(z)
ax.plot(np.sort(ann2["co2"]), p(np.sort(ann2["co2"])),
        "r--", linewidth=2, label=f"Trend (slope={z[0]:.4f})")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "eda_06_co2_vs_temp.png"), dpi=120)
plt.close()
print("✅  eda_06_co2_vs_temp.png")

print(f"\n✅  All EDA charts saved to: {OUT}/")
print("Run 'python main.py' for the full pipeline + Streamlit dashboard.")
