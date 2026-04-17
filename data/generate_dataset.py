"""
generate_dataset.py
-------------------
Generates a realistic synthetic climate dataset for 8 global regions
covering 1950–2023 at monthly frequency.

Features generated:
  date, year, month, quarter, season, region, country,
  temperature, rainfall, humidity, co2, sea_level
"""

import numpy as np
import pandas as pd
import os


# ─── Region Definitions ────────────────────────────────────────────────────────
REGIONS = {
    "South Asia": {
        "country": "India",
        "base_temp": 26.5,
        "base_rain": 95.0,
        "base_hum": 68,
        "rain_seasonal": [0.5,0.6,0.8,1.0,1.2,1.8,2.0,1.9,1.4,1.0,0.7,0.5],
        "temp_seasonal": [0.82,0.85,0.96,1.06,1.12,1.10,1.06,1.05,1.00,0.96,0.88,0.83],
    },
    "East Asia": {
        "country": "China",
        "base_temp": 14.0,
        "base_rain": 60.0,
        "base_hum": 58,
        "rain_seasonal": [0.6,0.7,0.9,1.0,1.1,1.4,1.6,1.5,1.1,0.9,0.7,0.6],
        "temp_seasonal": [0.60,0.65,0.80,1.00,1.15,1.25,1.30,1.25,1.10,0.95,0.75,0.62],
    },
    "Western Europe": {
        "country": "Germany",
        "base_temp": 10.5,
        "base_rain": 55.0,
        "base_hum": 75,
        "rain_seasonal": [0.9,0.85,0.88,0.9,1.0,1.05,0.95,1.0,1.05,1.1,1.1,0.95],
        "temp_seasonal": [0.50,0.55,0.72,0.92,1.10,1.22,1.30,1.28,1.15,0.95,0.68,0.52],
    },
    "North America": {
        "country": "USA",
        "base_temp": 12.0,
        "base_rain": 65.0,
        "base_hum": 60,
        "rain_seasonal": [0.85,0.85,0.95,1.0,1.05,1.1,1.0,0.95,0.95,0.95,0.95,0.9],
        "temp_seasonal": [0.50,0.55,0.72,0.92,1.10,1.22,1.30,1.26,1.12,0.92,0.65,0.50],
    },
    "Sub-Saharan Africa": {
        "country": "Nigeria",
        "base_temp": 28.0,
        "base_rain": 110.0,
        "base_hum": 72,
        "rain_seasonal": [0.3,0.4,0.8,1.2,1.5,1.6,1.4,1.3,1.6,1.4,0.8,0.4],
        "temp_seasonal": [1.02,1.04,1.06,1.05,1.02,0.98,0.95,0.96,0.99,1.01,1.02,1.02],
    },
    "South America": {
        "country": "Brazil",
        "base_temp": 25.0,
        "base_rain": 130.0,
        "base_hum": 78,
        "rain_seasonal": [1.5,1.4,1.3,1.0,0.7,0.5,0.5,0.6,0.8,1.0,1.2,1.4],
        "temp_seasonal": [1.05,1.05,1.02,0.98,0.94,0.91,0.91,0.93,0.96,1.00,1.03,1.05],
    },
    "Australia-Pacific": {
        "country": "Australia",
        "base_temp": 21.5,
        "base_rain": 45.0,
        "base_hum": 55,
        "rain_seasonal": [1.3,1.2,1.0,0.8,0.7,0.65,0.6,0.65,0.7,0.85,1.0,1.2],
        "temp_seasonal": [1.20,1.18,1.10,1.00,0.88,0.78,0.74,0.77,0.85,0.95,1.08,1.18],
    },
    "Northern Europe": {
        "country": "Sweden",
        "base_temp": 5.5,
        "base_rain": 50.0,
        "base_hum": 72,
        "rain_seasonal": [0.9,0.85,0.9,0.9,0.95,1.05,1.1,1.1,1.05,1.0,0.95,0.9],
        "temp_seasonal": [0.20,0.25,0.48,0.80,1.15,1.35,1.45,1.38,1.10,0.75,0.40,0.22],
    },
}


def _co2_value(year: int, rng: np.random.Generator) -> float:
    """Approximate Mauna Loa CO₂ curve."""
    base = 315.0
    linear = (year - 1960) * 1.35
    accel  = max(0, (year - 1990)) * 0.25
    return round(base + linear + accel + rng.normal(0, 0.4), 2)


def _sea_level(year: int, rng: np.random.Generator) -> float:
    """Approximate sea level rise (mm above 1950 baseline)."""
    rise = 1.5 * (year - 1950) + 0.015 * max(0, year - 1990) ** 1.6
    return round(rise + rng.normal(0, 3.0), 2)


def generate_climate_dataset(
    start_year: int = 1950,
    end_year: int = 2023,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="MS")

    # Warming trend: slow before 1980, faster after
    def warming(year: int) -> float:
        w = 0.014 * (year - start_year)
        if year > 1980:
            w += 0.006 * (year - 1980)
        if year > 2000:
            w += 0.004 * (year - 2000)
        return w

    records = []
    for region, props in REGIONS.items():
        for date in dates:
            y, m = date.year, date.month

            # Temperature
            t_sf = props["temp_seasonal"][m - 1]
            temp = (
                props["base_temp"] * t_sf
                + warming(y)
                + rng.normal(0, 0.75)
            )

            # Rainfall (slight long-term volatility increase)
            r_sf = props["rain_seasonal"][m - 1]
            rain_sigma = 10 + 0.05 * (y - start_year)
            rainfall = max(0.0, props["base_rain"] * r_sf + rng.normal(0, rain_sigma))

            # Humidity
            humidity = float(np.clip(props["base_hum"] + rng.normal(0, 5), 20, 100))

            # Season (Northern Hemisphere convention)
            season = (
                "Winter" if m in (12, 1, 2)
                else "Spring" if m in (3, 4, 5)
                else "Summer" if m in (6, 7, 8)
                else "Autumn"
            )

            records.append(
                dict(
                    date=date,
                    year=y,
                    month=m,
                    quarter=(m - 1) // 3 + 1,
                    season=season,
                    region=region,
                    country=props["country"],
                    temperature=round(float(temp), 2),
                    rainfall=round(float(rainfall), 2),
                    humidity=round(float(humidity), 2),
                    co2=_co2_value(y, rng),
                    sea_level=_sea_level(y, rng),
                )
            )

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["region", "date"]).reset_index(drop=True)


# ─── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)

    print("⏳ Generating synthetic climate dataset …")
    df = generate_climate_dataset()
    path = os.path.join(out_dir, "climate_raw.csv")
    df.to_csv(path, index=False)
    print(f"✅  Saved → {path}")
    print(f"   Shape  : {df.shape}")
    print(f"   Regions: {df['region'].nunique()}")
    print(f"   Range  : {df['date'].min().date()} → {df['date'].max().date()}")
    print(df.head(3))
