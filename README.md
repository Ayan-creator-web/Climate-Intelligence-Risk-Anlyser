# 🌍 Climate Intelligence & Risk Analyzer

> **A portfolio-grade, end-to-end climate analytics system** built with Python and Streamlit —
> featuring anomaly detection, regional risk scoring, trend analysis, forecasting,
> and auto-generated insights aligned with NOAA / NASA / IPCC methodology.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Industry Relevance](#-industry-relevance)
4. [Tech Stack](#-tech-stack)
5. [Architecture](#-architecture)
6. [Folder Structure](#-folder-structure)
7. [Installation](#-installation)
8. [Dataset](#-dataset)
9. [How to Run](#-how-to-run)
10. [Dashboard Pages](#-dashboard-pages)
11. [Key Features](#-key-features)
12. [Results & Outputs](#-results--outputs)
13. [Screenshots](#-screenshots)
14. [Future Improvements](#-future-improvements)
15. [Learning Outcomes](#-learning-outcomes)
16. [Author](#-author)

---

## 🔍 Project Overview

**Climate Intelligence & Risk Analyzer** is a full-stack data science project that:

- Ingests monthly climate data for 8 global regions (1950–2023)
- Applies **temperature anomaly analysis** relative to a selectable baseline period
- Detects **extreme climate events** using z-score and IQR statistical methods
- Computes a **composite Climate Risk Score** (0–100) for each region
- Forecasts future temperature and rainfall with **confidence intervals**
- Delivers **auto-generated plain-English insights** from the data
- Presents everything in a **6-page interactive Streamlit dashboard**

---

## ❓ Problem Statement

Climate change is one of the most critical challenges of the 21st century.
However, raw weather data is often hard to interpret. Decision-makers —
from government planners to NGO analysts — need systems that can:

- Convert raw temperature readings into **anomaly-based views**
- Identify **where and when** unusual events are occurring
- **Rank regions** by climate risk
- **Project future trends** based on historical patterns
- Deliver **actionable summaries** without requiring data expertise

This project builds exactly that system.

---

## 🏢 Industry Relevance

| Organization | How they use similar tools |
|---|---|
| **NOAA** | Temperature anomaly charts, trend lines per decade |
| **NASA GISS** | Baseline comparison, global surface temperature index |
| **IPCC** | Regional climate information for risk assessment |
| **World Bank** | Climate risk screening for infrastructure projects |
| **Smart City Planners** | Heat island monitoring, seasonal risk alerts |

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Python 3.10+, Pandas, NumPy |
| Statistical Analysis | Scikit-learn, SciPy, Statsmodels |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Forecasting | Linear Regression, SARIMA |
| Version Control | Git + GitHub |

---

## 🏗️ Architecture

```
Raw CSV Data
     │
     ▼
┌─────────────────────┐
│  preprocessing.py   │  Load → Validate → Interpolate → Clip outliers
└────────┬────────────┘
         │
         ▼
┌──────────────────────────┐
│  feature_engineering.py  │  Anomalies · Rolling avg · Z-score · Decade labels
└────────┬─────────────────┘
         │
    ┌────┴──────────────────┐
    │                       │
    ▼                       ▼
┌──────────────┐    ┌────────────────────┐
│trend_analysis│    │ anomaly_detection  │
│  (slopes)    │    │  (z-score/IQR)     │
└──────┬───────┘    └─────────┬──────────┘
       │                      │
       ▼                      ▼
┌──────────────────────────────────────────┐
│           risk_scoring.py                │
│  Composite score from 5 components       │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│         insights_generator.py            │
│    Auto plain-English climate findings   │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│          app/dashboard.py                │
│    6-page Streamlit interactive UI       │
└──────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
Climate-Intelligence-Risk-Analyzer/
│
├── data/
│   ├── generate_dataset.py    ← Synthetic dataset generator
│   └── climate_raw.csv        ← Auto-generated on first run
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       ← Load, validate, clean
│   ├── feature_engineering.py ← Anomalies, rolling avg, z-scores
│   ├── trend_analysis.py      ← Linear regression slopes
│   ├── anomaly_detection.py   ← Extreme event detection
│   ├── forecasting.py         ← Linear + ARIMA forecasting
│   ├── risk_scoring.py        ← Composite risk score engine
│   └── insights_generator.py ← Auto-text insights
│
├── app/
│   └── dashboard.py           ← Streamlit dashboard (main UI)
│
├── notebooks/
│   └── exploratory_analysis.ipynb  ← Jupyter EDA notebook
│
├── outputs/                   ← Auto-generated CSVs
│   ├── climate_clean.csv
│   ├── risk_scores.csv
│   ├── extreme_events.csv
│   └── region_trends.csv
│
├── images/                    ← Auto-generated charts (PNG)
│   ├── 01_global_temp_anomaly.png
│   ├── 02_region_temp_trends.png
│   ├── 03_extreme_events_timeline.png
│   ├── 04_risk_scores.png
│   ├── 05_decade_heatmap.png
│   ├── 06_seasonal_pattern.png
│   └── 07_forecast.png
│
├── reports/
│   └── auto_insights.txt      ← Auto-generated text report
│
├── main.py                    ← CLI pipeline entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Step 1: Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Climate-Intelligence-Risk-Analyzer.git
cd Climate-Intelligence-Risk-Analyzer
```

### Step 2: Create a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify installation
```bash
python -c "import pandas, numpy, streamlit, plotly, sklearn, statsmodels; print('All packages OK')"
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Synthetically generated (realistic simulation) |
| Regions | 8 global regions |
| Time Range | 1950–2023 (monthly) |
| Total Records | ~6,720 rows |
| Variables | date, region, country, temperature, rainfall, humidity, co2, sea_level |
| Derived Features | temp_anomaly, rain_anomaly, temp_zscore, temp_roll12, trend_label, event_type, risk_score |

The dataset simulates real-world climate patterns including:
- Historical warming trend (~0.018°C/year base, accelerating post-1980)
- Seasonal temperature and rainfall cycles per region
- Realistic CO₂ growth curve (Keeling-inspired)
- Sea level rise with acceleration post-1990

---

## 🚀 How to Run

### Option A: Full CLI Pipeline (generates all outputs + charts)
```bash
python main.py
```

### Option B: Interactive Streamlit Dashboard
```bash
streamlit run app/dashboard.py
```
Then open your browser at: **http://localhost:8501**

### Option C: Generate dataset only
```bash
python data/generate_dataset.py
```

### Option D: Custom baseline period
```bash
python main.py --baseline-start 1951 --baseline-end 1980
```

---

## 📄 Dashboard Pages

| Page | What it shows |
|---|---|
| 🏠 **Overview** | KPIs, global trends, CO₂, sea level, data quality |
| 📈 **Trend Analysis** | Per-region trends, rolling avg, anomaly bars, decade heatmap |
| 🚨 **Anomaly Detector** | Extreme events timeline, severity, top-20 critical events |
| 🗺️ **Regional Compare** | Side-by-side multi-region comparison, ranking, boxplots |
| 🔭 **Forecast** | 12/24/60-month temperature/rainfall projections + CI bands |
| 🛡️ **Risk Intelligence** | Risk scores, radar chart, recommendations per region |
| 💡 **Insights** | Auto-generated findings, CSV export |

---

## ✨ Key Features

- **Anomaly Engine** — Compute anomalies relative to 3 selectable baseline periods
- **Extreme Event Detection** — Z-score + IQR based; severity: Normal → Critical
- **Trend Slope per Decade** — Linear regression slopes for all regions
- **Climate Risk Score** — Weighted composite of 5 components (0–100)
- **Forecasting** — Linear regression + SARIMA with 90% confidence bands
- **Auto-Insights** — 10+ plain-English findings auto-generated from data
- **Exportable** — Download cleaned data and risk scores as CSV
- **Story Mode** — Dashboard flows: Overview → Where → Anomaly → Risk → Forecast → Insight

---

## 📈 Results & Outputs

Sample findings from the auto-insights engine:

- Global average temperature has risen by ~**1.8°C** from the 1950s to the 2020s
- **Sub-Saharan Africa** is the fastest-warming region (+0.28°C/decade)
- **XX extreme events** detected across all regions (~15% of all monthly records)
- CO₂ has risen from **315 ppm → 421 ppm** over the study period
- Sea levels are ~**110 mm** above the 1950 baseline

---

## 📸 Screenshots

> After running `python main.py`, screenshots are saved in `/images/`

| Chart | File |
|---|---|
| Global Temp Anomaly | `images/01_global_temp_anomaly.png` |
| Region Trends | `images/02_region_temp_trends.png` |
| Extreme Events | `images/03_extreme_events_timeline.png` |
| Risk Scores | `images/04_risk_scores.png` |
| Decade Heatmap | `images/05_decade_heatmap.png` |
| Seasonal Pattern | `images/06_seasonal_pattern.png` |
| Forecast | `images/07_forecast.png` |

---

## 🔭 Future Improvements

1. **Live API Integration** — NOAA Climate Data Online, OpenWeatherMap
2. **Satellite Data** — NASA GISTEMP, ERA5 reanalysis datasets
3. **City-level Analysis** — Sub-national granularity
4. **Pollution Correlation** — PM2.5 / AQI alongside climate metrics
5. **Geospatial Dashboard** — Choropleth maps using Folium / Kepler.gl
6. **Deep Learning Forecast** — LSTM for multi-variable time series
7. **Automated PDF Report** — ReportLab-based scheduled reporting
8. **Alert System** — Email/Slack alerts when anomaly threshold exceeded

---

## 🎓 Learning Outcomes

- End-to-end data science pipeline from raw data to interactive dashboard
- Time-series feature engineering (anomalies, rolling stats, z-scores)
- Statistical anomaly detection (z-score, IQR thresholding)
- Climate trend analysis (linear regression slopes per decade)
- Composite scoring system design and normalisation
- Forecasting with confidence intervals
- Professional Streamlit dashboard development
- GitHub project documentation and proof-building strategy

---

## 👤 Author

**[Your Name]**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## ⚠️ Disclaimer

This project uses a **synthetically generated dataset** for portfolio demonstration purposes.
It is aligned with real-world climate analysis methodology (NOAA, NASA, IPCC) but does not
represent official climate data or projections. Forecasts are illustrative only.

---

*Built as a portfolio project for Data Analyst / Data Scientist / Environmental Data Analyst roles.*
