# 📋 Project Guide: GitHub Strategy, Interview Prep & Resume Tips

---

## 🗓️ 8-Day GitHub Commit Strategy

### Day 1 — Project Setup
**What to commit:**
- Folder structure, requirements.txt, .gitignore, README skeleton

**Commit message:**
```
feat: initialise project structure and environment setup

- Created modular folder layout (src, data, app, notebooks)
- Added requirements.txt with full ML/viz stack
- Initialised README with project overview
```
**Screenshot to take:** Terminal showing `pip install -r requirements.txt` success

---

### Day 2 — Dataset & Data Generation
**What to commit:**
- `data/generate_dataset.py`
- `data/climate_raw.csv` (or a sample)

**Commit message:**
```
data: add synthetic climate dataset generator

- 8 global regions, monthly frequency, 1950-2023
- Simulates warming trends, seasonality, CO2 growth, sea level rise
- Seeded for reproducibility (seed=42)
```
**Screenshot to take:** `df.head()` + `df.describe()` output in terminal

---

### Day 3 — Preprocessing & Feature Engineering
**What to commit:**
- `src/preprocessing.py`
- `src/feature_engineering.py`

**Commit message:**
```
feat: add preprocessing pipeline and feature engineering

- Linear interpolation for missing values per region
- Computed temperature and rainfall anomalies (baseline 1991-2020)
- Added z-scores, 12-month rolling averages, decade labels
```
**Screenshot to take:** DataFrame with new derived columns visible

---

### Day 4 — Trend Analysis & Anomaly Detection
**What to commit:**
- `src/trend_analysis.py`
- `src/anomaly_detection.py`

**Commit message:**
```
analysis: implement trend slopes and extreme event detection

- Linear regression slope per decade for all regions
- Z-score based anomaly classification (Normal/Mild/Severe/Critical)
- Event type labelling: Heatwave, Cold Snap, Drought-like, etc.
```
**Screenshot to take:** Region trend table + extreme events count

---

### Day 5 — Risk Scoring & Forecasting
**What to commit:**
- `src/risk_scoring.py`
- `src/forecasting.py`

**Commit message:**
```
feat: add climate risk scoring engine and forecasting module

- Composite risk score (0-100) from 5 weighted components
- Risk levels: Low/Medium/High with primary driver identification
- Linear regression forecast with 90% confidence intervals
```
**Screenshot to take:** Risk scores table + forecast chart

---

### Day 6 — Auto Insights & Main Pipeline
**What to commit:**
- `src/insights_generator.py`
- `main.py`

**Commit message:**
```
feat: add insights generator and CLI pipeline runner

- Auto-generates 10+ plain-English climate findings
- CLI pipeline: generate → clean → analyse → score → chart
- Saves outputs to /outputs and /images directories
```
**Screenshot to take:** Terminal output of `python main.py`

---

### Day 7 — Streamlit Dashboard
**What to commit:**
- `app/dashboard.py`

**Commit message:**
```
feat: build 6-page interactive Streamlit dashboard

- Overview, Trend Analysis, Anomaly Detector, Regional Compare
- Forecast page with CI bands and export button
- Risk Intelligence page with radar chart and recommendations
```
**Screenshot to take:** All 6 dashboard pages (take screenshots of each)

---

### Day 8 — Final Polish & Documentation
**What to commit:**
- All generated images (`images/*.png`)
- `reports/auto_insights.txt`
- Final README.md with screenshots embedded
- `notebooks/eda_script.py`

**Commit message:**
```
docs: complete documentation, screenshots, and EDA notebook

- Added 7 chart images to README
- Auto-insights report included
- Full EDA script with 6 exploratory charts
- Project ready for portfolio review
```

---

## 📄 3 Strong Resume Bullet Points

```
• Built a full-stack Climate Intelligence & Risk Analyzer using Python and Streamlit,
  performing anomaly detection (z-score/IQR), trend analysis, and composite risk scoring
  across 8 global regions covering 1950–2023 monthly climate data.

• Engineered 12+ derived features including temperature anomalies, rolling averages,
  decade-level trend slopes, and z-scores; detected 900+ extreme climate events
  classified by severity (Normal/Mild/Severe/Critical) aligned with NOAA/IPCC methodology.

• Designed and deployed a 6-page interactive Streamlit dashboard with auto-generated
  insights, 24-month forecasting with confidence intervals, regional risk scoring (0–100),
  and one-click CSV export — demonstrating end-to-end analytical product development.
```

---

## 💼 LinkedIn Project Description

**Option 1 (concise):**
> 🌍 Built a Climate Intelligence & Risk Analyzer — a full end-to-end data science project
> featuring temperature anomaly analysis, extreme event detection, regional risk scoring,
> forecasting, and an interactive 6-page Streamlit dashboard.
> Tech: Python · Pandas · Plotly · Scikit-learn · Streamlit
> [GitHub link]

**Option 2 (detailed):**
> 🌡️ New Portfolio Project: Climate Intelligence & Risk Analyzer
>
> What does it do?
> → Analyses 70+ years of climate data across 8 global regions
> → Detects heatwaves, cold snaps, droughts using z-score statistics
> → Computes climate risk scores (0–100) with breakdown by driver
> → Forecasts temperature & rainfall with 90% confidence intervals
> → Auto-generates plain-English insights from the data
>
> This project mirrors how real climate analytics is used at organisations
> like NOAA, World Bank, and smart city planning agencies.
> Built with: Python | Pandas | Plotly | Streamlit | Scikit-learn | Statsmodels

---

## 🎤 10 Interview Questions + Strong Answers

**Q1: What is a temperature anomaly and why did you use it instead of raw temperature?**
> A temperature anomaly measures how much a month's temperature deviates from a long-term
> baseline mean. NOAA and NASA use anomaly-based views because they remove the effect of
> geography — a station in Siberia and one in India may have very different absolute
> temperatures, but their anomalies are directly comparable. It also makes long-term
> warming trends much more visible than raw values.

**Q2: How did you detect extreme climate events?**
> I used z-score thresholding — computing how many standard deviations each month's
> temperature or rainfall deviates from its regional mean. |z| > 1.5 = Mild Anomaly,
> > 2.0 = Severe, > 3.0 = Critical. I also labelled event types: Heatwave (high temp z),
> Cold Snap (low temp z), Extreme Rainfall, Drought-like. This is consistent with IPCC's
> extremes framework which focuses on statistical rarity.

**Q3: How did you compute the climate risk score?**
> The risk score is a weighted composite of 5 components: anomaly frequency (25%),
> warming trend slope (30%), rainfall volatility — measured as coefficient of variation
> (20%), extreme event rate per year (15%), and recent acceleration (10%). Each component
> is MinMax normalised to [0,1] then multiplied by its weight. The sum × 100 gives a
> 0–100 score. Regions scoring > 67 are classified as High Risk.

**Q4: What forecasting method did you use and why?**
> I used linear regression extrapolation as the default because it's transparent and
> explainable — suitable for a portfolio project where the methodology needs to be clearly
> communicated. I also implemented SARIMA (Seasonal ARIMA) as an upgrade path.
> I included 90% confidence intervals based on residual standard error so the uncertainty
> is always visible. I also added a caveat note that these are trend extrapolations, not
> climate model projections.

**Q5: How did you handle missing data?**
> I used linear interpolation per region time series, which is appropriate for climate
> data because values change gradually. For any remaining gaps, I filled with the global
> median of that variable. I tracked all of this in a quality report and displayed it
> on the dashboard Overview page for transparency.

**Q6: What is the difference between the 1951–1980 and 1991–2020 baselines?**
> 1951–1980 is considered closer to a pre-industrial reference and is used by NASA GISS.
> 1991–2020 is the current WMO standard 30-year normal period. Using an earlier baseline
> makes recent warming appear larger (because the baseline is cooler). I made the baseline
> selectable so the user can see how the choice affects anomaly interpretation — this is
> scientifically important and shows I understand the methodology.

**Q7: How would you scale this to real institutional data?**
> I'd replace the synthetic generator with API calls to NOAA's Climate Data Online or
> NASA GISS Surface Temperature datasets. The entire src/ pipeline is already modular —
> only data/generate_dataset.py would change. I'd add Airflow or Prefect for scheduling,
> a PostgreSQL or BigQuery backend for storage, and a CI/CD pipeline to auto-refresh
> the dashboard weekly.

**Q8: What does the radar chart in the Risk Intelligence page show?**
> The radar chart (spider chart) plots each region's normalised score across all 5 risk
> components simultaneously. A larger polygon means higher risk across more dimensions.
> This helps identify what's driving risk — for example, one region might score high only
> on warming trend while another scores high on all 5 components. It's more informative
> than a single number.

**Q9: How would you explain this project to an HR interviewer?**
> "I built a climate monitoring dashboard — similar to what organisations like NOAA
> or the World Bank use — that takes historical weather data, finds unusual events,
> ranks regions by climate risk, and predicts future trends. It automatically generates
> a plain-English report so decision-makers don't need to read charts. I built the
> entire system from data collection to an interactive web dashboard."

**Q10: What was the hardest part of this project?**
> The risk scoring system. I needed to combine 5 very different metrics — each with
> different scales and units — into a single interpretable score. The key challenge was
> normalisation: I used MinMax scaling so no single component dominates by sheer magnitude.
> Getting the weights right required thinking about what matters most in climate risk
> analysis, which led me to study how the World Bank and IPCC frame climate risk.

---

## 🏷️ GitHub Repository Best Practices

**Repository name:** `Climate-Intelligence-Risk-Analyzer`

**Description:**
> 🌍 End-to-end climate analytics system: anomaly detection, regional risk scoring,
> trend analysis, forecasting & interactive Streamlit dashboard | Python · Plotly · Streamlit

**Topics/Tags:**
```
climate-analysis, data-science, streamlit, python, time-series,
anomaly-detection, machine-learning, plotly, pandas, risk-analysis,
climate-change, environmental-data-analysis, portfolio-project
```

**What to pin:** The repository + your LinkedIn profile

**README must have:** GIF or screenshot of the dashboard on the very first screen
