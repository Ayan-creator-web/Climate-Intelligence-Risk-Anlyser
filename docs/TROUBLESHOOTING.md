# 🛠️ Troubleshooting Guide

---

## Installation Issues

### Error: `ModuleNotFoundError: No module named 'streamlit'`
**Cause:** Package not installed or wrong Python environment.
```bash
# Solution
pip install -r requirements.txt
# Verify
python -c "import streamlit; print(streamlit.__version__)"
```
**Prevention:** Always activate your virtual environment before running.

---

### Error: `ERROR: Could not find a version that satisfies the requirement kaleido`
**Cause:** kaleido has platform-specific wheels.
```bash
# Solution (use pip directly)
pip install kaleido==0.2.1
# If that fails:
pip install kaleido --pre
```

---

### Error: `statsmodels` installation fails on Windows
```bash
# Solution
pip install statsmodels --only-binary :all:
```

---

## Dataset Issues

### Error: `FileNotFoundError: Dataset not found at '…/data/climate_raw.csv'`
**Cause:** Dataset hasn't been generated yet.
```bash
# Solution
python data/generate_dataset.py
# Then retry
python main.py
```

---

### Error: Date column not parsing correctly
**Cause:** CSV date format mismatch.
```python
# Solution in preprocessing.py - already handled by:
df = pd.read_csv(path, parse_dates=["date"])
```

---

## Dashboard Issues

### Error: Streamlit dashboard shows blank page
**Cause:** Dataset not generated yet.
```bash
# Solution: Generate data first
python main.py --skip-charts
# Then launch dashboard
streamlit run app/dashboard.py
```

---

### Error: `DuplicateWidgetID` in Streamlit
**Cause:** Same `key=` used for two widgets.
**Solution:** Each `st.selectbox`, `st.slider`, `st.multiselect` needs a unique `key=` parameter. All widgets in this project already have unique keys.

---

### Dashboard loads slowly on first run
**Cause:** `@st.cache_data` hasn't cached yet.
**Solution:** Normal on first run only. Subsequent loads are fast. You can also pre-run:
```bash
python main.py
```

---

## Plotting Issues

### Error: `kaleido` not found when saving Plotly charts as PNG
```bash
pip install kaleido==0.2.1
```

---

### Matplotlib backend error on server/headless environment
```python
# Already handled in main.py and eda_script.py:
import matplotlib
matplotlib.use("Agg")
```

---

## Forecasting Issues

### Warning: ARIMA did not converge
**Cause:** Some time series are too short or non-stationary.
**Solution:** The code automatically falls back to linear regression. This is expected behavior.

---

### Forecast confidence interval is very wide
**Cause:** High variance in the historical series.
**Interpretation:** This is statistically correct — high variance = wider uncertainty. The dashboard displays this correctly.

---

## Git / GitHub Issues

### Error: `fatal: not a git repository`
```bash
cd Climate-Intelligence-Risk-Analyzer
git init
git add .
git commit -m "feat: initial project setup"
```

---

### Error: `remote origin already exists`
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/Climate-Intelligence-Risk-Analyzer.git
```

---

### Large file warning (CSV too big for GitHub free tier)
```bash
# Add to .gitignore:
echo "data/climate_raw.csv" >> .gitignore
echo "outputs/*.csv" >> .gitignore
# But keep the generator script — reviewers can regenerate the data
```

---

## Common Beginner Mistakes

| Mistake | Fix |
|---|---|
| Running `streamlit run app/dashboard.py` before generating data | Run `python main.py` first |
| Forgetting to activate virtual environment | `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows) |
| Committing `.env` files with API keys | Use `.gitignore` — already configured |
| Uploading the full CSV to GitHub (>25MB) | Add CSV to `.gitignore`, upload only scripts |
| Not setting unique `key=` in Streamlit widgets | Each widget in this project already has a unique key |
| Running from wrong directory | Always `cd Climate-Intelligence-Risk-Analyzer` first |
