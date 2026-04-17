"""
src/forecasting.py
------------------
Temperature and rainfall forecasting using:
  - Linear regression extrapolation  (fast, explainable, default)
  - ARIMA (more accurate, requires statsmodels)

Returns forecast DataFrame with columns:
  date, forecast, lower_ci, upper_ci, method
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ─── Linear regression forecast ──────────────────────────────────────────────
def forecast_linear(
    series: pd.Series,
    dates: pd.Series,
    periods: int = 24,
    ci: float = 0.90,
) -> pd.DataFrame:
    """
    Fit a linear trend to (dates, values) and extrapolate `periods` months.

    Returns DataFrame with columns:
      date, forecast, lower_ci, upper_ci, method
    """
    mask = series.notna()
    x_all = ((dates - dates.iloc[0]).dt.days.values / 365.25).reshape(-1, 1)
    y_all = series.values

    x_fit = x_all[mask]
    y_fit = y_all[mask]

    model = LinearRegression().fit(x_fit, y_fit)
    residuals = y_fit - model.predict(x_fit)
    std_err = residuals.std()

    # z-multiplier for CI
    z = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960}.get(ci, 1.645)

    # future dates
    last_date = dates.iloc[-1]
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=periods, freq="MS")
    future_x = (
        ((future_dates - dates.iloc[0]).days / 365.25).values.reshape(-1, 1)
    )

    preds = model.predict(future_x)

    return pd.DataFrame(
        dict(
            date=future_dates,
            forecast=preds.round(3),
            lower_ci=(preds - z * std_err).round(3),
            upper_ci=(preds + z * std_err).round(3),
            method="Linear Regression",
        )
    )


# ─── ARIMA forecast ──────────────────────────────────────────────────────────
def forecast_arima(
    series: pd.Series,
    periods: int = 24,
    order: tuple = (2, 1, 2),
    seasonal_order: tuple = (1, 1, 1, 12),
    ci: float = 0.90,
) -> pd.DataFrame:
    """
    Fit a SARIMA model and forecast `periods` months ahead.
    Falls back to linear regression if statsmodels fails.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                series.dropna(),
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=200)

        alpha = 1 - ci
        forecast_obj = result.get_forecast(steps=periods)
        summary = forecast_obj.summary_frame(alpha=alpha)

        last_date = series.index[-1] if isinstance(series.index, pd.DatetimeIndex) else pd.Timestamp("1950-01-01")
        future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=periods, freq="MS")

        return pd.DataFrame(
            dict(
                date=future_dates,
                forecast=summary["mean"].values.round(3),
                lower_ci=summary["mean_ci_lower"].values.round(3),
                upper_ci=summary["mean_ci_upper"].values.round(3),
                method="SARIMA",
            )
        )
    except Exception as e:
        warnings.warn(f"ARIMA failed ({e}), falling back to linear regression.")
        idx = series.index if isinstance(series.index, pd.DatetimeIndex) else None
        if idx is None:
            return pd.DataFrame()
        return forecast_linear(series, pd.Series(idx), periods=periods, ci=ci)


# ─── Public wrapper ──────────────────────────────────────────────────────────
def get_forecast(
    df: pd.DataFrame,
    region: str,
    variable: str = "temperature",
    periods: int = 24,
    method: str = "linear",      # "linear" or "arima"
    ci: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper.

    Parameters
    ----------
    df       : enriched climate DataFrame
    region   : region name string
    variable : "temperature" or "rainfall"
    periods  : months to forecast
    method   : "linear" or "arima"

    Returns
    -------
    historical : DataFrame  (date, value, region)
    forecast   : DataFrame  (date, forecast, lower_ci, upper_ci, method)
    """
    grp = df[df["region"] == region].sort_values("date").copy()
    series = grp.set_index("date")[variable]

    dates_s = grp["date"].reset_index(drop=True)
    vals_s  = grp[variable].reset_index(drop=True)

    if method == "arima":
        fc = forecast_arima(series, periods=periods, ci=ci)
    else:
        fc = forecast_linear(vals_s, dates_s, periods, ci)

    historical = grp[["date", variable]].rename(columns={variable: "value"})
    historical["region"] = region
    return historical, fc


# ─── Forecast quality note ───────────────────────────────────────────────────
FORECAST_CAVEAT = (
    "⚠️  Forecasts are illustrative trend extrapolations based on historical data. "
    "They are not climate model projections. "
    "Recent data may be preliminary and subject to revision."
)


# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from src.preprocessing import load_and_clean
    from src.feature_engineering import add_all_features

    DATA = os.path.join(os.path.dirname(__file__), "..", "data", "climate_raw.csv")
    df, _ = load_and_clean(DATA)
    df = add_all_features(df)

    hist, fc = get_forecast(df, "South Asia", "temperature", periods=24)
    print("✅  Forecast (linear) for South Asia – Temperature")
    print(fc.head(6))
