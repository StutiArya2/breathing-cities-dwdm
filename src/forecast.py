"""Prediction: forecast each city's monthly AQI for 2019 from data up to Dec 2018.

Models compared (MAPE on 2019, cities with >= 3 years of history and >= 10 months of 2019):
  seasonal_naive    same month last year (the baseline)
  holt_winters      additive damped trend + 12-month seasonality
  sarima            SARIMA(1,0,0)(0,1,1)12 on log AQI
  weather_adjusted  seasonal naive x (weather model for this month / weather model for the same
                    month last year). The weather model (deweather.py) is trained on <= 2018 only;
                    2019 weather comes from the fact_weather source and stands in for a monthly
                    weather outlook.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

import deweather
from db import OUTPUTS, get_engine, read_cube

warnings.filterwarnings("ignore")
MIN_MONTHS = 36


def mape(actual, pred):
    return float(np.mean(np.abs((pred - actual) / actual)) * 100)


def main():
    df = deweather.add_features(read_cube(get_engine()))
    model = deweather.fit(df, "aqi", df.year <= 2018)
    df["weather_aqi"] = deweather.predict(model, df, freeze_t=4.0)   # trend pinned at end-2018

    monthly = df.groupby(["city", pd.Grouper(key="full_date", freq="MS")])[["aqi", "weather_aqi"]].mean()
    rows, preds = [], []
    for city, m in monthly.groupby(level="city"):
        m = m.droplevel("city")
        hist = m.aqi[m.index < "2019-01-01"].dropna().asfreq("MS").interpolate()
        actual = m.aqi[m.index.year == 2019].dropna()
        if len(hist) < 12 or len(actual) < 10:
            continue
        idx = actual.index
        last_year = m.reindex(idx - pd.DateOffset(years=1))
        p = pd.DataFrame(index=idx)
        p["seasonal_naive"] = last_year.aqi.to_numpy()
        p["weather_adjusted"] = (last_year.aqi * m.weather_aqi.reindex(idx).to_numpy()
                                 / last_year.weather_aqi).to_numpy()
        if len(hist) >= MIN_MONTHS:
            p["holt_winters"] = ExponentialSmoothing(hist, trend="add", damped_trend=True, seasonal="add",
                                                     seasonal_periods=12).fit().forecast(12).reindex(idx)
            p["sarima"] = np.exp(SARIMAX(np.log(hist), order=(1, 0, 0), seasonal_order=(0, 1, 1, 12))
                                 .fit(disp=False).forecast(12)).reindex(idx)
        p = p.assign(actual=actual, city=city)
        preds.append(p)
        ok = p.seasonal_naive.notna()
        rows.append({"city": city, "long_history": len(hist) >= MIN_MONTHS,
                     **{c: mape(p.actual[ok], p[c][ok]) for c in
                        ["seasonal_naive", "holt_winters", "sarima", "weather_adjusted"] if c in p}})

    scores = pd.DataFrame(rows).round(1)
    OUTPUTS.mkdir(exist_ok=True)
    scores.to_csv(OUTPUTS / "forecast_backtest.csv", index=False)
    pd.concat(preds).rename_axis("month").round(1).to_csv(OUTPUTS / "forecast_2019.csv")
    print("MAPE % on 2019 monthly AQI:\n", scores.to_string(index=False))
    print("\nMean, cities with >= 3 years:", scores[scores.long_history].mean(numeric_only=True).round(1).to_dict())
    print("Mean, all cities:", scores[["seasonal_naive", "weather_adjusted"]].mean().round(1).to_dict())


if __name__ == "__main__":
    main()
