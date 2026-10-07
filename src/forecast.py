"""Prediction: monthly AQI forecasting + the COVID-lockdown 'what-if'.

1. Backtest   - fit on data up to Dec 2018, forecast 2019, compare Holt-Winters
                (additive trend + 12-month seasonality) with a seasonal-naive baseline.
2. What-if    - fit on data up to Feb 2020 and forecast Mar-Jun 2020 as if there had
                been no lockdown; the gap between forecast and actual estimates the
                lockdown effect per city.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from db import OUTPUTS, get_engine, read_cube

warnings.filterwarnings("ignore")
MIN_MONTHS = 36


def monthly_series(cube):
    m = cube.set_index("full_date").groupby("city")["aqi"].resample("MS").mean()
    return m.unstack(0)  # rows = month, columns = city


def holt_winters(history, horizon):
    model = ExponentialSmoothing(history, trend="add", damped_trend=True,
                                 seasonal="add", seasonal_periods=12).fit()
    return model.forecast(horizon)


def mape(actual, pred):
    mask = actual.notna() & pred.notna()
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def main():
    series = monthly_series(read_cube(get_engine()))
    backtest, whatif = [], []
    for city in series.columns:
        s = series[city].dropna()
        hist = s[:"2018-12"].asfreq("MS").interpolate()
        if len(hist) >= MIN_MONTHS and len(s["2019"]) >= 6:
            actual = s["2019"]
            hw = holt_winters(hist, 12).reindex(actual.index)
            naive = pd.Series(hist[-12:].values, index=pd.date_range("2019-01-01", periods=12, freq="MS"))
            backtest.append({"city": city, "mape_holt_winters": mape(actual, hw),
                             "mape_seasonal_naive": mape(actual, naive.reindex(actual.index))})

        pre = s[:"2020-02"].asfreq("MS").interpolate()
        actual = s["2020-03":"2020-06"]
        if len(pre) >= MIN_MONTHS and len(actual) >= 2:
            expected = holt_winters(pre, 4).reindex(actual.index)
            whatif.append({"city": city, "expected_aqi": expected.mean(), "actual_aqi": actual.mean(),
                           "lockdown_effect_pct": (actual.mean() / expected.mean() - 1) * 100})

    OUTPUTS.mkdir(exist_ok=True)
    series.round(1).to_csv(OUTPUTS / "monthly_aqi.csv")
    bt = pd.DataFrame(backtest).round(1)
    wi = pd.DataFrame(whatif).round(1).sort_values("lockdown_effect_pct")
    bt.to_csv(OUTPUTS / "forecast_backtest.csv", index=False)
    wi.to_csv(OUTPUTS / "lockdown_whatif.csv", index=False)
    print("Backtest (2019) MAPE %:\n", bt.to_string(index=False))
    print("\nLockdown what-if (Mar-Jun 2020):\n", wi.to_string(index=False))


if __name__ == "__main__":
    main()
