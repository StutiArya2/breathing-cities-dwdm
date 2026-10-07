"""Shared 'weather model': predicts a city's daily pollution from weather + calendar only.

Used three ways:
  forecast.py  adjust last year's monthly AQI for this year's weather
  lockdown.py  weather-normalised counterfactual: what would Mar-May 2020 have looked like?
  diwali.py    expected pollution on Diwali days without fireworks

Gradient boosting (HistGradientBoostingRegressor) on log(1 + value); handles missing weather.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = ["city_code", "doy_sin", "doy_cos", "dow", "t", "temp_mean", "humidity", "precipitation",
            "wind_mean", "wind_max", "radiation", "mixing_height", "mixing_height_lag1",
            "wind_mean_lag1", "precipitation_lag1", "rain_3d"]


def add_features(cube):
    df = cube.sort_values(["city", "full_date"]).copy()
    g = df.groupby("city")
    doy = df.full_date.dt.dayofyear / 365.25
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy)
    df["dow"] = df.full_date.dt.dayofweek
    df["t"] = (df.full_date - pd.Timestamp("2015-01-01")).dt.days / 365.25   # slow trend
    df["city_code"] = df.city.astype("category").cat.codes
    for col in ["mixing_height", "wind_mean", "precipitation"]:
        df[f"{col}_lag1"] = g[col].shift(1)
    df["rain_3d"] = g["precipitation"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    return df


def fit(df, target, mask):
    """Fit on rows where `mask` is true and the target is known."""
    train = df[mask & df[target].notna()]
    model = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, categorical_features=[0],
                                          random_state=42)
    return model.fit(train[FEATURES], np.log1p(train[target]))


def predict(model, df, freeze_t=None):
    """freeze_t pins the trend feature so only weather/calendar vary (pure weather effect)."""
    X = df[FEATURES].copy()
    if freeze_t is not None:
        X["t"] = freeze_t
    return np.expm1(model.predict(X))
