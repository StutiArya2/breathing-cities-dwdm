"""Generate a SYNTHETIC file shaped like Kaggle's city_day.csv for smoke-testing the pipeline.

The numbers are random and must never be reported as findings. Use the real
CPCB/Kaggle file for the project (see README).

    python scripts/make_synthetic_data.py   ->  data/sample/city_day_synthetic.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(7)
cities = pd.read_csv(ROOT / "data" / "city_meta.csv")
dates = pd.date_range("2015-01-01", "2020-07-01", freq="D")

rows = []
for _, c in cities.iterrows():
    base = 140 if c.region == "North" else 70 if c.is_coastal else 95
    start = dates[rng.integers(0, 900)]
    d = dates[dates >= start]
    season = 1 + 0.6 * np.cos((d.dayofyear - 15) / 365 * 2 * np.pi)
    lockdown = np.where((d >= "2020-03-25") & (d <= "2020-05-31"), 0.55, 1.0)
    pm25 = np.clip(base * 0.6 * season * lockdown * rng.lognormal(0, 0.25, len(d)), 3, None)
    frame = pd.DataFrame({
        "City": c.city, "Date": d.strftime("%Y-%m-%d"), "PM2.5": pm25,
        "PM10": pm25 * rng.uniform(1.4, 2.0, len(d)), "NO": rng.gamma(2, 8, len(d)),
        "NO2": rng.gamma(3, 9, len(d)) * lockdown, "NOx": rng.gamma(3, 11, len(d)),
        "NH3": rng.gamma(2, 10, len(d)), "CO": rng.gamma(2, 0.6, len(d)),
        "SO2": rng.gamma(2, 7, len(d)), "O3": rng.gamma(4, 9, len(d)),
        "Benzene": rng.gamma(1.5, 2, len(d)), "Toluene": rng.gamma(1.5, 6, len(d)),
        "Xylene": rng.gamma(1, 2, len(d)),
    })
    frame["AQI"] = np.clip(frame["PM2.5"] * 1.9 + rng.normal(0, 15, len(d)), 15, 600)
    rows.append(frame)

df = pd.concat(rows, ignore_index=True)
for col in ["PM2.5", "PM10", "NH3", "Xylene", "AQI"]:  # mimic real missingness
    df.loc[rng.random(len(df)) < (0.5 if col == "Xylene" else 0.12), col] = np.nan
bins = [0, 50, 100, 200, 300, 400, np.inf]
df["AQI_Bucket"] = pd.cut(df.AQI, bins, labels=["Good", "Satisfactory", "Moderate", "Poor",
                                                "Very Poor", "Severe"]).astype(object)
out = ROOT / "data" / "sample" / "city_day_synthetic.csv"
out.parent.mkdir(exist_ok=True)
df.round(2).to_csv(out, index=False)
print(f"Wrote {len(df):,} synthetic rows to {out}")
