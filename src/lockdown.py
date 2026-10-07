"""Lockdown what-if: how much cleaner was the air during India's COVID lockdown (25 Mar - 31 May 2020)?

Weather-normalised counterfactual: the weather model (deweather.py) is trained on everything before
1 Mar 2020, then asked "given the actual weather of Mar-May 2020, what pollution would we expect?".
The gap between actual and expected is the lockdown effect, with weather luck removed.

Placebo check: the same procedure on 25 Mar - 31 May 2019 (no lockdown, trained before 1 Mar 2019)
should show an effect close to zero. If it does, the method is not just producing noise.
"""
import pandas as pd

import deweather
from db import OUTPUTS, get_engine, read_cube

POLLUTANTS = {"pm25": "PM2.5", "no2": "NO2", "aqi": "AQI"}
MIN_DAYS = 30


def effect(df, start, end, train_before):
    window = df.full_date.between(start, end)
    rows = []
    for col in POLLUTANTS:
        model = deweather.fit(df, col, df.full_date < train_before)
        part = df[window & df[col].notna()].assign(expected=lambda d: deweather.predict(model, d))
        by_city = part.groupby("city").agg(days=(col, "size"), actual=(col, "mean"), expected=("expected", "mean"))
        by_city = by_city[by_city.days >= MIN_DAYS]
        by_city["change_pct"] = (by_city.actual / by_city.expected - 1) * 100
        rows.append(by_city.assign(pollutant=POLLUTANTS[col]).reset_index())
    return pd.concat(rows)


def main():
    df = deweather.add_features(read_cube(get_engine()))
    real = effect(df, "2020-03-25", "2020-05-31", "2020-03-01")
    placebo = effect(df, "2019-03-25", "2019-05-31", "2019-03-01")

    OUTPUTS.mkdir(exist_ok=True)
    real.round(1).to_csv(OUTPUTS / "lockdown_effect.csv", index=False)
    placebo.round(1).to_csv(OUTPUTS / "lockdown_placebo_2019.csv", index=False)

    # Daily timeline for Delhi (chart in dashboard / deck)
    model = deweather.fit(df, "pm25", df.full_date < "2020-03-01")
    delhi = df[(df.city == "Delhi") & (df.full_date >= "2020-01-01")].copy()
    delhi["expected_pm25"] = deweather.predict(model, delhi)
    delhi[["full_date", "pm25", "expected_pm25"]].round(1).to_csv(OUTPUTS / "lockdown_delhi_daily.csv", index=False)

    summary = pd.DataFrame({
        "lockdown_2020": real.groupby("pollutant").change_pct.median(),
        "placebo_2019": placebo.groupby("pollutant").change_pct.median(),
        "cities": real.groupby("pollutant").city.nunique(),
    }).round(1)
    summary.to_csv(OUTPUTS / "lockdown_summary.csv")
    print("Median % change vs weather-normalised expectation:\n", summary.to_string())
    print("\nPM2.5 by city (2020):\n",
          real[real.pollutant == "PM2.5"].sort_values("change_pct").round(1).to_string(index=False))


if __name__ == "__main__":
    main()
