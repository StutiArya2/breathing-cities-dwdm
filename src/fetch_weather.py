"""Second source: daily weather for every city from the Open-Meteo historical API (ERA5).

No API key needed. Writes data/raw/weather_daily.csv (one row per city per day).

    python src/fetch_weather.py
"""
import time

import pandas as pd
import requests

from db import ROOT

URL = "https://archive-api.open-meteo.com/v1/archive"
START, END = "2015-01-01", "2020-07-01"
DAILY = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
         "relative_humidity_2m_mean", "precipitation_sum", "wind_speed_10m_mean",
         "wind_speed_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum"]
RENAME = {"temperature_2m_mean": "temp_mean", "temperature_2m_max": "temp_max",
          "temperature_2m_min": "temp_min", "relative_humidity_2m_mean": "humidity",
          "precipitation_sum": "precipitation", "wind_speed_10m_mean": "wind_mean",
          "wind_speed_10m_max": "wind_max", "wind_direction_10m_dominant": "wind_dir",
          "shortwave_radiation_sum": "radiation"}


def get(params, tries=5):
    for attempt in range(tries):
        r = requests.get(URL, params=params, timeout=120)
        if r.status_code == 200:
            return r.json()
        time.sleep(15 * (attempt + 1))  # free tier is rate-limited
    r.raise_for_status()


def fetch_city(row):
    base = {"latitude": row.latitude, "longitude": row.longitude, "start_date": START,
            "end_date": END, "timezone": "Asia/Kolkata"}
    daily = pd.DataFrame(get({**base, "daily": ",".join(DAILY)})["daily"])
    hourly = pd.DataFrame(get({**base, "hourly": "boundary_layer_height"})["hourly"])
    # Mixing height: how high pollution can spread. Low = trapped smog.
    hourly["time"] = pd.to_datetime(hourly["time"]).dt.strftime("%Y-%m-%d")
    blh = hourly.groupby("time")["boundary_layer_height"].mean().rename("mixing_height")
    return daily.rename(columns=RENAME).merge(blh, left_on="time", right_index=True, how="left") \
        .rename(columns={"time": "date"}).assign(city=row.city)


def main():
    cities = pd.read_csv(ROOT / "data" / "city_meta.csv")
    frames = []
    for row in cities.itertuples():
        frames.append(fetch_city(row))
        print(f"{row.city}: {len(frames[-1])} days")
        time.sleep(1)
    out = ROOT / "data" / "raw" / "weather_daily.csv"
    pd.concat(frames)[["city", "date", *RENAME.values(), "mixing_height"]].round(2).to_csv(out, index=False)
    print("Wrote", out)


if __name__ == "__main__":
    main()
