"""Extract-Transform-Load: three sources -> fact constellation.

Sources (all in data/raw/ unless noted):
    city_day.csv, station_day.csv, stations.csv   CPCB via Kaggle
    weather_daily.csv                             Open-Meteo, made by src/fetch_weather.py
    data/city_meta.csv                            our own city lookup (state, region, coordinates)

Usage:
    python src/etl.py
    python src/etl.py --input path/to/city_day.csv     # e.g. the synthetic test file
"""
import argparse
import hashlib

import numpy as np
import pandas as pd

from db import POLLUTANTS, ROOT, get_engine, run_script

SOURCE_COLUMNS = {"City": "city", "Date": "date", "PM2.5": "pm25", "PM10": "pm10", "NO": "no",
                  "NO2": "no2", "NOx": "nox", "NH3": "nh3", "CO": "co", "SO2": "so2", "O3": "o3",
                  "Benzene": "benzene", "Toluene": "toluene", "Xylene": "xylene", "AQI": "aqi",
                  "AQI_Bucket": "aqi_bucket"}
STATION_COLUMNS = {"StationId": "station_id", "Date": "date", "PM2.5": "pm25", "PM10": "pm10",
                   "NO2": "no2", "CO": "co", "SO2": "so2", "O3": "o3", "AQI": "aqi"}
STATION_MEASURES = ["pm25", "pm10", "no2", "co", "so2", "o3", "aqi"]

# CPCB National AQI buckets; key 0 holds days with no AQI reported.
AQI_CATEGORIES = pd.DataFrame([
    (0, "Unknown", None, None, "Unknown"),
    (1, "Good", 0, 50, "Healthy"),
    (2, "Satisfactory", 51, 100, "Healthy"),
    (3, "Moderate", 101, 200, "Healthy"),
    (4, "Poor", 201, 300, "Unhealthy"),
    (5, "Very Poor", 301, 400, "Unhealthy"),
    (6, "Severe", 401, 500, "Unhealthy"),
], columns=["category_key", "bucket", "aqi_min", "aqi_max", "health_group"])

EVENTS = pd.DataFrame([
    (1, "Normal", "Baseline", "No special event"),
    (2, "Diwali", "Festival", "Diwali day -2 to +3 (firecrackers)"),
    (3, "Stubble burning", "Agricultural", "15 Oct - 30 Nov, North region crop-residue fires"),
    (4, "COVID lockdown", "Policy", "National lockdown phases 1-4, 25 Mar - 31 May 2020"),
], columns=["event_key", "event_name", "event_type", "description"])

DIWALI = pd.to_datetime(["2015-11-11", "2016-10-30", "2017-10-19", "2018-11-07",
                         "2019-10-27", "2020-11-14"])
LOCKDOWN = (pd.Timestamp("2020-03-25"), pd.Timestamp("2020-05-31"))
SEASON_BY_MONTH = {1: "Winter", 2: "Winter", 3: "Pre-monsoon", 4: "Pre-monsoon",
                   5: "Pre-monsoon", 6: "Monsoon", 7: "Monsoon", 8: "Monsoon",
                   9: "Monsoon", 10: "Post-monsoon", 11: "Post-monsoon", 12: "Post-monsoon"}

# Weather bands for dim_weather_condition: (labels, lower edges).
WIND_BANDS = (["Calm", "Light", "Breezy"], [0, 6, 12])        # km/h, daily mean
RAIN_BANDS = (["Dry", "Rain"], [0, 1])                         # mm per day
TEMP_BANDS = (["Cold", "Mild", "Hot"], [-50, 18, 28])         # deg C, daily mean
MIX_BANDS = (["Low", "Medium", "High"], [0, 400, 800])         # m, boundary-layer height
WEATHER_MEASURES = ["temp_mean", "temp_max", "temp_min", "humidity", "precipitation", "wind_mean",
                    "wind_max", "wind_dir", "radiation", "mixing_height"]


def extract(path):
    df = pd.read_csv(path)
    missing = set(SOURCE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    print(f"Extracted {len(df):,} rows x {df.shape[1]} columns "
          f"(md5 {hashlib.md5(open(path, 'rb').read()).hexdigest()[:10]})")
    return df.rename(columns=SOURCE_COLUMNS)


def date_key(dates):
    return pd.to_datetime(dates).dt.strftime("%Y%m%d").astype(int)


def build_dim_date(dates):
    d = pd.DataFrame({"full_date": pd.date_range(dates.min(), dates.max(), freq="D")})
    d["date_key"] = date_key(d.full_date)
    d["day"] = d.full_date.dt.day
    d["day_name"] = d.full_date.dt.day_name()
    d["is_weekend"] = (d.full_date.dt.dayofweek >= 5).astype(int)
    d["week"] = d.full_date.dt.isocalendar().week.astype(int)
    d["month"] = d.full_date.dt.month
    d["month_name"] = d.full_date.dt.month_name()
    d["season"] = d.month.map(SEASON_BY_MONTH)
    d["quarter"] = d.full_date.dt.quarter
    d["year"] = d.full_date.dt.year
    d["full_date"] = d.full_date.dt.date
    return d[["date_key", "full_date", "day", "day_name", "is_weekend", "week", "month",
              "month_name", "season", "quarter", "year"]]


def build_dim_weather():
    rows = [(0, "Unknown", "Unknown", "Unknown", "Unknown", "No weather data")]
    for w in WIND_BANDS[0]:
        for r in RAIN_BANDS[0]:
            for t in TEMP_BANDS[0]:
                for m in MIX_BANDS[0]:
                    rows.append((len(rows), w, r, t, m, f"{w}, {r}, {t}, {m} mixing"))
    return pd.DataFrame(rows, columns=["weather_key", "wind_band", "rain_band", "temp_band",
                                       "mixing_band", "label"])


def band(values, spec):
    labels, edges = spec
    return pd.cut(values, edges + [np.inf], labels=labels, right=False).astype(object)


def assign_weather_key(weather, dim_weather):
    bands = pd.DataFrame({"wind_band": band(weather.wind_mean, WIND_BANDS),
                          "rain_band": band(weather.precipitation, RAIN_BANDS),
                          "temp_band": band(weather.temp_mean, TEMP_BANDS),
                          "mixing_band": band(weather.mixing_height, MIX_BANDS)})
    keys = bands.merge(dim_weather, how="left", on=["wind_band", "rain_band", "temp_band", "mixing_band"])
    return keys.weather_key.fillna(0).astype(int).to_numpy()


def assign_event(date, region):
    """Priority: lockdown > Diwali window > stubble burning (North only) > normal."""
    event = np.ones(len(date), dtype=int)
    stubble = (region == "North") & (
        ((date.dt.month == 10) & (date.dt.day >= 15)) | (date.dt.month == 11))
    event[stubble.to_numpy()] = 3
    for day in DIWALI:
        event[((date >= day - pd.Timedelta(days=2)) & (date <= day + pd.Timedelta(days=3))).to_numpy()] = 2
    event[date.between(*LOCKDOWN).to_numpy()] = 4
    return event


def clean_pollutants(df, cols):
    """Negative or absurd concentrations are sensor errors -> NULL (kept missing, not imputed)."""
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[(df[col] < 0) | (df[col] > 2000), col] = np.nan
    return df


def category_from_aqi(aqi):
    keys = np.select([aqi <= 50, aqi <= 100, aqi <= 200, aqi <= 300, aqi <= 400, aqi > 400],
                     [1, 2, 3, 4, 5, 6], default=0)
    return keys.astype(int)


def transform(raw, city_meta, stations, station_day, weather):
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")
    df["city"] = df["city"].str.strip()
    before = len(df)
    df = df.drop_duplicates(subset=["city", "date"], keep="first")
    print(f"Removed {before - len(df):,} duplicate city-day rows")
    df = clean_pollutants(df, POLLUTANTS + ["aqi"])
    print("Missing %:", (df[POLLUTANTS + ["aqi"]].isna().mean() * 100).round(1).to_dict())

    unknown = sorted(set(df.city) - set(city_meta.city))
    if unknown:
        raise ValueError(f"Add these cities to data/city_meta.csv: {unknown}")
    dim_city = city_meta.sort_values("city").reset_index(drop=True)
    dim_city.insert(0, "city_key", dim_city.index + 1)
    dim_date = build_dim_date(df["date"])
    dim_weather = build_dim_weather()

    # fact_weather: second source at the same grain, sharing dim_city and dim_date
    weather = weather.merge(dim_city[["city", "city_key"]], on="city")
    weather["date_key"] = date_key(weather["date"])
    weather = weather[weather.date_key.isin(dim_date.date_key)].reset_index(drop=True)
    weather["weather_key"] = assign_weather_key(weather, dim_weather)
    fact_weather = weather[["city_key", "date_key"] + WEATHER_MEASURES]

    # fact_air_quality
    df = df.merge(dim_city[["city", "city_key", "region"]], on="city")
    df["date_key"] = date_key(df["date"])
    df["event_key"] = assign_event(df["date"], df["region"])
    df["category_key"] = category_from_aqi(df["aqi"])
    df = df.merge(weather[["city_key", "date_key", "weather_key"]], how="left", on=["city_key", "date_key"])
    df["weather_key"] = df["weather_key"].fillna(0).astype(int)
    # Systematic sensor faults: a city whose typical CO is >3x the cross-city norm (Ahmedabad,
    # median ~16 mg/m3 vs ~1) has CO-driven AQI that is not credible. Keep the row, flag it,
    # and let AQI-based mining leave it out. AQI is capped at the scale maximum of 500.
    city_co = df.groupby("city")["co"].median()
    faulty = city_co[city_co > 3 * city_co.median()].index
    df["quality_flag"] = (df["city"].isin(faulty) & (df["co"] > 10)).astype(int)
    df["aqi"] = df["aqi"].clip(upper=500)
    print(f"Flagged {df.quality_flag.sum():,} suspect city-days:",
          df[df.quality_flag == 1].city.value_counts().head(3).to_dict())
    fact = df[["city_key", "date_key", "event_key", "category_key", "weather_key"]
              + POLLUTANTS + ["aqi", "quality_flag"]]

    # dim_station + fact_station_air_quality: drill-down below city
    used = stations[stations.StationId.isin(station_day.StationId)]
    dim_station = (used.rename(columns={"StationId": "station_id", "StationName": "station_name",
                                        "Status": "status", "City": "city"})
                   .merge(dim_city[["city", "city_key"]], on="city")
                   .sort_values("station_id").reset_index(drop=True))
    dim_station.insert(0, "station_key", dim_station.index + 1)
    dim_station = dim_station[["station_key", "station_id", "station_name", "city_key", "status"]]
    sd = station_day.rename(columns=STATION_COLUMNS).drop_duplicates(["station_id", "date"])
    sd = clean_pollutants(sd, STATION_MEASURES).merge(dim_station[["station_id", "station_key"]],
                                                      on="station_id")
    sd["date_key"] = date_key(sd["date"])
    sd["category_key"] = category_from_aqi(sd["aqi"])
    fact_station = sd[["station_key", "date_key", "category_key"] + STATION_MEASURES]

    # dict order = load order: dimensions first, then facts
    return {"dim_date": dim_date, "dim_city": dim_city, "dim_station": dim_station,
            "dim_aqi_category": AQI_CATEGORIES, "dim_event": EVENTS,
            "dim_weather_condition": dim_weather, "fact_air_quality": fact,
            "fact_station_air_quality": fact_station, "fact_weather": fact_weather}


def load(engine, tables):
    run_script(engine, ROOT / "sql" / "01_create_warehouse.sql")
    with engine.begin() as conn:
        for name, table in tables.items():
            table.to_sql(name, conn, if_exists="append", index=False, chunksize=5000)
            print(f"Loaded {name:<26} {len(table):>8,} rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=ROOT / "data" / "raw" / "city_day.csv")
    args = parser.parse_args()
    raw_dir = ROOT / "data" / "raw"
    raw = extract(args.input)
    city_meta = pd.read_csv(ROOT / "data" / "city_meta.csv")
    stations = pd.read_csv(raw_dir / "stations.csv", encoding="utf-8-sig")
    station_day = pd.read_csv(raw_dir / "station_day.csv")
    weather = pd.read_csv(raw_dir / "weather_daily.csv")
    load(get_engine(), transform(raw, city_meta, stations, station_day, weather))


if __name__ == "__main__":
    main()
