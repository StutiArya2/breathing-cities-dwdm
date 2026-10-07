"""Extract-Transform-Load: CPCB city_day.csv -> star schema.

Usage:
    python src/etl.py                         # reads data/raw/city_day.csv
    python src/etl.py --input path/to/city_day.csv
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


def extract(path):
    df = pd.read_csv(path)
    missing = set(SOURCE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    print(f"Extracted {len(df):,} rows x {df.shape[1]} columns "
          f"(md5 {hashlib.md5(open(path, 'rb').read()).hexdigest()[:10]})")
    return df.rename(columns=SOURCE_COLUMNS)


def build_dim_date(dates):
    d = pd.DataFrame({"full_date": pd.date_range(dates.min(), dates.max(), freq="D")})
    d["date_key"] = d.full_date.dt.strftime("%Y%m%d").astype(int)
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


def category_from_aqi(aqi):
    keys = np.select([aqi <= 50, aqi <= 100, aqi <= 200, aqi <= 300, aqi <= 400, aqi > 400],
                     [1, 2, 3, 4, 5, 6], default=0)
    return keys.astype(int)


def transform(raw, city_meta):
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")
    df["city"] = df["city"].str.strip()
    before = len(df)
    df = df.drop_duplicates(subset=["city", "date"], keep="first")
    print(f"Removed {before - len(df):,} duplicate city-day rows")

    # Negative or absurd concentrations are sensor errors -> NULL (kept as missing, not imputed).
    for col in POLLUTANTS + ["aqi"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[(df[col] < 0) | (df[col] > 2000), col] = np.nan
    print("Missing %:", (df[POLLUTANTS + ["aqi"]].isna().mean() * 100).round(1).to_dict())

    unknown = sorted(set(df.city) - set(city_meta.city))
    if unknown:
        raise ValueError(f"Add these cities to data/city_meta.csv: {unknown}")
    dim_city = city_meta.sort_values("city").reset_index(drop=True)
    dim_city.insert(0, "city_key", dim_city.index + 1)

    dim_date = build_dim_date(df["date"])
    df = df.merge(dim_city[["city", "city_key", "region"]], on="city")
    df["date_key"] = df["date"].dt.strftime("%Y%m%d").astype(int)
    df["event_key"] = assign_event(df["date"], df["region"])
    df["category_key"] = category_from_aqi(df["aqi"])

    fact = df[["city_key", "date_key", "event_key", "category_key"] + POLLUTANTS + ["aqi"]]
    return dim_date, dim_city, fact


def load(engine, dim_date, dim_city, fact):
    run_script(engine, ROOT / "sql" / "01_create_warehouse.sql")
    with engine.begin() as conn:  # dimensions first, then facts
        dim_date.to_sql("dim_date", conn, if_exists="append", index=False)
        dim_city.to_sql("dim_city", conn, if_exists="append", index=False)
        AQI_CATEGORIES.to_sql("dim_aqi_category", conn, if_exists="append", index=False)
        EVENTS.to_sql("dim_event", conn, if_exists="append", index=False)
        fact.to_sql("fact_air_quality", conn, if_exists="append", index=False, chunksize=5000)
    print(f"Loaded dim_date={len(dim_date):,} dim_city={len(dim_city)} fact={len(fact):,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=ROOT / "data" / "raw" / "city_day.csv")
    args = parser.parse_args()
    raw = extract(args.input)
    city_meta = pd.read_csv(ROOT / "data" / "city_meta.csv")
    load(get_engine(), *transform(raw, city_meta))


if __name__ == "__main__":
    main()
