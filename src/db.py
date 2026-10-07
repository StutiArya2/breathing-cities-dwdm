"""Shared database helpers.

Set DATABASE_URL to use MySQL, e.g.
    mysql+pymysql://user:password@localhost/air_quality_dw
Otherwise a local SQLite file is used.
"""
import os
import re
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = f"sqlite:///{(ROOT / 'warehouse' / 'air_quality_dw.db').as_posix()}"
OUTPUTS = ROOT / "outputs"
POLLUTANTS = ["pm25", "pm10", "no", "no2", "nox", "nh3", "co", "so2", "o3",
              "benzene", "toluene", "xylene"]


def get_engine():
    return create_engine(os.environ.get("DATABASE_URL", DEFAULT_URL))


def run_script(engine, path):
    """Execute a ;-separated SQL file statement by statement."""
    sql = re.sub(r"--[^\n]*", "", Path(path).read_text(encoding="utf-8"))
    with engine.begin() as conn:
        for stmt in filter(None, (s.strip() for s in sql.split(";"))):
            conn.execute(text(stmt))


def load_named_queries(path):
    """Return {name: sql} for every '-- name: x' block in a SQL file."""
    blocks = re.split(r"^-- name:\s*(\w+)\s*$", Path(path).read_text(encoding="utf-8"), flags=re.M)
    return {name: body.strip().rstrip(";") for name, body in zip(blocks[1::2], blocks[2::2])}


WEATHER = ["temp_mean", "humidity", "precipitation", "wind_mean", "wind_max", "radiation",
           "mixing_height"]


def read_cube(engine, include_suspect=False):
    """Denormalised drill-across view of the constellation used by all mining scripts:
    fact_air_quality joined to its dimensions and to fact_weather on (city, date)."""
    query = f"""
        SELECT c.city, c.region, c.is_coastal, d.full_date, d.year, d.month, d.season,
               d.is_weekend, e.event_name, a.bucket,
               wc.wind_band, wc.rain_band, wc.temp_band, wc.mixing_band, f.quality_flag,
               {", ".join("f." + p for p in POLLUTANTS)}, f.aqi,
               {", ".join("w." + m for m in WEATHER)}
        FROM fact_air_quality f
        JOIN dim_city c               ON f.city_key = c.city_key
        JOIN dim_date d               ON f.date_key = d.date_key
        JOIN dim_event e              ON f.event_key = e.event_key
        JOIN dim_aqi_category a       ON f.category_key = a.category_key
        JOIN dim_weather_condition wc ON f.weather_key = wc.weather_key
        LEFT JOIN fact_weather w      ON f.city_key = w.city_key AND f.date_key = w.date_key
    """
    df = pd.read_sql(query, engine, parse_dates=["full_date"])
    num = POLLUTANTS + ["aqi"] + WEATHER
    df[num] = df[num].astype(float)
    if not include_suspect:
        # suspect rows keep their place in the time series but lose their readings
        df.loc[df.quality_flag == 1, POLLUTANTS + ["aqi"]] = float("nan")
        df.loc[df.quality_flag == 1, "bucket"] = "Unknown"
    return df.sort_values(["city", "full_date"]).reset_index(drop=True)
