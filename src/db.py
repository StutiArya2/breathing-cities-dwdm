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


def read_cube(engine):
    """Denormalised view of the star used by all mining scripts."""
    query = """
        SELECT c.city, c.region, c.is_coastal, d.full_date, d.year, d.month, d.season,
               d.is_weekend, e.event_name, a.bucket,
               f.pm25, f.pm10, f.no, f.no2, f.nox, f.nh3, f.co, f.so2, f.o3,
               f.benzene, f.toluene, f.xylene, f.aqi
        FROM fact_air_quality f
        JOIN dim_city c         ON f.city_key = c.city_key
        JOIN dim_date d         ON f.date_key = d.date_key
        JOIN dim_event e        ON f.event_key = e.event_key
        JOIN dim_aqi_category a ON f.category_key = a.category_key
    """
    df = pd.read_sql(query, engine, parse_dates=["full_date"])
    num = POLLUTANTS + ["aqi"]
    df[num] = df[num].astype(float)
    return df.sort_values(["city", "full_date"]).reset_index(drop=True)
