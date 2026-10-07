# Breathing Cities: India Air Quality Warehouse & Mining

Data Warehousing & Data Mining (CSI3010) lab project.

A star-schema data warehouse over six years of CPCB daily air-quality readings for 26 Indian
cities, with OLAP analysis, four data-mining tasks and a Streamlit dashboard. The creative hook:
the March 2020 COVID lockdown is a natural experiment. We forecast what the air *would* have
been without it and measure the gap.

| Task | What we do | Script |
|---|---|---|
| Data warehouse | Star schema, grain = city x day; dims Date, City, AQI Category, Event | `sql/01_create_warehouse.sql`, `src/etl.py` |
| OLAP | Roll-up, drill-down, slice, dice, pivot (+ event comparison) | `sql/02_olap_queries.sql`, `src/olap.py` |
| Classification | Will tomorrow be unhealthy (AQI > 200)? Decision Tree vs Random Forest | `src/classify.py` |
| Clustering | K-Means "pollution personalities" of cities, silhouette-selected k | `src/cluster.py` |
| Association rules | FP-Growth on daily pollutant/season/event baskets, per region | `src/associate.py` |
| Prediction | Holt-Winters monthly AQI forecast + lockdown what-if | `src/forecast.py` |
| Dashboard | Map, OLAP explorer, clusters, rules, what-if, risk | `dashboard/app.py` |

## Dataset

**Air Quality Data in India (2015-2020)**, compiled by Vopani (Rohan Rao) on Kaggle from the
Central Pollution Control Board (CPCB): <https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india>

Download it and place `city_day.csv` in `data/raw/`. The raw file is not committed.
`data/city_meta.csv` (state, region, coordinates, coastal flag) is our own lookup table for the
City dimension.

## Run

```bash
pip install -r requirements.txt
python src/etl.py            # builds warehouse/air_quality_dw.db (SQLite)
python src/olap.py           # prints and saves the OLAP results to outputs/
python src/classify.py
python src/cluster.py
python src/associate.py
python src/forecast.py
streamlit run dashboard/app.py
```

To use MySQL instead of SQLite:

```bash
export DATABASE_URL="mysql+pymysql://user:password@localhost/air_quality_dw"
```

### No data yet?

`python scripts/make_synthetic_data.py` writes a random file with the same columns to
`data/sample/`. Load it with `python src/etl.py --input data/sample/city_day_synthetic.csv`
to check that everything runs. Its numbers are made up and are not results.

## Schema

```
                 dim_date                         dim_city
   (date_key, full_date, day, week, month,   (city_key, city, state, region,
    season, quarter, year, is_weekend)        latitude, longitude, is_coastal)
                      \                         /
                       fact_air_quality  (PK city_key + date_key)
                       pm25, pm10, no, no2, nox, nh3, co, so2, o3,
                       benzene, toluene, xylene, aqi
                      /                         \
        dim_aqi_category                         dim_event
   (category_key, bucket, aqi_min,          (event_key, event_name,
    aqi_max, health_group)                   event_type, description)
```

Hierarchies: Day -> Month -> Season -> Year; City -> State -> Region -> India;
AQI value -> Bucket -> Healthy/Unhealthy.
Pollutant measures are **non-additive** (concentrations), so roll-ups use `AVG`; day counts use `COUNT`.
