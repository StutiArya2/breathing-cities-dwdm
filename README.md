# Breathing Cities: Mining India's Air Quality

**Data Warehousing & Data Mining (CSI3010) lab project**

A fact-constellation data warehouse that combines six years of CPCB air-quality readings for 26
Indian cities with daily weather from a second source, followed by OLAP, four data-mining tasks,
two natural experiments and a Streamlit dashboard.

The creative angle: the **2020 COVID lockdown** and **Diwali firecracker rules** are natural
experiments. A weather model predicts what the air *should* have been, so the effect of each event
can be measured with weather luck removed, and a placebo year checks the method.

## Headline results

All numbers come from `python src/run_all.py` on the real data. Outputs land in `outputs/`.

| Question | Result |
|---|---|
| Did the lockdown clean the air? | Weather-adjusted median change across cities: **NO2 −29%, PM2.5 −28%**. Placebo year (2019, no lockdown): −1% and −4%. Coal and mining towns (Brajrajnagar, Jorapokhar, Talcher) got *dirtier*. |
| Do Diwali firecracker rules work? | Delhi's raw PM2.5 on Diwali fell from 395 (2016) to 301 (2017, sales ban), but 2017 had favourable weather. Weather-adjusted excess *rose* to **+126%**, then fell to about +45% under the 2018–19 green-cracker rules. |
| What conditions bring bad air? | North: stubble burning + low mixing height → **83%** of days unhealthy (base rate 39%). East: low mixing + dry + cold → 81% (base 26%). |
| Can we predict tomorrow's air? | Random Forest F1 **0.83** vs 0.78 for "tomorrow = today", and right on 35% of the days the status changes (persistence: 0%). Weather adds nothing for next-day prediction. |
| Can we forecast monthly AQI? | Weather-adjusted seasonal naive: **14.2%** MAPE vs 16.8% naive, 23.9% Holt-Winters, 24.2% SARIMA (7 cities with ≥ 3 years). Wins in 12 of 14 cities. |
| How do cities group? | K-Means (k = 3): **smog belt** (Delhi, Gurugram, Lucknow, Patna; 50% of days unhealthy), **moderate urban** (16 cities), **low-traffic, SO2-heavy** (6 industrial and hill/coastal towns). |

## Data

| Source | Content | How to get it |
|---|---|---|
| [Air Quality Data in India (2015–2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india), CPCB data compiled by Vopani on Kaggle | `city_day.csv` (29,531 rows), `station_day.csv` (108,035), `stations.csv` | Download the zip and extract the CSVs into `data/raw/` |
| [Open-Meteo historical weather API](https://open-meteo.com/en/docs/historical-weather-api) (ERA5) | Daily temperature, humidity, rain, wind, radiation and boundary-layer (mixing) height per city | `python src/fetch_weather.py` (no key; `run_all.py` calls it automatically) |
| `data/city_meta.csv` (ours) | State, region, coordinates, coastal flag for the 26 cities | In the repo |

Raw data is not committed (`data/raw/` is gitignored).

### Data quality findings

- Missing values: Xylene 61%, PM10 38%, NH3 35%, PM2.5 16%, AQI 16%. Weather: 0%.
- Only 9 of 26 cities start in 2015; 8 start in 2019–20.
- No duplicate city-days, no negative readings.
- **Ahmedabad's CO sensor is faulty**: median 16 mg/m³ against about 1 everywhere else, pushing its AQI
  above the 500 ceiling on 412 days. A general rule (city CO above 3× the cross-city norm, daily CO > 10)
  flags 915 Ahmedabad days. They are kept in the warehouse with `quality_flag = 1` and excluded from AQI-based
  queries and mining.
- AQI is capped at 500, the top of the CPCB scale (543 rows above it).

## Warehouse design: fact constellation

```
                 DIM_Event        DIM_Date (conformed)      DIM_WeatherCondition
                        \        /        |        \        /
     FACT_Weather ------ FACT_AirQuality  |         FACT_StationAQ
     (city x day)        (city x day)     |         (station x day)
                        /        \        |        /        \
              DIM_City (conformed)      DIM_AQI_Category     DIM_Station -> DIM_City
```

| Table | Grain / content |
|---|---|
| `fact_air_quality` | city × day: 12 pollutants, AQI, `quality_flag`; FKs to date, city, event, AQI category, weather condition |
| `fact_station_air_quality` | station × day: PM2.5, PM10, NO2, CO, SO2, O3, AQI |
| `fact_weather` | city × day: temperature, humidity, precipitation, wind, radiation, mixing height |
| `dim_date` | day, week, month, IMD season, quarter, year |
| `dim_city`, `dim_station` | Station → City → State → Region |
| `dim_aqi_category` | CPCB bucket, AQI range, Healthy / Unhealthy |
| `dim_event` | Normal, Diwali (day −2 to +3), Stubble burning (15 Oct–30 Nov, North only), COVID lockdown |
| `dim_weather_condition` | wind band × rain band × temperature band × mixing band (54 combinations + Unknown) |

Pollutant concentrations are **non-additive**: roll-ups use `AVG`/`MAX`, day counts use `COUNT`.
The SQL is portable between SQLite (default) and MySQL 8.

### OLAP queries (`sql/02_olap_queries.sql`)

Roll-up, drill-down, slice, dice, pivot, an event comparison, plus three that need the constellation:
**drill-down to stations**, **dice on the weather dimension**, and **drill-across** two fact tables
through the conformed dimensions.

## Mining and analysis

| Script | What it does |
|---|---|
| `src/classify.py` | Next-day unhealthy-air classifier: persistence baseline, Decision Tree, Random Forest, Random Forest + weather. Chronological split (train ≤ 2018, test 2019–20). SHAP explanations. |
| `src/cluster.py` | K-Means on city profiles: pollutant mix, winter/monsoon ratio, unhealthy-day share, PM2.5 vs mixing-height sensitivity. k chosen by silhouette. |
| `src/associate.py` | FP-Growth per region. *Context rules* (weather, season, event → Unhealthy; no pollutants on the left, since AQI is computed from them) and *co-pollution rules*. |
| `src/forecast.py` | Monthly AQI forecast for 2019: seasonal naive, Holt-Winters, SARIMA and weather-adjusted seasonal naive. |
| `src/deweather.py` | Shared gradient-boosting model: daily pollution from weather + calendar only. |
| `src/lockdown.py` | Weather-normalised lockdown counterfactual for NO2, PM2.5 and AQI, plus a 2019 placebo. |
| `src/diwali.py` | Diwali excess PM2.5 vs weather-normalised expectation, 2015–2019. |

## Run it

Create a virtual environment once (Windows):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks `activate`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use
Command Prompt. In VS Code, pick the interpreter with *Python: Select Interpreter* → `.venv`.

Then, with the venv active:

```bash
python src/run_all.py            # weather fetch (first time), ETL, OLAP, all mining -> outputs/
streamlit run dashboard/app.py   # 10-tab dashboard
```

Individual steps can be run on their own, e.g. `python src/olap.py pivot`.
To use MySQL instead of SQLite, set
`DATABASE_URL=mysql+pymysql://user:password@localhost/air_quality_dw` before running.

## Dashboard

Tabs: Overview (KPIs, India map), OLAP explorer (all queries live), Stations (heatmap),
Weather, Tomorrow's risk (scores + SHAP), City clusters, Association rules, Forecast,
Lockdown what-if, Diwali.

To put it online, deploy `dashboard/app.py` on [Streamlit Community Cloud](https://streamlit.io/cloud)
from this repository. The app needs the warehouse file and `outputs/`, which are gitignored, so either
commit a built copy (check the dataset licence first) or add a build step.

## Repository layout

```
sql/        01_create_warehouse.sql (9 tables), 02_olap_queries.sql (9 queries)
src/        etl.py, fetch_weather.py, olap.py, classify.py, cluster.py, associate.py,
            forecast.py, deweather.py, lockdown.py, diwali.py, run_all.py, db.py
dashboard/  app.py
data/       city_meta.csv; raw/ (not committed)
docs/       Breathing_Cities_DWDM.pptx (project presentation)
```

## Limitations

- Next-day weather is the observed weather, standing in for a forecast; monthly forecasts use observed
  monthly weather the same way.
- The counterfactual models capture weather and season, not every other change (traffic growth, new
  sensors, policy). The placebo test checks for bias but cannot rule out all of it.
- The Diwali comparison across years is descriptive, not causal proof.
- Data ends on 1 July 2020.
