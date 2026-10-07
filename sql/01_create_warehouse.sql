-- Breathing Cities: India Air Quality Data Warehouse
-- Fact constellation: three fact tables share the conformed dimensions dim_date and dim_city.
--   fact_air_quality          city x day     (CPCB city_day.csv)
--   fact_station_air_quality  station x day  (CPCB station_day.csv)   -> drill-down below city
--   fact_weather              city x day     (Open-Meteo ERA5 API)   -> second source
-- Portable SQL: runs on MySQL 8 and SQLite (default for quick runs).

DROP TABLE IF EXISTS fact_station_air_quality;
DROP TABLE IF EXISTS fact_weather;
DROP TABLE IF EXISTS fact_air_quality;
DROP TABLE IF EXISTS dim_station;
DROP TABLE IF EXISTS dim_weather_condition;
DROP TABLE IF EXISTS dim_date;
DROP TABLE IF EXISTS dim_city;
DROP TABLE IF EXISTS dim_aqi_category;
DROP TABLE IF EXISTS dim_event;

CREATE TABLE dim_date (
  date_key     INT PRIMARY KEY,          -- YYYYMMDD
  full_date    DATE NOT NULL UNIQUE,
  day          SMALLINT,
  day_name     VARCHAR(10),
  is_weekend   SMALLINT,
  week         SMALLINT,
  month        SMALLINT,
  month_name   VARCHAR(10),
  season       VARCHAR(15),              -- Winter / Pre-monsoon / Monsoon / Post-monsoon
  quarter      SMALLINT,
  year         SMALLINT
);

CREATE TABLE dim_city (
  city_key     INT PRIMARY KEY,
  city         VARCHAR(40) NOT NULL UNIQUE,
  state        VARCHAR(40),
  region       VARCHAR(15),              -- North / South / East / West / Central / North-East
  latitude     DECIMAL(7,4),
  longitude    DECIMAL(7,4),
  is_coastal   SMALLINT
);

CREATE TABLE dim_station (
  station_key  INT PRIMARY KEY,
  station_id   VARCHAR(10) NOT NULL UNIQUE,
  station_name VARCHAR(120),
  city_key     INT NOT NULL REFERENCES dim_city(city_key),   -- hierarchy Station -> City
  status       VARCHAR(10)
);

CREATE TABLE dim_aqi_category (
  category_key INT PRIMARY KEY,
  bucket       VARCHAR(15) NOT NULL,     -- CPCB National AQI bucket
  aqi_min      INT,
  aqi_max      INT,
  health_group VARCHAR(10)               -- Healthy / Unhealthy / Unknown
);

CREATE TABLE dim_event (
  event_key    INT PRIMARY KEY,
  event_name   VARCHAR(30) NOT NULL,     -- Normal / Diwali / Stubble burning / COVID lockdown
  event_type   VARCHAR(20),              -- Baseline / Festival / Agricultural / Policy
  description  VARCHAR(120)
);

CREATE TABLE dim_weather_condition (
  weather_key  INT PRIMARY KEY,
  wind_band    VARCHAR(10),              -- Calm / Light / Breezy        (daily mean wind)
  rain_band    VARCHAR(10),              -- Dry / Rain                   (>= 1 mm)
  temp_band    VARCHAR(10),              -- Cold / Mild / Hot
  mixing_band  VARCHAR(10),              -- Low / Medium / High          (boundary-layer height)
  label        VARCHAR(50)
);

CREATE TABLE fact_air_quality (
  city_key     INT NOT NULL REFERENCES dim_city(city_key),
  date_key     INT NOT NULL REFERENCES dim_date(date_key),
  event_key    INT NOT NULL REFERENCES dim_event(event_key),
  category_key INT NOT NULL REFERENCES dim_aqi_category(category_key),
  weather_key  INT NOT NULL REFERENCES dim_weather_condition(weather_key),
  pm25         DECIMAL(8,2),             -- pollutant measures in ug/m3 (CO in mg/m3)
  pm10         DECIMAL(8,2),
  no           DECIMAL(8,2),
  no2          DECIMAL(8,2),
  nox          DECIMAL(8,2),
  nh3          DECIMAL(8,2),
  co           DECIMAL(8,2),
  so2          DECIMAL(8,2),
  o3           DECIMAL(8,2),
  benzene      DECIMAL(8,2),
  toluene      DECIMAL(8,2),
  xylene       DECIMAL(8,2),
  aqi          DECIMAL(8,2),
  quality_flag SMALLINT NOT NULL,        -- 1 = suspect: CO sensor fault (see ETL)
  PRIMARY KEY (city_key, date_key)
);

CREATE TABLE fact_station_air_quality (
  station_key  INT NOT NULL REFERENCES dim_station(station_key),
  date_key     INT NOT NULL REFERENCES dim_date(date_key),
  category_key INT NOT NULL REFERENCES dim_aqi_category(category_key),
  pm25         DECIMAL(8,2),
  pm10         DECIMAL(8,2),
  no2          DECIMAL(8,2),
  co           DECIMAL(8,2),
  so2          DECIMAL(8,2),
  o3           DECIMAL(8,2),
  aqi          DECIMAL(8,2),
  PRIMARY KEY (station_key, date_key)
);

CREATE TABLE fact_weather (
  city_key      INT NOT NULL REFERENCES dim_city(city_key),
  date_key      INT NOT NULL REFERENCES dim_date(date_key),
  temp_mean     DECIMAL(5,2),            -- deg C
  temp_max      DECIMAL(5,2),
  temp_min      DECIMAL(5,2),
  humidity      DECIMAL(5,2),            -- %
  precipitation DECIMAL(6,2),            -- mm
  wind_mean     DECIMAL(5,2),            -- km/h
  wind_max      DECIMAL(5,2),
  wind_dir      DECIMAL(5,1),            -- degrees
  radiation     DECIMAL(6,2),            -- MJ/m2
  mixing_height DECIMAL(7,1),            -- m, daily mean boundary-layer height
  PRIMARY KEY (city_key, date_key)
);

CREATE INDEX ix_fact_date    ON fact_air_quality (date_key);
CREATE INDEX ix_fact_event   ON fact_air_quality (event_key);
CREATE INDEX ix_fact_weather ON fact_air_quality (weather_key);
CREATE INDEX ix_station_date ON fact_station_air_quality (date_key);
