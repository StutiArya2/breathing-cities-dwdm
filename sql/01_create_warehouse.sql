-- Breathing Cities: India Air Quality Data Warehouse
-- Star schema, grain = one row per city per day.
-- Portable SQL: runs on MySQL 8 and SQLite (default for quick runs).

DROP TABLE IF EXISTS fact_air_quality;
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

CREATE TABLE fact_air_quality (
  city_key     INT NOT NULL REFERENCES dim_city(city_key),
  date_key     INT NOT NULL REFERENCES dim_date(date_key),
  event_key    INT NOT NULL REFERENCES dim_event(event_key),
  category_key INT NOT NULL REFERENCES dim_aqi_category(category_key),
  pm25         DECIMAL(8,2),             -- all pollutant measures in ug/m3 (CO in mg/m3)
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
  PRIMARY KEY (city_key, date_key)
);

CREATE INDEX ix_fact_date  ON fact_air_quality (date_key);
CREATE INDEX ix_fact_event ON fact_air_quality (event_key);
