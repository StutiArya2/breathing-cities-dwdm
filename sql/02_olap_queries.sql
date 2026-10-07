-- Sample OLAP operations on the air-quality star schema.
-- Each query is tagged "-- name: <op>" so src/olap.py can run them one by one.
-- Pollutant concentrations are non-additive: aggregate them with AVG, never SUM.

-- name: rollup
-- Roll-up: City -> Region, yearly average PM2.5
SELECT c.region, d.year, ROUND(AVG(f.pm25), 1) AS avg_pm25
FROM fact_air_quality f
JOIN dim_city c ON f.city_key = c.city_key
JOIN dim_date d ON f.date_key = d.date_key
GROUP BY c.region, d.year
ORDER BY c.region, d.year;

-- name: drilldown
-- Drill-down: Delhi, Year 2019 -> Month
SELECT d.month, d.month_name, ROUND(AVG(f.aqi), 0) AS avg_aqi
FROM fact_air_quality f
JOIN dim_city c ON f.city_key = c.city_key
JOIN dim_date d ON f.date_key = d.date_key
WHERE c.city = 'Delhi' AND d.year = 2019
GROUP BY d.month, d.month_name
ORDER BY d.month;

-- name: slice
-- Slice: Season = Winter, average AQI per city
SELECT c.city, ROUND(AVG(f.aqi), 0) AS winter_aqi
FROM fact_air_quality f
JOIN dim_city c ON f.city_key = c.city_key
JOIN dim_date d ON f.date_key = d.date_key
WHERE d.season = 'Winter'
GROUP BY c.city
ORDER BY winter_aqi DESC;

-- name: dice
-- Dice: North region x (Mar-May) x (2019, 2020) -> lockdown comparison
SELECT c.city, d.year, ROUND(AVG(f.pm25), 1) AS avg_pm25, ROUND(AVG(f.no2), 1) AS avg_no2
FROM fact_air_quality f
JOIN dim_city c ON f.city_key = c.city_key
JOIN dim_date d ON f.date_key = d.date_key
WHERE c.region = 'North'
  AND d.year IN (2019, 2020)
  AND d.month BETWEEN 3 AND 5
GROUP BY c.city, d.year
ORDER BY c.city, d.year;

-- name: pivot
-- Pivot: cities as rows, AQI buckets as columns (number of days)
SELECT c.city,
  SUM(CASE WHEN a.bucket = 'Good'         THEN 1 ELSE 0 END) AS good,
  SUM(CASE WHEN a.bucket = 'Satisfactory' THEN 1 ELSE 0 END) AS satisfactory,
  SUM(CASE WHEN a.bucket = 'Moderate'     THEN 1 ELSE 0 END) AS moderate,
  SUM(CASE WHEN a.bucket = 'Poor'         THEN 1 ELSE 0 END) AS poor,
  SUM(CASE WHEN a.bucket = 'Very Poor'    THEN 1 ELSE 0 END) AS very_poor,
  SUM(CASE WHEN a.bucket = 'Severe'       THEN 1 ELSE 0 END) AS severe
FROM fact_air_quality f
JOIN dim_city c         ON f.city_key = c.city_key
JOIN dim_aqi_category a ON f.category_key = a.category_key
GROUP BY c.city
ORDER BY severe DESC, very_poor DESC;

-- name: events
-- Bonus: event dimension, average AQI during Diwali / stubble burning / lockdown vs normal
SELECT e.event_name, COUNT(*) AS city_days, ROUND(AVG(f.aqi), 0) AS avg_aqi
FROM fact_air_quality f
JOIN dim_event e ON f.event_key = e.event_key
GROUP BY e.event_name
ORDER BY avg_aqi DESC;
