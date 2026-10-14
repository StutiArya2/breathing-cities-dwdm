"""Breathing Cities dashboard.  Run:  streamlit run dashboard/app.py"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from db import DEFAULT_URL, OUTPUTS, ROOT, get_engine, load_named_queries, read_cube  # noqa: E402

BUCKET_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
BUCKET_COLORS = {"Good": "#2e9d4a", "Satisfactory": "#9ccc3c", "Moderate": "#f2c12e",
                 "Poor": "#f28e2b", "Very Poor": "#d6402b", "Severe": "#7b1e1e"}
WIDE = {"width": "stretch"}

st.set_page_config(page_title="Breathing Cities", layout="wide")

if DEFAULT_URL.startswith("sqlite") and not (ROOT / "warehouse" / "air_quality_dw.db").exists():
    st.error("No warehouse found. Run `python src/run_all.py` first (see README).")
    st.stop()


@st.cache_data
def load_cube():
    return read_cube(get_engine())


@st.cache_data
def query(sql):
    return pd.read_sql(sql, get_engine())


def output(name):
    path = OUTPUTS / name
    if not path.exists():
        st.info(f"`outputs/{name}` not found. Run `python src/run_all.py` first.")
        return None
    return pd.read_csv(path)


cube = load_cube()
st.title("Breathing Cities: India Air Quality 2015-2020")
st.caption("CPCB daily readings for 26 cities + Open-Meteo weather, in a fact-constellation warehouse.")

with st.sidebar:
    cities = st.multiselect("Cities", sorted(cube.city.unique()), default=["Delhi", "Bengaluru", "Mumbai"])
    years = st.slider("Years", int(cube.year.min()), int(cube.year.max()),
                      (int(cube.year.min()), int(cube.year.max())))
view = cube[cube.city.isin(cities or cube.city.unique()) & cube.year.between(*years)]

tabs = st.tabs(["Overview", "OLAP explorer", "Stations", "Weather", "Tomorrow's risk", "City clusters",
                "Association rules", "Forecast", "Lockdown what-if", "Diwali"])

with tabs[0]:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("City-days", f"{len(view):,}")
    k2.metric("Average AQI", f"{view.aqi.mean():.0f}")
    k3.metric("Unhealthy days (AQI>200)", f"{(view.aqi > 200).mean():.0%}")
    k4.metric("Worst city", view.groupby("city").aqi.mean().idxmax() if view.aqi.notna().any() else "-")
    city_avg = cube[cube.year.between(*years)].groupby(["city", "region"], as_index=False).aqi.mean()
    meta = pd.read_csv(ROOT / "data" / "city_meta.csv")
    st.plotly_chart(px.scatter_geo(city_avg.merge(meta, on=["city", "region"]), lat="latitude",
                                   lon="longitude", size="aqi", color="aqi", hover_name="city",
                                   color_continuous_scale="OrRd", scope="asia", height=520)
                    .update_geos(fitbounds="locations"), **WIDE)
    monthly = view.set_index("full_date").groupby("city").aqi.resample("MS").mean().reset_index()
    st.plotly_chart(px.line(monthly, x="full_date", y="aqi", color="city", title="Monthly AQI"), **WIDE)

with tabs[1]:
    queries = load_named_queries(ROOT / "sql" / "02_olap_queries.sql")
    op = st.radio("Operation", list(queries), horizontal=True)
    st.code(queries[op], language="sql")
    st.dataframe(query(queries[op]), **WIDE)
    mix = view[view.bucket != "Unknown"].groupby(["city", "bucket"]).size().reset_index(name="days")
    st.plotly_chart(px.bar(mix, x="city", y="days", color="bucket", category_orders={"bucket": BUCKET_ORDER},
                           color_discrete_map=BUCKET_COLORS, title="Days per AQI bucket"), **WIDE)

with tabs[2]:
    st.subheader("Drill-down: city -> monitoring station")
    city = st.selectbox("City", sorted(cube.city.unique()), index=sorted(cube.city.unique()).index("Delhi"))
    stations = query(f"""
        SELECT s.station_name, d.year, ROUND(AVG(f.pm25), 1) AS avg_pm25, COUNT(*) AS days
        FROM fact_station_air_quality f
        JOIN dim_station s ON f.station_key = s.station_key
        JOIN dim_city c ON s.city_key = c.city_key
        JOIN dim_date d ON f.date_key = d.date_key
        WHERE c.city = '{city.replace("'", "''")}'
        GROUP BY s.station_name, d.year""")
    if stations.empty:
        st.info("No station-level data for this city.")
    else:
        pivot = stations.pivot(index="station_name", columns="year", values="avg_pm25")
        st.plotly_chart(px.imshow(pivot, color_continuous_scale="OrRd", aspect="auto",
                                  title=f"Average PM2.5 by station and year, {city}"), **WIDE)

with tabs[3]:
    st.subheader("Weather source: what conditions trap pollution?")
    w = view.dropna(subset=["pm25", "mixing_height"])
    st.plotly_chart(px.scatter(w.sample(min(len(w), 4000), random_state=0), x="mixing_height", y="pm25",
                               color="season", opacity=0.4, log_x=True, title="Daily PM2.5 vs mixing height"),
                    **WIDE)
    st.dataframe(query(queries["weather_dice"]), **WIDE)

with tabs[4]:
    sc = output("classification_scores.csv")
    shap_imp = output("shap_importance.csv")
    if sc is not None and shap_imp is not None:
        st.subheader("Will tomorrow be unhealthy (AQI > 200)? Test years 2019-2020")
        st.dataframe(sc, **WIDE)
        shap_imp.columns = ["feature", "mean_abs_shap"]
        st.plotly_chart(px.bar(shap_imp.head(12).iloc[::-1], x="mean_abs_shap", y="feature", orientation="h",
                               title="SHAP: what drives the prediction (Random Forest + weather)"), **WIDE)

with tabs[5]:
    cl = output("city_clusters.csv")
    if cl is not None:
        fig = px.scatter(cl, x="pc1", y="pc2", color="cluster_name", text="city", height=600,
                         color_discrete_map={"Smog belt": "#e8453c", "Moderate urban": "#4a90d9",
                                             "Low-traffic, SO2-heavy": "#f5a623"},
                         labels={"pc1": "Principal component 1", "pc2": "Principal component 2",
                                 "cluster_name": "Cluster"},
                         title="Cities in PCA space, coloured by K-Means cluster")
        fig.update_traces(marker=dict(size=16, line=dict(width=1.5, color="white")),
                          textposition="top center", textfont=dict(size=12))
        fig.update_layout(legend=dict(font=dict(size=14), itemsizing="constant"))
        st.plotly_chart(fig, **WIDE)
        st.dataframe(output("cluster_summary.csv"), **WIDE)

with tabs[6]:
    ctx, cp = output("rules_context.csv"), output("rules_copollution.csv")
    if ctx is not None and cp is not None:
        region = st.selectbox("Region (segment)", sorted(ctx.region.unique()))
        st.markdown("**Context rules**: which weather / season / event conditions lead to unhealthy air")
        st.dataframe(ctx[ctx.region == region].head(20), **WIDE)
        st.markdown("**Co-pollution rules**: which pollutants exceed their standards together")
        st.dataframe(cp[cp.region == region].head(20), **WIDE)

with tabs[7]:
    bt, fc = output("forecast_backtest.csv"), output("forecast_2019.csv")
    if bt is not None and fc is not None:
        st.subheader("Monthly AQI forecast for 2019 (trained to Dec 2018), MAPE %")
        st.dataframe(bt, **WIDE)
        c = st.selectbox("City", sorted(fc.city.unique()), key="fc_city")
        f = fc[fc.city == c].melt(id_vars=["month", "city"], var_name="series", value_name="aqi")
        st.plotly_chart(px.line(f, x="month", y="aqi", color="series", markers=True), **WIDE)

with tabs[8]:
    eff, summ, daily = output("lockdown_effect.csv"), output("lockdown_summary.csv"), \
        output("lockdown_delhi_daily.csv")
    if eff is not None and summ is not None and daily is not None:
        st.subheader("Actual vs weather-normalised expectation, 25 Mar - 31 May 2020")
        st.dataframe(summ, **WIDE)
        p = st.radio("Pollutant", ["PM2.5", "NO2", "AQI"], horizontal=True)
        e = eff[eff.pollutant == p].sort_values("change_pct")
        st.plotly_chart(px.bar(e, x="change_pct", y="city", orientation="h", height=650,
                               title=f"{p}: % change vs expected"), **WIDE)
        fig = go.Figure([go.Scatter(x=daily.full_date, y=daily.pm25, name="Actual PM2.5"),
                         go.Scatter(x=daily.full_date, y=daily.expected_pm25, name="Expected (no lockdown)")])
        fig.add_vrect(x0="2020-03-25", x1="2020-05-31", fillcolor="grey", opacity=0.15, line_width=0)
        st.plotly_chart(fig.update_layout(title="Delhi, 2020"), **WIDE)

with tabs[9]:
    dw = output("diwali_excess.csv")
    if dw is not None:
        st.subheader("Extra PM2.5 on Diwali night + next day, vs weather-normalised expectation")
        st.plotly_chart(px.box(dw, x="year", y="excess_pct", points="all", hover_name="city",
                               title="% excess PM2.5 across cities"), **WIDE)
        st.dataframe(dw[dw.city == "Delhi"], **WIDE)
