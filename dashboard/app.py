"""Breathing Cities dashboard.  Run:  streamlit run dashboard/app.py"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from db import OUTPUTS, ROOT, get_engine, load_named_queries, read_cube  # noqa: E402

BUCKET_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
BUCKET_COLORS = {"Good": "#2e9d4a", "Satisfactory": "#9ccc3c", "Moderate": "#f2c12e",
                 "Poor": "#f28e2b", "Very Poor": "#d6402b", "Severe": "#7b1e1e"}

st.set_page_config(page_title="Breathing Cities", layout="wide")


@st.cache_data
def load_cube():
    return read_cube(get_engine())


def output(name):
    path = OUTPUTS / name
    return pd.read_csv(path) if path.exists() else None


cube = load_cube()
st.title("Breathing Cities: India Air Quality 2015-2020")

with st.sidebar:
    cities = st.multiselect("Cities", sorted(cube.city.unique()), default=["Delhi", "Bengaluru", "Mumbai"])
    years = st.slider("Years", int(cube.year.min()), int(cube.year.max()),
                      (int(cube.year.min()), int(cube.year.max())))
view = cube[cube.city.isin(cities or cube.city.unique()) & cube.year.between(*years)]

overview, olap, clusters, rules, forecast, risk = st.tabs(
    ["Overview", "OLAP explorer", "City clusters", "Association rules", "Lockdown what-if", "Tomorrow's risk"])

with overview:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("City-days", f"{len(view):,}")
    k2.metric("Average AQI", f"{view.aqi.mean():.0f}")
    k3.metric("Unhealthy days (AQI>200)", f"{(view.aqi > 200).mean():.0%}")
    k4.metric("Worst city", view.groupby("city").aqi.mean().idxmax() if len(view) else "-")
    city_avg = cube[cube.year.between(*years)].groupby(["city", "region"], as_index=False).aqi.mean()
    meta = pd.read_csv(ROOT / "data" / "city_meta.csv")
    st.plotly_chart(px.scatter_geo(city_avg.merge(meta, on=["city", "region"]), lat="latitude",
                                   lon="longitude", size="aqi", color="aqi", hover_name="city",
                                   color_continuous_scale="OrRd", scope="asia", height=520)
                    .update_geos(fitbounds="locations"), width="stretch")
    monthly = view.set_index("full_date").groupby("city").aqi.resample("MS").mean().reset_index()
    st.plotly_chart(px.line(monthly, x="full_date", y="aqi", color="city", title="Monthly AQI"),
                    width="stretch")

with olap:
    queries = load_named_queries(ROOT / "sql" / "02_olap_queries.sql")
    op = st.radio("Operation", list(queries), horizontal=True)
    st.code(queries[op], language="sql")
    st.dataframe(pd.read_sql(queries[op], get_engine()), width="stretch")
    mix = view[view.bucket != "Unknown"].groupby(["city", "bucket"]).size().reset_index(name="days")
    st.plotly_chart(px.bar(mix, x="city", y="days", color="bucket", category_orders={"bucket": BUCKET_ORDER},
                           color_discrete_map=BUCKET_COLORS, title="Days per AQI bucket"),
                    width="stretch")

with clusters:
    cl = output("city_clusters.csv")
    if cl is None:
        st.info("Run `python src/cluster.py` first.")
    else:
        cl["cluster"] = cl.cluster.astype(str)
        st.plotly_chart(px.scatter(cl, x="pc1", y="pc2", color="cluster", text="city",
                                   title="Cities in PCA space, coloured by K-Means cluster"),
                        width="stretch")
        st.dataframe(output("cluster_summary.csv"), width="stretch")

with rules:
    ar = output("association_rules.csv")
    if ar is None:
        st.info("Run `python src/associate.py` first.")
    else:
        region = st.selectbox("Region (segment)", sorted(ar.region.unique()))
        target = st.selectbox("Consequent", sorted(ar.consequents.unique()))
        st.dataframe(ar[(ar.region == region) & (ar.consequents == target)].head(25),
                     width="stretch")

with forecast:
    wi = output("lockdown_whatif.csv")
    if wi is None:
        st.info("Run `python src/forecast.py` first.")
    else:
        st.plotly_chart(px.bar(wi, x="lockdown_effect_pct", y="city", orientation="h",
                               title="Actual vs expected AQI, Mar-Jun 2020 (% change)"),
                        width="stretch")
        st.dataframe(output("forecast_backtest.csv"), width="stretch")

with risk:
    fi = output("feature_importance.csv")
    report = OUTPUTS / "classification_report.txt"
    if fi is None or not report.exists():
        st.info("Run `python src/classify.py` first.")
    else:
        st.plotly_chart(px.bar(fi.head(10), x="importance", y=fi.columns[0], orientation="h",
                               title="What predicts an unhealthy tomorrow? (Random Forest)"),
                        width="stretch")
        st.text(report.read_text(encoding="utf-8").split("== Decision tree rules ==")[0])
