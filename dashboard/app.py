"""Breathing Cities dashboard.  Run:  streamlit run dashboard/app.py"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from db import DEFAULT_URL, OUTPUTS, ROOT, get_engine, load_named_queries, read_cube  # noqa: E402

# ------------------------------------------------------------------ design tokens
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
INK, INK2, MUTED, GRID, GREY = "#0b0b0b", "#52514e", "#8a8984", "#e8e7e3", "#b9b7b0"
SEQ = ["#fdf0e9", "#f9c9b0", "#f19a70", "#eb6834", "#c24a1c", "#8a3311"]        # orange ramp, magnitude
BUCKETS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
BUCKET_COLORS = dict(zip(BUCKETS, ["#2e9d4a", "#9ccc3c", "#f2c12e", "#f28e2b", "#d6402b", "#7b1e1e"]))
CLUSTER_COLORS = {"Moderate urban": BLUE, "Smog belt": ORANGE, "Low-traffic, SO2-heavy": AQUA}
SEASON_COLORS = {"Winter": BLUE, "Pre-monsoon": ORANGE, "Monsoon": AQUA, "Post-monsoon": YELLOW}

pio.templates["bc"] = go.layout.Template(layout=dict(
    font=dict(family="Segoe UI, Inter, Arial, sans-serif", size=13, color=INK2),
    title=dict(font=dict(size=16, color=INK), x=0.01, xanchor="left", y=0.985, yref="container", yanchor="top"),
    colorway=[BLUE, ORANGE, AQUA, YELLOW, "#e87ba4", "#008300", "#4a3aa7", RED],
    paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(showgrid=False, linecolor=GRID, zeroline=False, ticks="", title=dict(font=dict(size=12))),
    yaxis=dict(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)", ticks="",
               title=dict(font=dict(size=12))),
    hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=INK, size=13)),
    legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title_text="", font=dict(size=12)),
    margin=dict(l=12, r=12, t=84, b=12), bargap=0.3))
pio.templates.default = "plotly_white+bc"

st.set_page_config(page_title="Breathing Cities", page_icon="🌫️", layout="wide")
st.markdown(f"""
<style>
.block-container {{padding-top: 2rem; max-width: 1400px;}}
.hero {{background: linear-gradient(120deg, #101828 0%, #1d2b45 100%); color: #fff; border-radius: 16px;
        padding: 26px 30px; margin-bottom: 18px;}}
.hero h1 {{font-size: 2.1rem; margin: 0 0 4px 0; color: #fff; font-weight: 700;}}
.hero p {{color: #c7cede; margin: 0 0 14px 0; font-size: 1.02rem;}}
.chip {{display: inline-block; background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.18);
        color: #e6eaf2; border-radius: 999px; padding: 3px 12px; margin: 0 6px 6px 0; font-size: .85rem;}}
.kpi {{background: #fff; border: 1px solid {GRID}; border-radius: 14px; padding: 14px 18px; min-height: 118px; margin-bottom: 10px;}}
[data-testid="stPlotlyChart"] {{border: 1px solid {GRID}; border-radius: 14px; overflow: hidden; background: #fff;}}
.kpi .label {{font-size: .74rem; color: {INK2}; text-transform: uppercase; letter-spacing: .05em; font-weight: 600;}}
.kpi .value {{font-size: 1.9rem; font-weight: 700; color: {INK}; line-height: 1.25;}}
.kpi .note {{font-size: .82rem; color: {MUTED};}}
.insight {{background: #eef4fc; border-radius: 10px; padding: 10px 14px; color: {INK}; font-size: .95rem;
           margin: 4px 0 14px 0;}}
.section {{font-size: 1.25rem; font-weight: 700; color: {INK}; margin: 18px 0 2px 0;}}
.sub {{color: {INK2}; font-size: .92rem; margin-bottom: 8px;}}
.card {{background: #fff; border: 1px solid {GRID}; border-radius: 14px; padding: 14px 16px; margin-bottom: 10px;}}
.card h4 {{margin: 0 0 4px 0; font-size: 1.02rem;}}
.dot {{display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px;}}
.stTabs [data-baseweb="tab-list"] {{gap: 6px;}}
.stTabs [data-baseweb="tab"] p {{font-size: 15px; font-weight: 600;}}
</style>""", unsafe_allow_html=True)

if DEFAULT_URL.startswith("sqlite") and not (ROOT / "warehouse" / "air_quality_dw.db").exists():
    st.error("No warehouse found. Run `python src/run_all.py` first (see README).")
    st.stop()


# ------------------------------------------------------------------ helpers
@st.cache_data
def load_cube():
    return read_cube(get_engine())


@st.cache_data
def query(sql):
    return pd.read_sql(sql, get_engine())


@st.cache_data
def output(name):
    path = OUTPUTS / name
    return pd.read_csv(path) if path.exists() else None


def show(fig, height=None):
    if height:
        fig.update_layout(height=height)
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, theme=None, width="stretch", config={"displayModeBar": False})


def kpi(col, label, value, note=""):
    col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                 f'<div class="note">{note}</div></div>', unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<div class="section">{title}</div>' + (f'<div class="sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)


def bucket_of(aqi):
    for limit, name in zip([50, 100, 200, 300, 400], BUCKETS):
        if aqi <= limit:
            return name
    return "Severe"


def missing(name):
    st.info(f"`outputs/{name}` not found. Run `python src/run_all.py` first.")


cube = load_cube()
all_cities = sorted(cube.city.unique())

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 🌫️ Breathing Cities")
    st.caption("DWDM (CSI3010) lab project")
    st.markdown("**Filters** (Overview tab)")
    cities = st.multiselect("Cities to compare", all_cities, default=["Delhi", "Bengaluru", "Mumbai"],
                            max_selections=4)
    years = st.slider("Years", int(cube.year.min()), int(cube.year.max()),
                      (int(cube.year.min()), int(cube.year.max())))
    st.divider()
    st.caption("Sources: CPCB via Kaggle (city + station readings) and Open-Meteo (weather). "
               "Warehouse: fact constellation, 9 tables, SQLite.")

ranged = cube[cube.year.between(*years)]
view = ranged[ranged.city.isin(cities or all_cities)]

# ------------------------------------------------------------------ header
st.markdown("""
<div class="hero">
  <h1>Breathing Cities: India Air Quality, 2015–2020</h1>
  <p>A data warehouse of pollution and weather for 26 Indian cities, used to find what makes the air bad,
  and how much the COVID lockdown and Diwali really changed it.</p>
  <span class="chip">26 cities</span><span class="chip">110 monitoring stations</span>
  <span class="chip">29,531 city-days</span><span class="chip">3 data sources</span>
  <span class="chip">Fact constellation · 9 tables</span>
</div>""", unsafe_allow_html=True)

tabs = st.tabs(["Overview", "OLAP explorer", "Stations & weather", "Prediction", "Patterns",
                "Lockdown what-if", "Diwali"])

# ================================================================== 1. Overview
with tabs[0]:
    lk = output("lockdown_summary.csv")
    c = st.columns(5)
    avg = view.aqi.mean()
    kpi(c[0], "City-days", f"{len(view):,}", "selected cities & years")
    kpi(c[1], "Average AQI", f"{avg:.0f}", bucket_of(avg))
    kpi(c[2], "Unhealthy days", f"{(view.aqi > 200).mean():.0%}", "AQI above 200")
    worst = view.groupby("city").aqi.mean().sort_values()
    kpi(c[3], "Worst selected city", worst.index[-1] if len(worst) else "–",
        f"average AQI {worst.iloc[-1]:.0f}" if len(worst) else "")
    if lk is not None:
        pm = lk.set_index("pollutant").loc["PM2.5"]
        kpi(c[4], "Lockdown effect", f"{pm.lockdown_2020:.0f}%", "PM2.5 vs expected")

    section("Where is the air worst?", "Average AQI per city for the selected years. Selected cities are highlighted.")
    left, right = st.columns([3, 2])
    city_avg = ranged.groupby(["city", "region"], as_index=False).aqi.mean().dropna()
    meta = pd.read_csv(ROOT / "data" / "city_meta.csv")
    geo = city_avg.merge(meta, on=["city", "region"])
    with left:
        fig = px.scatter_geo(geo, lat="latitude", lon="longitude", size="aqi", color="aqi", hover_name="city",
                             color_continuous_scale=SEQ, size_max=34,
                             hover_data={"aqi": ":.0f", "latitude": False, "longitude": False, "region": True})
        fig.update_geos(fitbounds="locations", showcountries=True, countrycolor="#cfcdc6", showland=True,
                        landcolor="#f3f2ef", showocean=True, oceancolor="#eaf1f9", showframe=False,
                        projection_type="mercator")
        fig.update_traces(marker=dict(line=dict(width=1, color="white"), opacity=0.92))
        fig.update_layout(coloraxis_colorbar=dict(title="AQI", thickness=12, len=0.6), margin=dict(t=10))
        show(fig, 520)
    with right:
        r = city_avg.sort_values("aqi")
        colors = [ORANGE if x in cities else GREY for x in r.city]
        fig = go.Figure(go.Bar(x=r.aqi, y=r.city, orientation="h", marker=dict(color=colors, cornerradius=4),
                               hovertemplate="%{y}: AQI %{x:.0f}<extra></extra>"))
        fig.update_layout(title="Average AQI by city", xaxis=dict(showgrid=True, gridcolor=GRID),
                          yaxis=dict(showgrid=False, tickfont=dict(size=11)))
        show(fig, 520)
    insight("Pollution is concentrated in the Indo-Gangetic plain (Delhi, Patna, Gurugram, Lucknow). "
            "Coastal and hill cities such as Aizawl, Shillong and Kochi are far cleaner.")

    section("How does it change over time?", "Monthly average AQI. The shaded band is the 2020 COVID lockdown.")
    monthly = view.set_index("full_date").groupby("city").aqi.resample("MS").mean().reset_index()
    fig = px.line(monthly, x="full_date", y="aqi", color="city", labels={"full_date": "", "aqi": "AQI"})
    fig.update_traces(line=dict(width=2.2))
    fig.update_layout(legend_title_text="", margin=dict(t=40))
    fig.add_vrect(x0="2020-03-25", x1="2020-05-31", fillcolor=GREY, opacity=0.25, line_width=0,
                  annotation_text="Lockdown", annotation_position="top left")
    show(fig, 380)
    insight("Every city follows the same yearly wave: worst in winter (Nov–Jan), cleanest in the monsoon "
            "(Jul–Sep), when rain washes pollution out and the air mixes higher.")

    section("Seasonal fingerprint", "Average AQI for each city and month. Darker = worse.")
    fp = ranged.groupby(["city", "month"]).aqi.mean().unstack()
    fp = fp.loc[fp.mean(axis=1).sort_values(ascending=False).index]
    fp.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(fp.columns)]
    fig = px.imshow(fp, color_continuous_scale=SEQ, aspect="auto", labels=dict(color="AQI", x="", y=""))
    fig.update_traces(hovertemplate="%{y}, %{x}: AQI %{z:.0f}<extra></extra>", xgap=2, ygap=2)
    fig.update_layout(coloraxis_colorbar=dict(thickness=12), xaxis=dict(side="top"))
    show(fig, 640)

# ================================================================== 2. OLAP explorer
with tabs[1]:
    queries = load_named_queries(ROOT / "sql" / "02_olap_queries.sql")
    OPS = {
        "rollup": ("Roll-up", "City → Region, yearly average PM2.5", "Zoom out: combine cities into regions."),
        "drilldown": ("Drill-down", "Delhi 2019: Year → Month", "Zoom in: split one year into months."),
        "slice": ("Slice", "Season = Winter, AQI by city", "Fix one dimension to a single value."),
        "dice": ("Dice", "North × Mar–May × 2019 vs 2020", "Filter on several dimensions at once."),
        "pivot": ("Pivot", "Cities × AQI buckets", "Rotate: turn bucket values into columns."),
        "events": ("Event dimension", "Average AQI by event", "Group by the Event dimension."),
        "drill_station": ("Drill-down to stations", "Delhi, Nov 2019: City → Station",
                          "Go below city level using the station fact table."),
        "weather_dice": ("Dice on weather", "Winter × wind × mixing height",
                         "Slice and dice by the Weather Condition dimension."),
        "drill_across": ("Drill-across", "Delhi 2019: pollution fact + weather fact",
                         "Combine two fact tables through the shared Date and City dimensions."),
    }
    st.markdown('<div class="section">Run OLAP operations live on the warehouse</div>', unsafe_allow_html=True)
    key = st.radio("Operation", list(OPS), format_func=lambda k: OPS[k][0], horizontal=True,
                   label_visibility="collapsed")
    name, what, how = OPS[key]
    st.markdown(f'<div class="sub"><b>{what}</b> · {how}</div>', unsafe_allow_html=True)
    res = query(queries[key])
    chart_col, data_col = st.columns([3, 2])

    with chart_col:
        if key == "rollup":
            fig = px.line(res, x="year", y="avg_pm25", color="region", markers=True,
                          labels={"avg_pm25": "Average PM2.5 (µg/m³)", "year": ""})
            fig.update_traces(line=dict(width=2.2), marker=dict(size=8))
            fig.update_xaxes(dtick=1)
        elif key in ("drilldown", "slice", "events"):
            x, y = {"drilldown": ("month_name", "avg_aqi"), "slice": ("city", "winter_aqi"),
                    "events": ("event_name", "avg_aqi")}[key]
            d = res.dropna(subset=[y])
            fig = go.Figure(go.Bar(x=d[x], y=d[y], marker=dict(color=[BUCKET_COLORS[bucket_of(v)] for v in d[y]],
                                                                 cornerradius=4),
                                   text=d[y].round(0), textposition="outside",
                                   hovertemplate="%{x}: AQI %{y:.0f}<extra></extra>"))
            fig.update_layout(yaxis_title="AQI", xaxis_tickangle=-40 if key == "slice" else 0)
            fig.add_annotation(text="Bar colour = official AQI bucket", xref="paper", yref="paper", x=1, y=1.08,
                               showarrow=False, font=dict(size=11, color=MUTED), xanchor="right")
        elif key == "dice":
            d = res.copy()
            d["year"] = d.year.astype(str)
            fig = px.bar(d, x="city", y="avg_pm25", color="year", barmode="group",
                         color_discrete_map={"2019": GREY, "2020": BLUE},
                         labels={"avg_pm25": "Average PM2.5 (µg/m³)", "city": "", "year": ""})
            fig.update_traces(marker_cornerradius=4)
        elif key == "pivot":
            d = res.set_index("city")[[b.lower().replace(" ", "_") for b in BUCKETS]]
            d.columns = BUCKETS
            share = d.div(d.sum(axis=1), axis=0).head(14).iloc[::-1] * 100
            fig = go.Figure([go.Bar(y=share.index, x=share[b], name=b, orientation="h",
                                    marker=dict(color=BUCKET_COLORS[b], line=dict(color="white", width=1)),
                                    hovertemplate="%{y}: %{x:.0f}% " + b + "<extra></extra>") for b in BUCKETS])
            fig.update_layout(barmode="stack", xaxis=dict(title="% of days", ticksuffix="%"),
                              title="Share of days in each AQI bucket (14 worst cities)")
        elif key == "drill_station":
            d = res.sort_values("avg_pm25")
            d["station"] = d.station_name.str.split(",").str[0]
            fig = go.Figure(go.Bar(x=d.avg_pm25, y=d.station, orientation="h",
                                   marker=dict(color=d.avg_pm25, colorscale=SEQ[1:], cornerradius=3),
                                   hovertemplate="%{y}: PM2.5 %{x:.0f}<extra></extra>"))
            fig.update_layout(xaxis_title="Average PM2.5 (µg/m³)", yaxis=dict(tickfont=dict(size=10)),
                              xaxis=dict(showgrid=True, gridcolor=GRID))
            fig.update_layout(height=max(420, 20 * len(d)))
        elif key == "weather_dice":
            d = res.pivot(index="wind_band", columns="mixing_band", values="pct_unhealthy")
            d = d.reindex(index=["Calm", "Light", "Breezy"], columns=["Low", "Medium", "High"])
            fig = px.imshow(d, color_continuous_scale=SEQ, aspect="auto",
                            labels=dict(x="Mixing height", y="Wind", color="% unhealthy"))
            fig.update_traces(text=[["no data" if pd.isna(v) else f"{v:.0f}%" for v in r_] for r_ in d.values],
                              xgap=3, ygap=3, texttemplate="%{text}", textfont=dict(size=16),
                              hovertemplate="%{y} wind, %{x} mixing: %{z:.0f}% of winter days unhealthy<extra></extra>")
            fig.update_layout(title="% of winter days that are unhealthy")
        else:  # drill_across: two panels sharing the month axis (never a dual axis)
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            m = [months[i - 1] for i in res.month]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                subplot_titles=("PM2.5 (from fact_air_quality)", "Mixing height, m (from fact_weather)"))
            fig.add_bar(x=m, y=res.avg_pm25, marker=dict(color=ORANGE, cornerradius=4), row=1, col=1,
                        hovertemplate="%{x}: PM2.5 %{y:.0f}<extra></extra>", showlegend=False)
            fig.add_scatter(x=m, y=res.avg_mixing_height, mode="lines+markers", line=dict(color=BLUE, width=2.5),
                            marker=dict(size=8), row=2, col=1, showlegend=False,
                            hovertemplate="%{x}: %{y:.0f} m<extra></extra>")
            fig.update_annotations(font=dict(size=13, color=INK2), x=0, xanchor="left")
        show(fig, fig.layout.height or 460)

    with data_col:
        st.dataframe(res, hide_index=True, width="stretch", height=300)
        with st.expander("Show SQL", expanded=True):
            st.code(queries[key], language="sql")

    notes = {
        "rollup": "North India's PM2.5 fell steadily from 2016 to 2019, and dropped further in 2020 (half-year, incl. lockdown).",
        "drilldown": "Delhi is in 'Very Poor' territory in winter and only 'Satisfactory' in August.",
        "slice": "In winter, the Indo-Gangetic cities (Delhi, Patna, Lucknow) are far worse than everyone else.",
        "dice": "Every northern city was cleaner in the lockdown spring of 2020 than in spring 2019.",
        "pivot": "Delhi, Patna and Lucknow spend a large share of days in Very Poor or Severe.",
        "events": "Stubble-burning days average AQI ~293, nearly double a normal day; lockdown days only ~88.",
        "drill_station": "Inside one city, pollution differs about 1.7× between the cleanest and worst station.",
        "weather_dice": "Same season, very different air: calm + low mixing → ~60% unhealthy; breezy + high → ~1%.",
        "drill_across": "Delhi's worst months are exactly the months with the lowest mixing height.",
    }
    insight(notes[key])

# ================================================================== 3. Stations & weather
with tabs[2]:
    section("Inside a city: monitoring stations", "Drill-down from City to Station. Average PM2.5 per station and year.")
    city = st.selectbox("City", all_cities, index=all_cities.index("Delhi"))
    stations = query(f"""
        SELECT s.station_name, d.year, ROUND(AVG(f.pm25), 1) AS avg_pm25
        FROM fact_station_air_quality f
        JOIN dim_station s ON f.station_key = s.station_key
        JOIN dim_city c ON s.city_key = c.city_key
        JOIN dim_date d ON f.date_key = d.date_key
        WHERE c.city = '{city.replace("'", "''")}'
        GROUP BY s.station_name, d.year""")
    if stations.empty:
        st.info("No station-level data for this city.")
    else:
        stations["station"] = stations.station_name.str.split(",").str[0]
        pv = stations.pivot_table(index="station", columns="year", values="avg_pm25")
        latest = 2019 if 2019 in pv.columns else pv.columns.max()
        pv = pv.loc[pv[latest].fillna(pv.mean(axis=1)).sort_values(ascending=False).index]
        fig = px.imshow(pv, color_continuous_scale=SEQ, aspect="auto", text_auto=".0f",
                        labels=dict(color="PM2.5", x="", y=""))
        fig.update_traces(xgap=2, ygap=2, hovertemplate="%{y}, %{x}: PM2.5 %{z:.0f}<extra></extra>")
        fig.update_layout(xaxis=dict(side="top", dtick=1), coloraxis_colorbar=dict(thickness=12))
        show(fig, max(320, 26 * len(pv) + 80))
        insight("Stations are sorted by their 2019 level, worst first. Blank cells mean the station was not "
                "reporting yet; most of Delhi's stations opened in 2018.")

    section("Weather: what traps pollution?",
            "Mixing height is how high the air can mix. Low = pollution trapped near the ground.")
    left, right = st.columns(2)
    with left:
        w = ranged.dropna(subset=["pm25", "mixing_height"]).copy()
        edges = [0, 200, 300, 400, 500, 650, 800, 1000, 1300, 2000, 6000]
        w["bin"] = pd.cut(w.mixing_height, edges)
        g = w.groupby(["season", "bin"], observed=True).agg(pm25=("pm25", "mean"), n=("pm25", "size"),
                                                            mid=("mixing_height", "median")).reset_index()
        g = g[g.n >= 30]
        fig = go.Figure()
        for s_, col in SEASON_COLORS.items():
            d = g[g.season == s_]
            fig.add_scatter(x=d.mid, y=d.pm25, name=s_, mode="lines+markers", line=dict(color=col, width=2.5),
                            marker=dict(size=8), hovertemplate=s_ + ": %{x:.0f} m → PM2.5 %{y:.0f}<extra></extra>")
        fig.update_layout(title="Winter smog fades as the mixing height rises",
                          xaxis=dict(title="Mixing height (m, log scale)", type="log",
                                     tickvals=[200, 300, 500, 1000, 2000],
                                     ticktext=["200", "300", "500", "1,000", "2,000"]),
                          yaxis_title="Average PM2.5 (µg/m³)")
        show(fig, 420)
    with right:
        wd = query(queries["weather_dice"]).pivot(index="wind_band", columns="mixing_band", values="pct_unhealthy")
        wd = wd.reindex(index=["Calm", "Light", "Breezy"], columns=["Low", "Medium", "High"])
        fig = px.imshow(wd, color_continuous_scale=SEQ, aspect="auto",
                        labels=dict(x="Mixing height", y="Wind", color="% unhealthy"))
        fig.update_traces(text=[["no data" if pd.isna(v) else f"{v:.0f}%" for v in r_] for r_ in wd.values],
                          xgap=3, ygap=3, texttemplate="%{text}", textfont=dict(size=16),
                          hovertemplate="%{y} wind, %{x} mixing: %{z:.0f}%<extra></extra>")
        fig.update_layout(title="% of winter days unhealthy, by wind × mixing")
        show(fig, 420)
    insight("In winter and post-monsoon, PM2.5 roughly halves as the mixing height rises from 200 m to 1,000 m. "
            "In pre-monsoon and monsoon the air is already clean (rain, sea breeze), so mixing height matters little. "
            "On winter days, calm wind + low mixing makes ~60% of days unhealthy; breezy + high mixing, ~1%.")

# ================================================================== 4. Prediction
with tabs[3]:
    sc, shap_imp = output("classification_scores.csv"), output("shap_importance.csv")
    if sc is None or shap_imp is None:
        missing("classification_scores.csv")
    else:
        section("Classification: will tomorrow be unhealthy?",
                "Trained on 2015–2018, tested on 2019–2020. Unhealthy = AQI above 200.")
        s = sc.set_index("model")
        c = st.columns(4)
        kpi(c[0], "Random Forest F1", f"{s.loc['rf_pollution', 'f1']:.2f}", "higher is better")
        kpi(c[1], "Baseline F1", f"{s.loc['persistence', 'f1']:.2f}", "'tomorrow = today'")
        kpi(c[2], "Change days predicted", f"{s.loc['rf_pollution', 'flip_day_accuracy']:.0%}",
            "baseline gets 0% by definition")
        kpi(c[3], "ROC-AUC", f"{s.loc['rf_pollution', 'roc_auc']:.3f}", "ranking quality, 1 = perfect")
        names = {"persistence": "Baseline<br>(tomorrow = today)", "tree_pollution": "Decision<br>Tree",
                 "rf_pollution": "Random<br>Forest", "rf_weather": "Random Forest<br>+ weather"}
        left, right = st.columns(2)
        with left:
            d = s.reset_index()
            d["Model"] = d.model.map(names)
            fig = go.Figure([go.Bar(x=d.Model, y=d.f1, name="F1 score", marker=dict(color=BLUE, cornerradius=4),
                                    text=d.f1.map("{:.2f}".format), textposition="outside"),
                             go.Bar(x=d.Model, y=d.flip_day_accuracy, name="Accuracy on change days",
                                    marker=dict(color=ORANGE, cornerradius=4),
                                    text=d.flip_day_accuracy.map("{:.0%}".format), textposition="outside")])
            fig.update_layout(title="Model comparison (test years 2019–2020)", barmode="group",
                              yaxis=dict(range=[0, 1.05]), xaxis=dict(tickangle=0))
            show(fig, 430)
        with right:
            nice = {"aqi": "Today's AQI", "pm25": "Today's PM2.5", "aqi_3day_mean": "3-day average AQI",
                    "pm10": "PM10", "co": "CO", "region_North": "North region", "nox": "NOx",
                    "precipitation": "Rain today", "next_precipitation": "Rain tomorrow",
                    "temp_mean": "Temperature", "aqi_change": "AQI change vs yesterday", "month": "Month",
                    "no2": "NO2", "nh3": "NH3", "o3": "O3", "so2": "SO2", "mixing_height": "Mixing height"}
            shap_imp.columns = ["feature", "value"]
            top = shap_imp.head(10).iloc[::-1]
            fig = go.Figure(go.Bar(x=top.value, y=top.feature.map(lambda f: nice.get(f, f)), orientation="h",
                                   marker=dict(color=BLUE, cornerradius=4),
                                   hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
            fig.update_layout(title="What drives the prediction (SHAP)", xaxis=dict(title="Mean |SHAP| (impact)",
                                                                                   showgrid=True, gridcolor=GRID))
            show(fig, 430)
        insight("Random Forest beats the 'tomorrow = today' baseline and catches a third of the days when air "
                "quality changes. SHAP shows it relies mostly on today's and recent pollution levels.")

    bt, fc = output("forecast_backtest.csv"), output("forecast_2019.csv")
    if bt is not None and fc is not None:
        section("Forecasting monthly AQI for 2019", "Each model was trained only on data up to December 2018.")
        long_ = bt[bt.long_history].mean(numeric_only=True)
        left, right = st.columns([2, 3])
        with left:
            meth = {"seasonal_naive": "Seasonal naive", "holt_winters": "Holt-Winters", "sarima": "SARIMA",
                    "weather_adjusted": "Weather-adjusted"}
            vals = [long_[m] for m in meth]
            fig = go.Figure(go.Bar(x=list(meth.values()), y=vals, text=[f"{v:.1f}%" for v in vals],
                                   textposition="outside",
                                   marker=dict(color=[GREY, GREY, GREY, BLUE], cornerradius=4)))
            fig.update_layout(title="Average error (MAPE, lower is better)", yaxis=dict(ticksuffix="%"))
            show(fig, 420)
        with right:
            city_f = st.selectbox("City", sorted(fc.city.unique()),
                                  index=sorted(fc.city.unique()).index("Delhi") if "Delhi" in set(fc.city) else 0)
            d = fc[fc.city == city_f]
            fig = go.Figure()
            for col_, nm, colr, dash in [("actual", "Actual", INK, "solid"),
                                         ("seasonal_naive", "Seasonal naive", GREY, "dot"),
                                         ("weather_adjusted", "Weather-adjusted", BLUE, "solid")]:
                fig.add_scatter(x=pd.to_datetime(d.month), y=d[col_], name=nm, mode="lines+markers",
                                line=dict(color=colr, width=2.5, dash=dash), marker=dict(size=7))
            fig.update_layout(title=f"{city_f}, 2019: actual vs forecast", yaxis_title="AQI")
            show(fig, 420)
        insight("Most cities have only 1–3 years of history, too little for Holt-Winters or SARIMA. Adjusting "
                f"last year's value for this year's weather is the most accurate ({long_.weather_adjusted:.1f}% vs "
                f"{long_.seasonal_naive:.1f}%).")

# ================================================================== 5. Patterns (clusters + rules)
with tabs[4]:
    cl, cs = output("city_clusters.csv"), output("cluster_summary.csv")
    if cl is not None and cs is not None:
        section("Clustering: city pollution profiles",
                "K-Means on 9 features per city (k = 3, chosen by silhouette score). PCA compresses 9 features into 2 axes: "
                "nearby cities are similar.")
        left, right = st.columns([3, 2])
        with left:
            fig = px.scatter(cl, x="pc1", y="pc2", color="cluster_name", text="city", color_discrete_map=CLUSTER_COLORS,
                             hover_data={"pc1": False, "pc2": False, "pm25": ":.0f", "no2": ":.0f",
                                         "unhealthy_day_share": ":.0%"},
                             labels={"pc1": "Component 1", "pc2": "Component 2", "cluster_name": "Cluster",
                                     "pm25": "PM2.5", "no2": "NO2", "unhealthy_day_share": "Unhealthy days"})
            fig.update_traces(marker=dict(size=15, line=dict(width=2, color="white")), textposition="top center",
                              textfont=dict(size=11, color=INK2))
            fig.update_layout(xaxis=dict(showticklabels=False, showgrid=True, gridcolor=GRID),
                              yaxis=dict(showticklabels=False))
            show(fig, 520)
        with right:
            blurb = {"Smog belt": "Indo-Gangetic plain. Half of all days unhealthy; hit hardest by trapped air.",
                     "Moderate urban": "Typical city air: traffic NO2, rarely unhealthy.",
                     "Low-traffic, SO2-heavy": "Industrial and hill/coastal towns: little NO2, more SO2 (coal)."}
            for nm in ["Smog belt", "Moderate urban", "Low-traffic, SO2-heavy"]:
                r = cs[cs.name == nm].iloc[0]
                members = r.cities.split(", ")
                who = r.cities if len(members) <= 6 else f"{len(members)} cities incl. " + ", ".join(members[:5]) + "…"
                st.markdown(f"""<div class="card"><h4><span class="dot" style="background:{CLUSTER_COLORS[nm]}"></span>{nm}</h4>
                <div class="sub" style="margin:0 0 6px 0">{blurb[nm]}</div>
                <div style="font-size:.86rem;color:{INK2}">{who}</div>
                <div style="margin-top:8px;font-size:.92rem"><b>PM2.5</b> {r.pm25:.0f} · <b>NO2</b> {r.no2:.0f} ·
                <b>SO2</b> {r.so2:.0f} · <b>Unhealthy</b> {r.unhealthy_day_share:.0%}</div></div>""",
                            unsafe_allow_html=True)

    ctx, cp = output("rules_context.csv"), output("rules_copollution.csv")
    if ctx is not None and cp is not None:
        section("Association rules: which conditions bring bad air?",
                "FP-Growth, mined separately per region. Bars = how often the air is unhealthy when the conditions "
                "hold; dashed line = the region's normal rate.")
        regions = sorted(ctx.region.unique())
        region = st.radio("Region", regions, index=regions.index("North"), horizontal=True)
        d = ctx[ctx.region == region].sort_values(["confidence", "lift"], ascending=False).head(8).iloc[::-1]
        plain = {"Mixing=Low": "low mixing", "Mixing=Medium": "medium mixing", "Mixing=High": "high mixing",
                 "Wind=Calm": "calm wind", "Wind=Light": "light wind", "Wind=Breezy": "breezy",
                 "Rain=Dry": "dry", "Rain=Rain": "rain", "Temp=Cold": "cold", "Temp=Mild": "mild",
                 "Temp=Hot": "hot", "Day=Weekday": "weekday", "Day=Weekend": "weekend",
                 "Event=Stubble burning": "stubble burning", "Event=Diwali": "Diwali",
                 "Event=Normal": "normal day", "Event=COVID lockdown": "lockdown"}
        d["rule"] = d.antecedents.apply(lambda a: " + ".join(plain.get(i, i.split("=")[-1].lower())
                                                             for i in a.split(" & ")))
        base = d.base_rate.iloc[0]
        fig = go.Figure(go.Bar(x=d.confidence, y=d.rule, orientation="h", marker=dict(color=BLUE, cornerradius=4),
                               text=[f"{c:.0%}  (lift {l:.1f})" for c, l in zip(d.confidence, d.lift)],
                               textposition="outside",
                               customdata=d[["support", "lift"]],
                               hovertemplate="%{y}<br>Unhealthy: %{x:.0%}<br>Support: %{customdata[0]:.1%}"
                                             "<br>Lift: %{customdata[1]:.2f}<extra></extra>"))
        fig.add_vline(x=base, line=dict(color=ORANGE, dash="dash", width=2),
                      annotation_text=f"normal rate {base:.0%}", annotation_position="top")
        fig.update_layout(xaxis=dict(range=[0, 1.15], tickformat=".0%", showgrid=True, gridcolor=GRID,
                                     title="Share of days unhealthy when the rule applies"),
                          yaxis=dict(tickfont=dict(size=12)), margin=dict(l=8, r=8, t=40, b=8))
        show(fig, max(320, 52 * len(d) + 80))
        top = d.iloc[-1]
        insight(f"{region}: when <b>{top.rule}</b>, the air is unhealthy on {top.confidence:.0%} of days, "
                f"versus {base:.0%} normally ({top.lift:.1f}× as likely).")
        with st.expander("Co-pollution rules: which pollutants exceed their limits together"):
            st.dataframe(cp[cp.region == region].round(3), hide_index=True, width="stretch")

# ================================================================== 6. Lockdown
with tabs[5]:
    eff, summ, daily = output("lockdown_effect.csv"), output("lockdown_summary.csv"), output("lockdown_delhi_daily.csv")
    if eff is None or summ is None or daily is None:
        missing("lockdown_effect.csv")
    else:
        section("Did the COVID lockdown clean the air?",
                "Actual pollution (25 Mar – 31 May 2020) vs what a weather model expected without a lockdown.")
        sm = summ.set_index("pollutant")
        c = st.columns(3)
        for col_, p in zip(c, ["NO2", "PM2.5", "AQI"]):
            kpi(col_, f"{p} change", f"{sm.loc[p, 'lockdown_2020']:.0f}%",
                f"placebo year 2019: {sm.loc[p, 'placebo_2019']:.0f}% · median of {int(sm.loc[p, 'cities'])} cities")

        daily["full_date"] = pd.to_datetime(daily.full_date)
        fig = go.Figure()
        fig.add_scatter(x=daily.full_date, y=daily.expected_pm25, name="Expected without lockdown",
                        line=dict(color=GREY, width=2.5, dash="dot"))
        fig.add_scatter(x=daily.full_date, y=daily.pm25, name="Actual", line=dict(color=BLUE, width=2.5))
        fig.add_vrect(x0="2020-03-25", x1="2020-05-31", fillcolor=GREY, opacity=0.2, line_width=0,
                      annotation_text="Lockdown", annotation_position="top left")
        fig.update_layout(title="Delhi 2020: daily PM2.5, actual vs expected", yaxis_title="PM2.5 (µg/m³)")
        show(fig, 400)
        insight("Before the lockdown the model tracks reality closely. During the lockdown, real PM2.5 sits well "
                "below what the weather alone would predict. That gap is the lockdown effect.")

        left, right = st.columns([2, 3])
        with left:
            pols = ["NO2", "PM2.5", "AQI"]
            fig = go.Figure([go.Bar(x=pols, y=[sm.loc[p, "lockdown_2020"] for p in pols], name="Lockdown 2020",
                                    marker=dict(color=BLUE, cornerradius=4),
                                    text=[f"{sm.loc[p, 'lockdown_2020']:.0f}%" for p in pols], textposition="outside"),
                             go.Bar(x=pols, y=[sm.loc[p, "placebo_2019"] for p in pols], name="Placebo 2019",
                                    marker=dict(color=GREY, cornerradius=4),
                                    text=[f"{sm.loc[p, 'placebo_2019']:.0f}%" for p in pols], textposition="outside")])
            fig.update_layout(title="Placebo test: ~0 in a normal year", barmode="group",
                              yaxis=dict(ticksuffix="%", range=[-36, 5]))
            show(fig, 560)
        with right:
            p = st.radio("Pollutant", ["PM2.5", "NO2", "AQI"], horizontal=True)
            e = eff[eff.pollutant == p].sort_values("change_pct", ascending=False)
            e = e[e.change_pct.abs() < 150]   # hide sensor-level outliers (tiny baselines)
            fig = go.Figure(go.Bar(x=e.change_pct, y=e.city, orientation="h",
                                   marker=dict(color=[RED if v > 0 else BLUE for v in e.change_pct], cornerradius=3),
                                   hovertemplate="%{y}: %{x:.0f}%<extra></extra>"))
            fig.update_layout(title=f"{p}: change vs expected, by city", xaxis=dict(ticksuffix="%", showgrid=True,
                                                                                   gridcolor=GRID),
                              yaxis=dict(tickfont=dict(size=11)))
            show(fig, 560)
        insight("Blue = cleaner than expected, red = dirtier. Traffic-heavy cities improved most; coal and mining "
                "towns (Brajrajnagar, Jorapokhar) got dirtier because industry kept running.")

# ================================================================== 7. Diwali
with tabs[6]:
    dw = output("diwali_excess.csv")
    if dw is None:
        missing("diwali_excess.csv")
    else:
        section("Diwali fireworks: did the firecracker rules work?",
                "PM2.5 on Diwali night + the next day, vs what the weather model expected without fireworks.")
        dl = dw[dw.city == "Delhi"].set_index("year")
        c = st.columns(3)
        kpi(c[0], "Delhi raw PM2.5, 2016 → 2017", f"{dl.loc[2016, 'actual']:.0f} → {dl.loc[2017, 'actual']:.0f}",
            "looks like the 2017 ban worked")
        kpi(c[1], "Weather-adjusted spike, 2017", f"+{dl.loc[2017, 'excess_pct']:.0f}%",
            f"expected only {dl.loc[2017, 'expected']:.0f} without fireworks")
        kpi(c[2], "Spike in 2018–19", f"+{dl.loc[2019, 'excess_pct']:.0f}%", "green crackers, 8–10 pm only")
        left, right = st.columns(2)
        with left:
            yrs = [str(y) for y in dl.index]
            fig = go.Figure([go.Bar(x=yrs, y=dl.expected.tolist(), name="Expected (weather only)",
                                    marker=dict(color=GREY, cornerradius=4)),
                             go.Bar(x=yrs, y=dl.actual.tolist(), name="Actual", marker=dict(color=ORANGE, cornerradius=4),
                                    text=[f"+{v:.0f}%" for v in dl.excess_pct], textposition="outside")])
            fig.add_annotation(x=yrs.index("2017"), y=dl.loc[2017, "actual"] + 60, text="Sales ban", showarrow=False,
                               font=dict(color=INK, size=12))
            fig.add_annotation(x=yrs.index("2018"), y=dl.loc[2018, "actual"] + 60, text="Green crackers", showarrow=False,
                               font=dict(color=INK, size=12))
            fig.update_layout(title="Delhi: actual vs expected PM2.5 on Diwali", barmode="group",
                              xaxis=dict(type="category"),
                              yaxis=dict(title="PM2.5 (µg/m³)", range=[0, dl.actual.max() * 1.3]))
            show(fig, 440)
        with right:
            d = dw[dw.excess_pct.between(-100, 300)].copy()
            d["year"] = d.year.astype(str)
            fig = px.strip(d, x="year", y="excess_pct", hover_name="city", color_discrete_sequence=[BLUE],
                           labels={"excess_pct": "Extra PM2.5 vs expected (%)", "year": ""})
            fig.update_traces(marker=dict(size=10, opacity=0.75, line=dict(width=1, color="white")), jitter=0.4)
            med = dw.groupby("year").excess_pct.median()
            fig.add_scatter(x=[str(y) for y in med.index], y=med.values, mode="lines+markers", name="Median city",
                            line=dict(color=INK, width=2), marker=dict(size=9, symbol="diamond"))
            fig.add_hline(y=0, line=dict(color=GRID, width=1))
            fig.update_layout(title="All cities: Diwali spike each year", yaxis=dict(ticksuffix="%"),
                              xaxis=dict(type="category"))
            show(fig, 440)
        insight("Raw numbers made the 2017 ban look effective, but 2017 simply had good weather. After removing "
                "the weather's effect, the firework spike was actually the largest that year.")
