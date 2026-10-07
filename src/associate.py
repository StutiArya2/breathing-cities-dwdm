"""Association rules: which pollutant + context conditions co-occur with bad-air days?

Each city-day becomes a 'basket' of items such as PM2.5=High, NO2=High,
Season=Winter, Event=Diwali, AQI=Very Poor. FP-Growth mines frequent itemsets
per region (the 'segment'), and we keep rules whose consequent is an AQI bucket.
"""
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth

from db import OUTPUTS, get_engine, read_cube

# Bins follow the CPCB sub-index breakpoints (24h standard = upper edge of "Satisfactory").
BINS = {
    "PM2.5": ("pm25", [0, 60, 120, float("inf")]),
    "PM10": ("pm10", [0, 100, 250, float("inf")]),
    "NO2": ("no2", [0, 80, 180, float("inf")]),
    "SO2": ("so2", [0, 80, 380, float("inf")]),
    "CO": ("co", [0, 2, 10, float("inf")]),
    "O3": ("o3", [0, 100, 168, float("inf")]),
}
LEVELS = ["Low", "High", "Very high"]


def to_baskets(cube):
    items = pd.DataFrame(index=cube.index)
    for label, (col, edges) in BINS.items():
        items[label] = label + "=" + pd.cut(cube[col], edges, labels=LEVELS, right=False).astype(str)
    items["Season"] = "Season=" + cube.season
    items["Event"] = "Event=" + cube.event_name
    items["Day"] = "Day=" + cube.is_weekend.map({0: "Weekday", 1: "Weekend"})
    items["AQI"] = "AQI=" + cube.bucket
    onehot = pd.get_dummies(items, prefix="", prefix_sep="").astype(bool)
    return onehot.loc[:, ~onehot.columns.str.endswith("=nan") & (onehot.columns != "AQI=Unknown")]


def mine(baskets, min_support=0.03, min_confidence=0.6):
    itemsets = fpgrowth(baskets, min_support=min_support, use_colnames=True, max_len=4)
    if itemsets.empty:
        return pd.DataFrame()
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    is_aqi = rules.consequents.apply(lambda c: len(c) == 1 and next(iter(c)).startswith("AQI="))
    no_aqi_lhs = rules.antecedents.apply(lambda a: not any(i.startswith("AQI=") for i in a))
    rules = rules[is_aqi & no_aqi_lhs & (rules.lift > 1.2)].copy()
    rules["antecedents"] = rules.antecedents.apply(lambda s: " & ".join(sorted(s)))
    rules["consequents"] = rules.consequents.apply(lambda s: next(iter(s)))
    return rules[["antecedents", "consequents", "support", "confidence", "lift"]] \
        .sort_values("lift", ascending=False)


def main():
    cube = read_cube(get_engine())
    all_rules = []
    for region, part in cube.groupby("region"):
        rules = mine(to_baskets(part))
        if not rules.empty:
            all_rules.append(rules.assign(region=region))
            print(f"\n=== {region}: {len(rules)} rules ===")
            print(rules.head(5).round(3).to_string(index=False))
    OUTPUTS.mkdir(exist_ok=True)
    if all_rules:
        pd.concat(all_rules).round(4).to_csv(OUTPUTS / "association_rules.csv", index=False)


if __name__ == "__main__":
    main()
