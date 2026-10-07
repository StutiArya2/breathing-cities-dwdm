"""Association rules (FP-Growth), mined separately for each region (the 'segment').

Each city-day is a 'basket' of items. Two rule sets:

1. Context rules     {weather, season, event, weekday items} -> Air=Unhealthy
                     Pollutant readings are NOT allowed on the left, because AQI is computed from
                     them (PM2.5 high -> AQI bad is trivially true). These rules say which
                     *conditions* bring bad air.
2. Co-pollution rules {pollutant exceedances} -> {another pollutant exceedance}
                     Which pollutants rise together, hinting at shared sources
                     (e.g. NO2 + CO = traffic, SO2 = industry / coal).
"""
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth

from db import OUTPUTS, get_engine, read_cube

# A pollutant is "High" above its CPCB 24-hour standard (the top of the Satisfactory band).
STANDARDS = {"PM2.5": ("pm25", 60), "PM10": ("pm10", 100), "NO2": ("no2", 80),
             "SO2": ("so2", 80), "CO": ("co", 2), "O3": ("o3", 100), "NH3": ("nh3", 200)}
CONTEXT_PREFIXES = ("Season=", "Event=", "Day=", "Wind=", "Mixing=", "Rain=", "Temp=")


def context_baskets(cube):
    items = pd.DataFrame({
        "Season": "Season=" + cube.season, "Event": "Event=" + cube.event_name,
        "Day": "Day=" + cube.is_weekend.map({0: "Weekday", 1: "Weekend"}),
        "Wind": "Wind=" + cube.wind_band, "Mixing": "Mixing=" + cube.mixing_band,
        "Rain": "Rain=" + cube.rain_band, "Temp": "Temp=" + cube.temp_band,
    })
    onehot = pd.get_dummies(items, prefix="", prefix_sep="").astype(bool)
    onehot["Air=Unhealthy"] = cube.aqi > 200
    return onehot[cube.aqi.notna()]


def pollutant_baskets(cube):
    onehot = pd.DataFrame({f"{name}=High": cube[col] > limit for name, (col, limit) in STANDARDS.items()})
    return onehot[onehot.any(axis=1)]  # days where at least one standard is exceeded


def mine(baskets, min_support, min_confidence, consequent_filter=None):
    itemsets = fpgrowth(baskets, min_support=min_support, use_colnames=True, max_len=4)
    if itemsets.empty:
        return pd.DataFrame()
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[(rules.lift > 1.2) & (rules.consequents.apply(len) == 1)]
    if consequent_filter:
        rules = rules[rules.consequents.apply(lambda c: next(iter(c)) == consequent_filter)]
    rules = rules.assign(antecedents=rules.antecedents.apply(lambda s: " & ".join(sorted(s))),
                         consequents=rules.consequents.apply(lambda s: next(iter(s))))
    return rules[["antecedents", "consequents", "support", "confidence", "lift"]] \
        .sort_values(["lift", "support"], ascending=False)


def main():
    cube = read_cube(get_engine())
    context, copoll = [], []
    for region, part in cube.groupby("region"):
        base = (part.aqi > 200).mean()
        rules = mine(context_baskets(part), 0.02, 0.5, "Air=Unhealthy")
        if not rules.empty:
            context.append(rules.assign(region=region, base_rate=round(base, 3)))
            print(f"\n=== {region}: context rules (base unhealthy rate {base:.1%}) ===")
            print(rules.head(4).round(3).to_string(index=False))
        rules = mine(pollutant_baskets(part), 0.05, 0.6)
        if not rules.empty:
            copoll.append(rules.assign(region=region))
    OUTPUTS.mkdir(exist_ok=True)
    pd.concat(context).round(4).to_csv(OUTPUTS / "rules_context.csv", index=False)
    cp = pd.concat(copoll).round(4)
    cp.to_csv(OUTPUTS / "rules_copollution.csv", index=False)
    print("\n=== Co-pollution rules (top by lift) ===")
    print(cp.sort_values("lift", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
