"""Diwali: how much extra PM2.5 do fireworks add, and did the firecracker restrictions help?

The weather model is trained on all days EXCEPT the Diwali windows and the lockdown, so it learns
what late-Oct/Nov air looks like for the given weather without fireworks. On Diwali night and the
morning after (day 0 and day +1) we compare actual PM2.5 with that expectation.

Policy timeline (Supreme Court of India):
  2017  sale of firecrackers suspended in Delhi-NCR before Diwali
  2018  only 'green' crackers allowed, 8-10 pm, nationwide
Comparing the excess by year is descriptive, not proof; other factors change between years too.
"""
import pandas as pd

import deweather
from db import OUTPUTS, get_engine, read_cube

DIWALI = pd.to_datetime(["2015-11-11", "2016-10-30", "2017-10-19", "2018-11-07", "2019-10-27"])
LOCKDOWN = ("2020-03-25", "2020-05-31")


def main():
    df = deweather.add_features(read_cube(get_engine()))
    window = pd.Series(False, index=df.index)
    for day in DIWALI:
        window |= df.full_date.between(day - pd.Timedelta(days=2), day + pd.Timedelta(days=3))
    train = ~window & ~df.full_date.between(*LOCKDOWN)
    model = deweather.fit(df, "pm25", train)

    rows = []
    for day in DIWALI:
        peak = df[df.full_date.isin([day, day + pd.Timedelta(days=1)]) & df.pm25.notna()].copy()
        peak["expected"] = deweather.predict(model, peak)
        rows.append(peak.groupby("city").agg(actual=("pm25", "mean"), expected=("expected", "mean"))
                    .assign(year=day.year).reset_index())
    res = pd.concat(rows)
    res["excess"] = res.actual - res.expected
    res["excess_pct"] = (res.actual / res.expected - 1) * 100

    by_year = res.groupby("year").agg(cities=("city", "nunique"), median_excess_pct=("excess_pct", "median"))
    delhi = res[res.city == "Delhi"].set_index("year")[["actual", "expected", "excess", "excess_pct"]]
    OUTPUTS.mkdir(exist_ok=True)
    res.round(1).to_csv(OUTPUTS / "diwali_excess.csv", index=False)
    print("All cities, median % excess PM2.5 on Diwali night + next day:\n", by_year.round(1).to_string())
    print("\nDelhi:\n", delhi.round(1).to_string())


if __name__ == "__main__":
    main()
