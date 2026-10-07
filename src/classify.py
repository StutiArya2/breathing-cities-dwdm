"""Classification: will TOMORROW be an unhealthy-air day (AQI > 200) in this city?

Features use only today's readings + calendar/context, so there is no leakage of
tomorrow's AQI. Chronological split: train <= 2018, test 2019-2020.
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from db import OUTPUTS, get_engine, read_cube

FEATURES = ["pm25", "pm10", "no2", "nox", "nh3", "co", "so2", "o3", "aqi",
            "aqi_3day_mean", "month", "is_weekend", "is_coastal",
            "region_North", "event_Stubble burning", "event_Diwali", "event_COVID lockdown"]


def build_dataset(cube):
    df = cube.copy()
    g = df.groupby("city")
    df["aqi_3day_mean"] = g["aqi"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["next_date"] = g["full_date"].shift(-1)
    df["next_aqi"] = g["aqi"].shift(-1)
    # keep only consecutive days with a known next-day AQI
    df = df[(df.next_date - df.full_date).dt.days.eq(1) & df.next_aqi.notna()].copy()
    df["target_unhealthy"] = (df.next_aqi > 200).astype(int)
    df = pd.get_dummies(df, columns=["region", "event_name"], prefix=["region", "event"], dtype=int)
    for col in FEATURES:
        if col not in df:
            df[col] = 0
    return df


def main():
    df = build_dataset(read_cube(get_engine()))
    train, test = df[df.year <= 2018], df[df.year >= 2019]
    print(f"Train {len(train):,} rows | test {len(test):,} rows | "
          f"unhealthy share {df.target_unhealthy.mean():.1%}")

    models = {
        "decision_tree": DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                class_weight="balanced", n_jobs=-1, random_state=42),
    }
    OUTPUTS.mkdir(exist_ok=True)
    report = []
    for name, model in models.items():
        pipe = make_pipeline(SimpleImputer(strategy="median"), model)
        pipe.fit(train[FEATURES], train.target_unhealthy)
        prob = pipe.predict_proba(test[FEATURES])[:, 1]
        text = classification_report(test.target_unhealthy, prob >= 0.5,
                                     target_names=["Healthy", "Unhealthy"], digits=3)
        report.append(f"== {name} ==  ROC-AUC {roc_auc_score(test.target_unhealthy, prob):.3f}\n{text}")
        joblib.dump(pipe, OUTPUTS / f"classifier_{name}.joblib")

    tree = models["decision_tree"]
    report.append("== Decision tree rules ==\n" + export_text(tree, feature_names=FEATURES))
    pd.Series(models["random_forest"].feature_importances_, index=FEATURES) \
        .sort_values(ascending=False).to_csv(OUTPUTS / "feature_importance.csv", header=["importance"])
    (OUTPUTS / "classification_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report[:2]))


if __name__ == "__main__":
    main()
