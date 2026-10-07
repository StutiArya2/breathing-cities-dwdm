"""Classification: will TOMORROW be an unhealthy-air day (AQI > 200) in this city?

Four models, compared on a chronological split (train <= 2018, test 2019-2020):
  persistence     tomorrow = today (the baseline any model has to beat)
  tree_pollution  Decision Tree on today's pollution + calendar/context
  rf_pollution    Random Forest on the same features
  rf_weather      Random Forest + weather today and tomorrow (from the fact_weather source).
                  Tomorrow's weather stands in for a 1-day weather forecast, which is
                  available in practice; tomorrow's pollution is never used.
SHAP values explain what drives the rf_weather predictions.
"""
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from db import OUTPUTS, get_engine, read_cube

POLLUTION = ["pm25", "pm10", "no2", "nox", "nh3", "co", "so2", "o3", "aqi", "aqi_3day_mean",
             "aqi_change", "month", "is_weekend", "is_coastal", "region_North", "region_East",
             "event_Stubble burning", "event_Diwali", "event_COVID lockdown"]
WEATHER = ["temp_mean", "humidity", "precipitation", "wind_mean", "mixing_height"]
NEXT_WEATHER = [f"next_{w}" for w in WEATHER]
WITH_WEATHER = POLLUTION + WEATHER + NEXT_WEATHER


def build_dataset(cube):
    df = cube.copy()
    g = df.groupby("city")
    df["aqi_3day_mean"] = g["aqi"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["aqi_change"] = df["aqi"] - g["aqi"].shift(1)
    df["next_date"] = g["full_date"].shift(-1)
    df["next_aqi"] = g["aqi"].shift(-1)
    for w in WEATHER:
        df[f"next_{w}"] = g[w].shift(-1)
    df = df[(df.next_date - df.full_date).dt.days.eq(1) & df.next_aqi.notna() & df.aqi.notna()].copy()
    df["target_unhealthy"] = (df.next_aqi > 200).astype(int)
    df = pd.get_dummies(df, columns=["region", "event_name"], prefix=["region", "event"], dtype=int)
    for col in WITH_WEATHER:
        if col not in df:
            df[col] = 0
    return df


def scores(y, prob, pred, flips):
    """flips = test days whose status changes tomorrow; persistence always gets these wrong."""
    return {"precision": precision_score(y, pred), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, prob),
            "flip_day_accuracy": float((pred[flips] == y[flips]).mean())}


def main():
    df = build_dataset(read_cube(get_engine()))
    train, test = df[df.year <= 2018], df[df.year >= 2019]
    y_train, y_test = train.target_unhealthy, test.target_unhealthy
    print(f"Train {len(train):,} | test {len(test):,} | unhealthy share {df.target_unhealthy.mean():.1%}")

    flips = ((test.aqi > 200).astype(int) != y_test).to_numpy()
    y_arr = y_test.to_numpy()
    print(f"Status changes on {flips.mean():.1%} of test days")
    results = {"persistence": scores(y_arr, test.aqi / 500, (test.aqi > 200).astype(int).to_numpy(), flips)}
    models = {
        "tree_pollution": (DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42),
                           POLLUTION),
        "rf_pollution": (RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                                n_jobs=-1, random_state=42), POLLUTION),
        "rf_weather": (RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                              n_jobs=-1, random_state=42), WITH_WEATHER),
    }
    OUTPUTS.mkdir(exist_ok=True)
    fitted = {}
    for name, (model, feats) in models.items():
        pipe = make_pipeline(SimpleImputer(strategy="median"), model).fit(train[feats], y_train)
        prob = pipe.predict_proba(test[feats])[:, 1]
        results[name] = scores(y_arr, prob, (prob >= 0.5).astype(int), flips)
        fitted[name] = pipe
        joblib.dump(pipe, OUTPUTS / f"classifier_{name}.joblib")

    table = pd.DataFrame(results).T.round(3)
    table.to_csv(OUTPUTS / "classification_scores.csv", index_label="model")
    print(table.to_string())

    tree = models["tree_pollution"][0]
    (OUTPUTS / "decision_tree_rules.txt").write_text(export_text(tree, feature_names=POLLUTION), encoding="utf-8")

    # SHAP: mean |contribution| of each feature to the rf_weather prediction
    pipe = fitted["rf_weather"]
    sample = test.sample(1500, random_state=42)
    X = pd.DataFrame(pipe[0].transform(sample[WITH_WEATHER]), columns=WITH_WEATHER)
    sv = shap.TreeExplainer(pipe[-1]).shap_values(X)
    sv = sv[:, :, 1] if np.ndim(sv) == 3 else sv[1]
    importance = pd.Series(np.abs(sv).mean(axis=0), index=WITH_WEATHER).sort_values(ascending=False)
    importance.round(4).to_csv(OUTPUTS / "shap_importance.csv", header=["mean_abs_shap"])
    direction = pd.Series([np.corrcoef(X[c], sv[:, i])[0, 1] if X[c].std() > 0 and sv[:, i].std() > 0 else 0
                           for i, c in enumerate(WITH_WEATHER)], index=WITH_WEATHER)
    direction.round(3).to_csv(OUTPUTS / "shap_direction.csv", header=["corr_value_vs_shap"])
    print("\nTop SHAP features:\n", importance.head(10).round(4).to_string())


if __name__ == "__main__":
    main()
