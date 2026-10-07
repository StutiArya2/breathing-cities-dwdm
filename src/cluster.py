"""Clustering: group cities into 'pollution personalities'.

Each city is profiled by its median pollutant mix plus how strongly PM2.5 swings
between winter and monsoon. K-Means is run for k = 2..6 and the k with the best
silhouette score is kept.
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from db import OUTPUTS, get_engine, read_cube

PROFILE = ["pm25", "pm10", "no2", "co", "so2", "o3", "nh3"]


def city_profiles(cube):
    prof = cube.groupby("city")[PROFILE].median()
    seasonal = cube.pivot_table(index="city", columns="season", values="pm25", aggfunc="mean")
    prof["winter_monsoon_ratio"] = seasonal["Winter"] / seasonal["Monsoon"]
    prof["unhealthy_day_share"] = cube.assign(u=cube.aqi > 200).groupby("city")["u"].mean()
    return prof.fillna(prof.median())


def main():
    prof = city_profiles(read_cube(get_engine()))
    X = StandardScaler().fit_transform(prof)

    scores = {}
    for k in range(2, min(7, len(prof))):
        labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    best_k = max(scores, key=scores.get)
    print("Silhouette by k:", {k: round(v, 3) for k, v in scores.items()}, "-> k =", best_k)

    km = KMeans(n_clusters=best_k, n_init=20, random_state=42).fit(X)
    prof["cluster"] = km.labels_
    prof[["pc1", "pc2"]] = PCA(n_components=2, random_state=42).fit_transform(X)
    OUTPUTS.mkdir(exist_ok=True)
    prof.round(3).to_csv(OUTPUTS / "city_clusters.csv")

    summary = prof.groupby("cluster")[PROFILE + ["winter_monsoon_ratio", "unhealthy_day_share"]].mean()
    summary["cities"] = prof.reset_index().groupby("cluster")["city"].apply(", ".join)
    summary.round(2).to_csv(OUTPUTS / "cluster_summary.csv")
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
