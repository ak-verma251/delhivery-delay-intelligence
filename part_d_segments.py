"""Part D (first half) - group the 24 hubs into three operating segments and check the tier labels.

Run from the repository root:  python part_d_segments.py
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from src.shipment_prep import SEED, save_table

table = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

# ---- 1. Hub profile: how each hub really behaves (sorted by hub_id, HUB01 first) -------
profile = table.groupby("hub_id").agg(
    shipments=("shipment_id", "count"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean"),
).reset_index().sort_values("hub_id").reset_index(drop=True)
save_table(profile, "outputs/hub_profile.csv")

# ---- 2. K-Means on the four behaviour columns (scaled so no column dominates) ----------
PROFILE_COLUMNS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
scaled = StandardScaler().fit_transform(profile[PROFILE_COLUMNS])


def cluster(k):
    return KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(scaled)


# ---- 3. Separation evidence for K = 2..5 ------------------------------------------------
distances = pairwise_distances(scaled)
first, second = np.triu_indices(len(scaled), k=1)          # all 276 pairs of hubs
pair_distance = distances[first, second]

separation_rows = []
for k in [2, 3, 4, 5]:
    labels = cluster(k)
    same_cluster = labels[first] == labels[second]
    sizes = sorted(np.bincount(labels), reverse=True)
    separation_rows.append({
        "K": k,
        "across_over_within": pair_distance[~same_cluster].mean() / pair_distance[same_cluster].mean(),
        "cluster_sizes": "-".join(str(int(s)) for s in sizes),
    })
save_table(pd.DataFrame(separation_rows), "outputs/part_d_separation.csv")

# ---- 4. Name the three clusters by how often their hubs are late ------------------------
labels = cluster(3)                                         # K = 3 is fixed by the business
profile["cluster"] = labels
order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index   # lowest delay rate first
names = dict(zip(order, ["Standard", "Watch", "Escalate"]))
profile["segment"] = profile["cluster"].map(names)

segments = profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
save_table(segments, "outputs/hub_segments.csv")

segment_profile = profile.groupby("segment").agg(
    hubs=("hub_id", "count"),
    mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean"),
).reindex(["Standard", "Watch", "Escalate"]).reset_index()
save_table(segment_profile, "outputs/segment_profile.csv")

# ---- 5. Which hubs behave unlike their tier label? ---------------------------------------
tier_implies = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatch = segments[segments["segment"] != segments["hub_tier"].map(tier_implies)]
save_table(mismatch, "outputs/tier_vs_segment_mismatch.csv")

print(segment_profile.round(2).to_string(index=False))
print("mismatched hubs:", ", ".join(mismatch["hub_id"]))
