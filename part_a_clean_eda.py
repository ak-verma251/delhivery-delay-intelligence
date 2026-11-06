"""Part A - clean the raw export, build the modelling table and write the first evidence about delays.

Run from the repository root:  python part_a_clean_eda.py
"""
import pandas as pd

from src.shipment_prep import FEATURES, PROMISED_HOURS, build_features, clean_shipments, save_table

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# ---- Clean, add features, add the target ------------------------------------------
clean, medians = clean_shipments(raw)
table = build_features(clean, hubs)
table["is_delayed"] = (table["delivery_hours"] > table["service_type"].map(PROMISED_HOURS)).astype(int)

# The modelling table is written at full precision so that Parts B-D see exactly what Part A built.
table.to_csv("data/modelling_table.csv", index=False)

# ---- Checks file: every number is computed here, none is typed in ------------------
before = raw.drop_duplicates()                      # "before" counts: after duplicates, before any filling
q1, q3 = table["delivery_hours"].quantile(0.25), table["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr       # bounds from the unrounded quartiles
outliers = ((table["delivery_hours"] < lower) | (table["delivery_hours"] > upper)).sum()

checked_columns = FEATURES + ["delivery_hours", "is_delayed"]
metrics = [
    ("raw_rows", len(raw)),
    ("duplicate_rows_removed", len(raw) - len(before)),
    ("rows_after_dedup", len(before)),
    ("missing_distance_km_before", before["distance_km"].isna().sum()),
    ("missing_parcel_weight_before", before["parcel_weight_kg"].isna().sum()),
    ("sentinel_weight_rows", (before["parcel_weight_kg"] > 100).sum()),
    ("missing_declared_value_before", before["declared_value"].isna().sum()),
    ("missing_payment_mode_before", before["payment_mode"].isna().sum()),
    ("service_type_spellings_before", before["service_type"].nunique()),
    ("service_type_spellings_after", clean["service_type"].nunique()),
    ("declared_value_mean_inr", table["declared_value_inr"].mean()),
    ("declared_value_median_inr", table["declared_value_inr"].median()),
    ("log_declared_value_mean", table["log_declared_value"].mean()),
    ("log_declared_value_median", table["log_declared_value"].median()),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", len(table)),
    ("nulls_in_features_and_target", int(table[checked_columns].isna().sum().sum())),
    ("delayed_count", table["is_delayed"].sum()),
    ("delay_rate", table["is_delayed"].mean()),
    ("hours_q1", q1),
    ("hours_q3", q3),
    ("hours_lower_bound", lower),
    ("hours_upper_bound", upper),
    ("hours_outlier_count", outliers),
]
def tidy(v):
    """Whole numbers stay whole; decimals are rounded to 4 places."""
    v = v.item() if hasattr(v, "item") else v
    return round(v, 4) if isinstance(v, float) else v


checks = pd.DataFrame({
    "metric": [name for name, _ in metrics],
    # dtype=object keeps whole numbers as 1224 instead of 1224.0 in a column that also holds decimals
    "value": pd.Series([tidy(v) for _, v in metrics], dtype=object),
})
checks.to_csv("outputs/part_a_checks.csv", index=False)

# ---- Delay evidence by service type, hub tier and distance band -------------------
table["distance_band"] = pd.cut(table["distance_km"], bins=[0, 200, 400, 600, 2000],
                                labels=["0-200", "201-400", "401-600", "601+"])
rows = []
for dimension in ["service_type", "hub_tier", "distance_band"]:
    grouped = table.groupby(dimension, observed=True)
    summary = pd.DataFrame({
        "shipments": grouped.size(),
        "delayed": grouped["is_delayed"].sum(),
        "delay_rate": grouped["is_delayed"].mean(),
        "mean_hours": grouped["delivery_hours"].mean(),
    }).reset_index().rename(columns={dimension: "group"})
    summary.insert(0, "dimension", dimension)
    summary["group"] = summary["group"].astype(str)
    rows.append(summary)
save_table(pd.concat(rows, ignore_index=True), "outputs/part_a_delay_by_group.csv")

# ---- Correlation matrix of the numeric columns -------------------------------------
numeric = table.drop(columns=["distance_band"]).select_dtypes("number")
numeric.corr().round(4).to_csv("outputs/part_a_correlations.csv", index_label="column")

print("rows in modelling table:", len(table), "| delayed:", int(table["is_delayed"].sum()))
