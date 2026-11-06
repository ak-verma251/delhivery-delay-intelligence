"""Shared building blocks used by every script in this repository.

Keeping the cleaning and feature code in ONE place means the model is trained and
used on parcels that were prepared in exactly the same way.
"""
import numpy as np
import pandas as pd

# ---- Constants pinned by the brief -------------------------------------------------
PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

# ---- Small extras shared by the scripts --------------------------------------------
MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
SENTINEL_WEIGHT_KG = 100   # the network accepts parcels up to 40 kg, so anything above 100 is a scanner fault
SEED = 42                  # the random_state used everywhere in this project


def save_table(df, path):
    """Write a CSV without row numbers, with every decimal number rounded to 4 places."""
    df.round(4).to_csv(path, index=False)


# ---- Step 1: clean the raw export ---------------------------------------------------
def clean_shipments(raw, medians=None):
    """Turn a raw shipment export into a tidy table.

    Returns (clean_df, medians). If `medians` is None they are worked out from the data
    passed in; otherwise the dictionary passed in is used to fill the gaps.
    """
    df = raw.drop_duplicates().copy()                                   # 1. exact duplicates

    df["service_type"] = df["service_type"].str.strip().str.title()     # 2. Express / Standard / Economy

    df["payment_mode"] = df["payment_mode"].fillna("Unknown")           # 3. blank payment mode

    value_text = df["declared_value"].astype("string")                  # 4. "₹1,290" -> 1290.0
    value_text = value_text.str.replace("\u20b9", "", regex=False).str.replace(",", "", regex=False)
    df["declared_value_inr"] = pd.to_numeric(value_text, errors="coerce").astype("float64")
    df = df.drop(columns=["declared_value"])

    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], format="%Y-%m-%d %H:%M")   # 5. text -> date

    too_heavy = df["parcel_weight_kg"] > SENTINEL_WEIGHT_KG             # 6. scanner-fault weights
    df.loc[too_heavy, "parcel_weight_kg"] = np.nan

    if medians is None:                                                 # 7. fill gaps with the median
        medians = {c: float(df[c].median()) for c in MEDIAN_COLUMNS}
    else:
        medians = {c: float(medians[c]) for c in MEDIAN_COLUMNS}
    for c in MEDIAN_COLUMNS:
        df[c] = df[c].fillna(medians[c])

    return df.reset_index(drop=True), medians


# ---- Step 2: add the columns the models read ---------------------------------------
def build_features(clean, hubs):
    """Attach hub details and the new feature columns. Never looks at delivery_hours,
    so it also works for parcels that have not been delivered yet."""
    hub_columns = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    out = clean.merge(hubs[hub_columns], on="hub_id", how="left")
    if len(out) != len(clean):
        raise ValueError("The hub merge changed the number of rows - check hubs.csv for repeated hub_id values.")

    out["promised_hours"] = out["service_type"].map(PROMISED_HOURS).astype(int)
    out["is_cod"] = (out["payment_mode"] == "COD").astype(int)           # "Unknown" counts as 0
    out["is_weekend"] = (out["dispatch_time"].dt.dayofweek >= 5).astype(int)
    out["dispatch_hour"] = out["dispatch_time"].dt.hour.astype(int)
    out["log_declared_value"] = np.log(out["declared_value_inr"])
    return out
