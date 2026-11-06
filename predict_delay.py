"""Turn the batch of new parcels into an action list for the hub teams.

Run from the repository root:  python predict_delay.py
Needs the files written by part_d_segments.py and package_model.py.
"""
import joblib
import pandas as pd

from src.shipment_prep import build_features, clean_shipments

# ---- Load everything this script needs ------------------------------------------------------
new_parcels = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
segments = pd.read_csv("outputs/hub_segments.csv")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
impute_values = pd.read_csv("models/impute_values.csv")
medians = dict(zip(impute_values["column"], impute_values["median"]))
try:
    model = joblib.load("models/delay_model.pkl")
except Exception as error:      # most often: the pickle was written by a different scikit-learn version
    raise SystemExit(f"Could not load models/delay_model.pkl ({error}).\n"
                     "Run  python package_model.py  to rebuild it with the installed scikit-learn.")

# ---- Prepare the parcels exactly the way the training data was prepared ----------------------
clean, _ = clean_shipments(new_parcels, medians)       # same cleaning code, stored fill-in values
features = build_features(clean, hubs)
inputs = features[feature_order]
assert inputs.isna().sum().sum() == 0, "A feature column still has gaps after cleaning."

# ---- Score, flag and decide the action --------------------------------------------------------
result = features[["shipment_id", "hub_id"]].copy()
result["delay_probability"] = model.predict_proba(inputs)[:, 1].round(4)
# Flag on the rounded probability, so the file always agrees with its own numbers.
result["flagged"] = (result["delay_probability"] >= threshold).astype(int)
result = result.merge(segments[["hub_id", "segment"]], on="hub_id", how="left")


def decide(row):
    if row["flagged"] == 0:
        return "No action"
    return "Escalate to hub manager" if row["segment"] == "Escalate" else "Proactive customer call"


result["action"] = result.apply(decide, axis=1)
result = result[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
result.to_csv("outputs/new_shipment_actions.csv", index=False)

print(f"{len(result)} parcels scored | flagged: {int(result['flagged'].sum())} | "
      f"hub-manager escalations: {int((result['action'] == 'Escalate to hub manager').sum())}")
