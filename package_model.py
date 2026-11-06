"""Part D (second half) - rebuild the final delay classifier, save it, and prove the saved copy works.

Run from the repository root:  python package_model.py
This regenerates everything in models/. The saved pickle only loads reliably on the scikit-learn
version pinned in requirements.txt, so run this script again if you use a different version.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, SEED, clean_shipments, save_table

table = pd.read_csv("data/modelling_table.csv")
X, y = table[FEATURES], table["is_delayed"]

# ---- 1. Read the configuration Part C chose -----------------------------------------------
config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
max_depth = None if str(config["max_depth"]) == "unlimited" else int(config["max_depth"])
threshold = float(config["threshold"])

# ---- 2. Same stratified split as Part C, then fit the final pipeline on the training rows ----
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

model = Pipeline([
    ("selector", SelectKBest(f_classif, k=int(config["selector_k"]))),
    ("model", RandomForestClassifier(
        random_state=SEED, n_estimators=int(config["n_estimators"]),
        max_depth=max_depth, min_samples_leaf=int(config["min_samples_leaf"]))),
]).fit(X_train, y_train)

# ---- 3. Save the four files the scoring script needs ---------------------------------------
joblib.dump(model, "models/delay_model.pkl")
save_table(pd.DataFrame({"threshold": [threshold]}), "models/delay_threshold.csv")
save_table(pd.DataFrame({"feature": FEATURES}), "models/feature_order.csv")

_, medians = clean_shipments(pd.read_csv("data/raw_shipments.csv"))   # fill-in values for gaps in new parcels
save_table(pd.DataFrame({"column": list(medians), "median": list(medians.values())}), "models/impute_values.csv")

# ---- 4. Reload the saved file and check it gives the same answers ---------------------------
reloaded = joblib.load("models/delay_model.pkl")
in_memory = model.predict_proba(X_test)[:, 1]
from_file = reloaded.predict_proba(X_test)[:, 1]
largest_difference = float(np.abs(in_memory - from_file).max())

check = pd.DataFrame({
    "metric": ["test_rows_scored", "reload_max_abs_probability_diff", "threshold"],
    "value": pd.Series([len(X_test), round(largest_difference, 4), threshold], dtype=object),
})
check.to_csv("outputs/part_d_reload_check.csv", index=False)

print("saved models/; test rows scored:", len(X_test), "| largest difference after reload:", largest_difference)
