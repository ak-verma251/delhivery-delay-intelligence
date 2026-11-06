"""Part C - flag the parcels that will miss their promise, and pick the alert threshold.

Run from the repository root:  python part_c_classification.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS, SEED, save_table

table = pd.read_csv("data/modelling_table.csv")
X, y = table[FEATURES], table["is_delayed"]


def cells(truth, predicted):
    """Confusion-matrix cells plus the four usual scores, all worked out from those cells."""
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": (tn + tp) / (tn + fp + fn + tp), "precision": precision, "recall": recall, "f1": f1}


def flag(model, features, threshold=0.5):
    """1 = 'delayed' when the delay probability is AT or ABOVE the threshold.
    The same rule is used for every model and every table in this script. (scikit-learn's built-in
    predict() sends an exact 50-50 tie to 'on time', which would make the 0.5 row of the threshold
    table disagree with the model tables.)"""
    return (model.predict_proba(features)[:, 1] >= threshold).astype(int)


# ---- 1. Stratified split (keeps the share of delayed parcels equal in all three parts) ---
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)
assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)
assert (y_train.sum(), y_val.sum(), y_test.sum()) == (195, 42, 41)

# ---- 2. Blind baseline and three classifiers (default 0.5 rule) --------------------------
model_rows = [{"model": "blind_all_on_time", **cells(y_val, np.zeros(len(y_val), dtype=int))}]

logistic = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
model_rows.append({"model": "logistic_regression", **cells(y_val, flag(logistic, X_val))})

best_depth, best_f1 = None, -1.0
for depth in [2, 3, 4, 5, 6, 8, 10]:                       # ascending, strict ">" keeps the smaller depth on ties
    tree = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=SEED).fit(X_train, y_train)
    f1 = cells(y_val, flag(tree, X_val))["f1"]
    if f1 > best_f1:
        best_depth, best_f1, best_tree = depth, f1, tree
model_rows.append({"model": f"decision_tree_depth_{best_depth}", **cells(y_val, flag(best_tree, X_val))})

forest = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(X_train, y_train)
model_rows.append({"model": "random_forest_default", **cells(y_val, flag(forest, X_val))})
save_table(pd.DataFrame(model_rows), "outputs/part_c_models.csv")

# ---- 3. Which columns carry signal? -------------------------------------------------------
f_scores = SelectKBest(f_classif, k="all").fit(X_train, y_train).scores_
scores_table = pd.DataFrame({"feature": FEATURES, "f_score": f_scores,
                             "rf_importance": forest.feature_importances_})
scores_table["f_rank"] = scores_table["f_score"].rank(ascending=False, method="first").astype(int)
scores_table["is_noise"] = scores_table["feature"].isin(NOISE_COLUMNS)
scores_table = scores_table.sort_values("f_rank")[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
save_table(scores_table, "outputs/part_c_feature_scores.csv")

selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                     ("model", RandomForestClassifier(n_estimators=100, random_state=SEED))]).fit(X_train, y_train)
    selectk_rows.append({"k": k, "val_f1": cells(y_val, flag(pipe, X_val))["f1"]})
save_table(pd.DataFrame(selectk_rows), "outputs/part_c_selectk.csv")

# ---- 4. Tuning on the 7 best columns (selector learned from training rows only) -----------
K = 7
selector = SelectKBest(f_classif, k=K).fit(X_train, y_train)
Xt_train, Xt_val = selector.transform(X_train), selector.transform(X_val)

grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
searches = {
    "grid_search": GridSearchCV(RandomForestClassifier(random_state=SEED), grid, cv=5),
    "randomized_search": RandomizedSearchCV(RandomForestClassifier(random_state=SEED), grid,
                                            n_iter=9, cv=5, random_state=SEED),
}
tuning_rows = []
chosen_params = {}
for name, search in searches.items():
    search.fit(Xt_train, y_train)
    candidates = len(search.cv_results_["params"])
    chosen_params[name] = dict(search.best_params_)
    tuning_rows.append({
        "search": name, "candidates": candidates, "cv_folds": 5, "total_fits": candidates * 5,
        "best_params": str(dict(search.best_params_)), "best_cv_accuracy": search.best_score_,
        "val_f1": cells(y_val, flag(search.best_estimator_, Xt_val))["f1"],
    })

default_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
untuned = RandomForestClassifier(random_state=SEED).fit(Xt_train, y_train)
chosen_params["default_forest_same_7_features"] = default_params
tuning_rows.append({
    "search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
    "best_params": str(default_params), "best_cv_accuracy": np.nan,
    "val_f1": cells(y_val, flag(untuned, Xt_val))["f1"],
})
tuning = pd.DataFrame(tuning_rows)
save_table(tuning, "outputs/part_c_tuning.csv")

# ---- 5. Choose the final configuration: highest validation F1; ties -> default, grid, randomized
priority = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
rounded_f1 = tuning["val_f1"].round(4)                      # compare the numbers exactly as written in the CSV
winner = min(range(len(tuning)), key=lambda i: (-rounded_f1[i], priority[tuning["search"][i]]))
chosen_from = tuning["search"][winner]
params = chosen_params[chosen_from]

final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=K)),
    ("model", RandomForestClassifier(random_state=SEED, **params)),
]).fit(X_train, y_train)

# ---- 6. Alert threshold from the operations rule (validation rows only) -------------------
val_proba = final_model.predict_proba(X_val)[:, 1]
threshold_rows = []
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:     # exact one-decimal values, not np.arange
    threshold_rows.append({"threshold": t, **cells(y_val, (val_proba >= t).astype(int))})
thresholds = pd.DataFrame(threshold_rows)
save_table(thresholds, "outputs/part_c_thresholds.csv")

# Highest threshold that still catches at least 80% of the delayed parcels; if none does, use 0.1
enough_recall = thresholds[thresholds["recall"] >= 0.80]
chosen_threshold = float(enough_recall["threshold"].max()) if len(enough_recall) else 0.1

# ---- 7. One look at the test rows -----------------------------------------------------------
test_proba = final_model.predict_proba(X_test)[:, 1]
final_test = {"threshold": chosen_threshold, **cells(y_test, (test_proba >= chosen_threshold).astype(int))}
save_table(pd.DataFrame([final_test]), "outputs/part_c_final_test.csv")

save_table(pd.DataFrame([{
    "selector_k": K, "n_estimators": params["n_estimators"],
    "max_depth": "unlimited" if params["max_depth"] is None else params["max_depth"],
    "min_samples_leaf": params["min_samples_leaf"],
    "threshold": chosen_threshold, "chosen_from": chosen_from,
}]), "outputs/part_c_final_config.csv")

print("chosen configuration:", chosen_from, params, "| threshold:", chosen_threshold)
