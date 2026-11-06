"""Part B - estimate delivery_hours for every parcel and decide which model to trust.

Run from the repository root:  python part_b_regression.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import FEATURES, SEED, save_table

table = pd.read_csv("data/modelling_table.csv")
X, y = table[FEATURES], table["delivery_hours"]

# ---- 1. Split: 70% train, 15% validation, 15% test ---------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)
assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)

# ---- 2. Scaling without leakage: learn the scale from the training rows ONLY --------
scaler = StandardScaler().fit(X_train)
train_scaled = scaler.transform(X_train)
test_scaled = scaler.transform(X_test)          # test rows are only transformed, never learned from
save_table(pd.DataFrame({
    "metric": ["train_rows", "val_rows", "test_rows", "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
    "value": pd.Series([len(X_train), len(X_val), len(X_test),
                        round(float(np.abs(train_scaled.mean(axis=0)).max()), 4),
                        round(float(np.abs(test_scaled.mean(axis=0)).max()), 4)], dtype=object),
}), "outputs/part_b_checks.csv")


def scores(model_name, split, truth, predicted):
    mse = mean_squared_error(truth, predicted)
    return {"model": model_name, "split": split, "MAE": mean_absolute_error(truth, predicted),
            "MSE": mse, "RMSE": float(np.sqrt(mse)), "R2": r2_score(truth, predicted)}


# ---- 3. Baseline: always guess the average delivery time of the training rows -------
metric_rows = [scores("mean_baseline", "val", y_val, np.full(len(y_val), y_train.mean()))]

# ---- 4. Tree-depth diagnostic -------------------------------------------------------
depth_rows = []
for depth in [1, 2, 3, 4, 6, 8, 12, None]:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=SEED).fit(X_train, y_train)
    depth_rows.append({
        "max_depth": "unlimited" if depth is None else depth,
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, tree.predict(X_train)))),
        "val_rmse": float(np.sqrt(mean_squared_error(y_val, tree.predict(X_val)))),
        "_depth": depth,
    })
depth_table = pd.DataFrame(depth_rows)
best_depth = None
best_rmse = np.inf
for row in depth_rows:                      # ascending depth, "unlimited" last, so ties keep the smaller depth
    if row["val_rmse"] < best_rmse:
        best_rmse, best_depth = row["val_rmse"], row["_depth"]
save_table(depth_table.drop(columns="_depth"), "outputs/part_b_tree_depth.csv")

# ---- 5. Four candidate models, fitted on training rows, scored on validation rows ---
candidates = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=SEED),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=SEED),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=SEED),
}
for name, model in candidates.items():
    model.fit(X_train, y_train)
    metric_rows.append(scores(name, "val", y_val, model.predict(X_val)))

# ---- 6. Learning-rate experiment for gradient boosting ------------------------------
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=SEED).fit(X_train, y_train)
    pred = gb.predict(X_val)
    lr_rows.append({"learning_rate": lr, "val_rmse": float(np.sqrt(mean_squared_error(y_val, pred))),
                    "val_r2": r2_score(y_val, pred)})
save_table(pd.DataFrame(lr_rows), "outputs/part_b_gb_learning_rate.csv")

# ---- 7. Rank by validation RMSE, cross-check the best two, score the winner once on test
val_rmse = {r["model"]: r["RMSE"] for r in metric_rows if r["model"] in candidates}
ranking = sorted(val_rmse, key=val_rmse.get)
top_two = ranking[:2]

X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])
folds = KFold(n_splits=5, shuffle=True, random_state=SEED)     # one object, shared by both models
cv_rows = []
for name in top_two:
    r2s = cross_val_score(candidates[name], X_trainval, y_trainval, cv=folds)
    cv_rows.append({"model": name, "mean_r2": r2s.mean(), "std_r2": r2s.std()})
save_table(pd.DataFrame(cv_rows), "outputs/part_b_cv.csv")

final_name = ranking[0]
metric_rows.append(scores(final_name, "test", y_test, candidates[final_name].predict(X_test)))   # the one test look
save_table(pd.DataFrame(metric_rows), "outputs/part_b_metrics.csv")

print("validation ranking:", ranking)
print("final model:", final_name, "| tree depth used:", best_depth)
