"""
Hyperparameter tuning of a Gradient Boosting Regressor (paddy yield, kg/ha).

Three versions are compared on the same test set:
  1. Default parameters
  2. GridSearchCV        (exhaustive search over a small grid)
  3. RandomizedSearchCV  (random search over a wider space)
"""
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE = Path(__file__).resolve().parent
TARGET = "Paddy yield(in Kg)"

# ---------------- data ----------------
df = pd.read_csv(BASE / "paddydataset.csv")
df.columns = [c.strip() for c in df.columns]
y = df[TARGET] / df["Hectares"]                    # yield per hectare
X = df.drop(columns=[TARGET])
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def make_pipeline(**params):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
                            remainder="passthrough")
    return Pipeline([("pre", pre), ("model", GradientBoostingRegressor(random_state=42, **params))])


def test_scores(model):
    p = model.predict(X_test)
    return {"MAE": mean_absolute_error(y_test, p),
            "RMSE": mean_squared_error(y_test, p) ** 0.5,
            "R2": r2_score(y_test, p)}


cv = KFold(5, shuffle=True, random_state=42)
results, best_params, tuned_models = {}, {}, {}

# ---------------- 1. default ----------------
t0 = time.time()
default = make_pipeline().fit(X_train, y_train)
results["Default"] = {**test_scores(default), "CV_R2": np.nan, "Seconds": time.time() - t0}

# ---------------- 2. GridSearchCV ----------------
grid = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [2, 3, 4],
    "model__subsample": [0.8, 1.0],
}
t0 = time.time()
gs = GridSearchCV(make_pipeline(), grid, scoring="r2", cv=cv, n_jobs=-1, verbose=1).fit(X_train, y_train)
results["GridSearchCV"] = {**test_scores(gs.best_estimator_), "CV_R2": gs.best_score_, "Seconds": time.time() - t0}
best_params["GridSearchCV"] = gs.best_params_
tuned_models["GridSearchCV"] = gs
pd.DataFrame(gs.cv_results_).sort_values("rank_test_score")[
    ["rank_test_score", "mean_test_score", "std_test_score", "params"]
].to_csv(BASE / "gridsearch_all_trials.csv", index=False)

# ---------------- 3. RandomizedSearchCV ----------------
dist = {
    "model__n_estimators": randint(100, 600),
    "model__learning_rate": uniform(0.01, 0.19),
    "model__max_depth": randint(2, 7),
    "model__min_samples_split": randint(2, 20),
    "model__min_samples_leaf": randint(1, 15),
    "model__subsample": uniform(0.6, 0.4),
    "model__max_features": uniform(0.3, 0.7),
}
t0 = time.time()
rs = RandomizedSearchCV(make_pipeline(), dist, n_iter=40, scoring="r2", cv=cv,
                        n_jobs=-1, random_state=42, verbose=1).fit(X_train, y_train)
results["RandomizedSearchCV"] = {**test_scores(rs.best_estimator_), "CV_R2": rs.best_score_, "Seconds": time.time() - t0}
best_params["RandomizedSearchCV"] = rs.best_params_
tuned_models["RandomizedSearchCV"] = rs
pd.DataFrame(rs.cv_results_).sort_values("rank_test_score")[
    ["rank_test_score", "mean_test_score", "std_test_score", "params"]
].to_csv(BASE / "randomsearch_all_trials.csv", index=False)

# ---------------- save outputs ----------------
res = pd.DataFrame(results).T
res.round(4).to_csv(BASE / "tuning_results.csv", index_label="Method")

rows = []
for method, params in best_params.items():
    for k, v in params.items():
        rows.append({"Method": method, "Parameter": k.replace("model__", ""),
                     "Best value": round(float(v), 4)})
pd.DataFrame(rows).to_csv(BASE / "best_parameters.csv", index=False)

best_method = res.loc[["GridSearchCV", "RandomizedSearchCV"], "R2"].idxmax()
joblib.dump(tuned_models[best_method].best_estimator_, BASE / "tuned_yield_model.pkl")

print(res.round(3).to_string())
print("\nBest parameters:")
print(pd.DataFrame(rows).to_string(index=False))
print("\nTuned model saved from:", best_method)

# ---------------- chart ----------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
colors = ["#a9bcc9", "#5b8db8", "#2a6f97"]
for ax, metric, note in zip(axes, ["R2", "MAE"], ["higher is better", "lower is better"]):
    vals = res[metric].values
    ax.bar(res.index, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}" if metric == "R2" else f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Test {metric} ({note})")
    ax.tick_params(axis="x", labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylim(0, 0.6)
plt.suptitle("Gradient Boosting: default vs tuned")
plt.tight_layout()
plt.savefig(BASE / "tuning_comparison.png")
