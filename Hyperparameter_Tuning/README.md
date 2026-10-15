# Hyperparameter Tuning: Paddy Yield Prediction

Gradient Boosting Regressor tuned with **GridSearchCV** and **RandomizedSearchCV** (5-fold CV, scoring = R2).
Target: yield per hectare (kg/ha). Train / test split 80 / 20 (`random_state=42`).

| Method | Test R2 | Test MAE (kg/ha) | Test RMSE | CV R2 |
|---|---|---|---|---|
| Default | 0.500 | 154.9 | 199.0 | - |
| GridSearchCV (54 combinations) | 0.512 | 152.7 | 196.7 | 0.538 |
| RandomizedSearchCV (40 trials) | 0.511 | 153.1 | 197.0 | 0.538 |

Best GridSearchCV parameters: `learning_rate=0.05, max_depth=2, n_estimators=100, subsample=0.8`

Files: `hyperparameter_tuning.py`, `paddydataset.csv`, `tuning_results.csv`, `best_parameters.csv`,
`gridsearch_all_trials.csv`, `randomsearch_all_trials.csv`, `tuning_comparison.png`, `tuned_yield_model.pkl`

Run: `pip install pandas scikit-learn scipy matplotlib joblib` then `python hyperparameter_tuning.py`
