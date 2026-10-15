# Paddy Yield Prediction using Machine Learning

Machine learning project that predicts paddy yield (kg/ha) from farm inputs, fertilizer and weather data.

**Live App:** https://paddy-yieldprediction-xqbhzptay9mbcbta4vnh3d.streamlit.app/

## Project structure
| Folder | Contents |
|---|---|
| `1_Dataset_Assigned` | Dataset (2,789 records, 44 features) and data summary |
| `2_Model_Implementation` | Preprocessing, Gradient Boosting training, saved model |
| `3_Performance_Evaluation_Metrics` | MAE, MSE, RMSE, R2, MAPE, actual vs predicted plot |
| `4_Model_Comparison` | 10 regression models compared |
| `5_Hyperparameter_Tuning` | GridSearchCV and RandomizedSearchCV tuning |
| `app.py`, `yield_model.pkl`, `requirements.txt` | Streamlit web app (deployed) |

## Results (test set, 558 records)
| Model | R2 | MAE (kg/ha) |
|---|---|---|
| Gradient Boosting (default) | 0.500 | 155 |
| Gradient Boosting (tuned) | 0.512 | 153 |

## Run order
pip install pandas scikit-learn matplotlib scipy joblib
python 2_Model_Implementation/model_implementation.py
python 3_Performance_Evaluation_Metrics/evaluate_model.py
python 4_Model_Comparison/model_comparison.py
python 5_Hyperparameter_Tuning/hyperparameter_tuning.py
