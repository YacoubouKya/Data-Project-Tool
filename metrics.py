# metrics.py
# modules/utils/metrics.py
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import numpy as np
def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted"))
    }

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    #rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "r2": float(r2)}
