import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual).ravel()
    predicted = np.asarray(predicted).ravel()
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
    }

def naive_predictions(values):
    values = np.asarray(values).ravel()
    return values[:-1]

def directional_accuracy(actual, predicted):
    actual = np.asarray(actual).ravel()
    predicted = np.asarray(predicted).ravel()
    if len(actual) < 2 or len(predicted) != len(actual):
        return 0.0
    actual_direction = np.diff(actual) >= 0
    predicted_direction = np.diff(predicted) >= 0
    return float(np.mean(actual_direction == predicted_direction) * 100)
