import numpy as np
from sklearn.preprocessing import MinMaxScaler

def chronological_split(values, train_ratio=0.70, val_ratio=0.15):
    values = np.asarray(values, dtype=float)
    n = len(values)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    if train_end < 10 or val_end <= train_end or val_end >= n:
        raise ValueError("Not enough rows for a chronological train/validation/test split.")
    return {"train": values[:train_end], "validation": values[train_end:val_end], "test": values[val_end:]}

def scale_series(train, validation, test):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(validation)
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, val_scaled, test_scaled

def make_sequences(values, lookback=60):
    values = np.asarray(values, dtype=float)
    if len(values) <= lookback:
        return np.empty((0, lookback, values.shape[1])), np.empty((0, values.shape[1]))
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i-lookback:i])
        y.append(values[i])
    return np.asarray(X), np.asarray(y)
