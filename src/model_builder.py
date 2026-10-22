def tensorflow_available():
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False

def build_model(model_type="LSTM", lookback=60, n_features=1):
    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout, Input
        from tensorflow.keras.optimizers import Adam
    except Exception as exc:
        raise RuntimeError("TensorFlow could not be imported. Use Python 3.11 and follow README.md setup instructions.") from exc

    model = Sequential()
    model.add(Input(shape=(lookback, n_features)))
    if model_type.upper() == "LSTM":
        model.add(LSTM(64, return_sequences=False))
    else:
        model.add(SimpleRNN(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(n_features))
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model
