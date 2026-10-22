import numpy as np

def train_model(model, X_train, y_train, X_val, y_val):
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, min_delta=1e-5),
        ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-5, verbose=0),
    ]
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=50, batch_size=32, shuffle=False, callbacks=callbacks, verbose=0
    )
    return model, history

def recursive_forecast(model, train_scaled, val_scaled, test_scaled, lookback, horizon):
    combined = np.vstack([train_scaled, val_scaled, test_scaled])
    window = combined[-lookback:].copy()
    predictions = []
    for _ in range(horizon):
        pred = model.predict(window.reshape(1, lookback, combined.shape[1]), verbose=0)[0]
        predictions.append(float(pred[0]))
        window = np.vstack([window[1:], pred.reshape(1, -1)])
    return np.asarray(predictions)
