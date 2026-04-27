import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error
import math

np.random.seed(42)
tf.random.set_seed(42)

print("=" * 50)
print("TRAINING LSTM TEMPERATURE FORECASTING MODEL")
print("=" * 50)

data = []
with open("data/raw/coldchain_data.ndjson", "r") as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
temps = df['temperature_celsius'].values
print(f"Loaded {len(temps)} temperature readings")

SEQ_LENGTH = 12
FORECAST_HORIZON = 12

def create_sequences(data, seq_length, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length+forecast_horizon])
    return np.array(X), np.array(y)

X, y = create_sequences(temps, SEQ_LENGTH, FORECAST_HORIZON)
X = X.reshape(X.shape[0], X.shape[1], 1)

train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]
X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=1)

y_pred = model.predict(X_test, verbose=0)
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(((y_test - y_pred.flatten()) ** 2).mean())

print(f"\nLSTM FORECASTING RESULTS:")
print(f"  MAE: {mae:.3f}°C")
print(f"  RMSE: {rmse:.3f}°C")

if mae < 1.0:
    print(f"  MAE < 1.0°C target ACHIEVED")
else:
    print(f"  MAE < 1.0°C target NOT YET ACHIEVED")

model.save("models/lstm_forecast_model.h5")

import json
metrics = {
    "mae": float(mae),
    "rmse": float(rmse),
    "forecast_horizon": f"{FORECAST_HORIZON * 5} minutes"
}
with open("models/lstm_forecast_metrics.json", "w") as f:
    json.dump(metrics, f)

print("\nModel saved to: models/lstm_forecast_model.h5")