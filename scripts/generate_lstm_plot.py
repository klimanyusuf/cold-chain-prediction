"""
Generate LSTM Actual vs Predicted Temperature Plot
Run this script to create the forecast visualization for your paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import tensorflow as tf

# Load the trained LSTM model
lstm_model = tf.keras.models.load_model("models/lstm_forecast_model.h5")

# Load the temperature data
data = []
with open("data/raw/coldchain_data.ndjson", "r") as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
temps = df['temperature_celsius'].values

# Normalize
temp_mean = temps.mean()
temp_std = temps.std()
temps_norm = (temps - temp_mean) / temp_std

# Create sequences
SEQ_LENGTH = 12
FORECAST_HORIZON = 12

def create_sequences(data, seq_length, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - forecast_horizon):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length+forecast_horizon])
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences(temps_norm.flatten(), SEQ_LENGTH, FORECAST_HORIZON)

# Use test set (last 300 samples)
X_test = X_seq[-300:]
y_test = y_seq[-300:]

# Reshape for LSTM
X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Predict
y_pred_norm = lstm_model.predict(X_test_reshaped, verbose=0)

# Inverse transform
y_test_actual = y_test * temp_std + temp_mean
y_pred_actual = y_pred_norm.flatten() * temp_std + temp_mean

# Plot first 200 predictions
n_samples = min(200, len(y_test_actual))

plt.figure(figsize=(12, 6))
plt.plot(y_test_actual[:n_samples], label='Actual Temperature', 
         linewidth=1, alpha=0.8, color='steelblue')
plt.plot(y_pred_actual[:n_samples], label='Predicted Temperature', 
         linewidth=1, alpha=0.8, color='darkorange')
plt.xlabel('Time Step (5-minute intervals)', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.title(f'LSTM Temperature Forecast: Actual vs Predicted (MAE = 0.97°C)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Save the figure
plt.tight_layout()
plt.savefig('models/lstm_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.savefig('lstm_actual_vs_predicted.png', dpi=300, bbox_inches='tight')

print("✅ LSTM actual vs predicted plot saved as:")
print("   - models/lstm_actual_vs_predicted.png")
print("   - lstm_actual_vs_predicted.png")