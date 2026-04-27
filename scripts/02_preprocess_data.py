import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

print("=" * 50)
print("PREPROCESSING COLD CHAIN DATA")
print("=" * 50)

data = []
with open("data/raw/coldchain_data.ndjson", "r") as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)
print(f"Loaded {len(df)} records")

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['temp_rate_change'] = df['temperature_celsius'].diff().fillna(0)
df['temp_rolling_mean'] = df['temperature_celsius'].rolling(12, min_periods=1).mean()
df['temp_rolling_std'] = df['temperature_celsius'].rolling(12, min_periods=1).std()
df['door_open_count'] = df['door_open'].rolling(12, min_periods=1).sum()

feature_cols = ['temperature_celsius', 'humidity_percent', 'battery_percent',
                'temp_rate_change', 'temp_rolling_mean', 'temp_rolling_std',
                'door_open_count', 'hour', 'day_of_week']

X = df[feature_cols].fillna(0).values
y = df['has_failure'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

np.savez("data/processed/xgboost_data.npz",
         X_train=X_train, X_val=X_val, X_test=X_test,
         y_train=y_train, y_val=y_val, y_test=y_test)

scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, "models/scaler.pkl")

import pickle
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

print(f"\nData splits created:")
print(f"  Train: {len(X_train)} samples")
print(f"  Validation: {len(X_val)} samples")
print(f"  Test: {len(X_test)} samples")