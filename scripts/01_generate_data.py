import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

print("=" * 50)
print("GENERATING SYNTHETIC COLD CHAIN DATA")
print("=" * 50)

records = []
start_time = datetime.now() - timedelta(days=7)
base_drift = 0

for i in range(2016):
    timestamp = start_time + timedelta(minutes=i*5)
    hour = timestamp.hour
    
    if i > 1000:
        base_drift = min(2.0, base_drift + 0.002)
    
    daily_variation = 1.5 * abs((hour - 14) / 12)
    noise = np.random.normal(0, 0.3)
    temp = 5.0 + base_drift + daily_variation + noise
    
    has_failure = 0
    if np.random.random() < 0.02:
        has_failure = 1
        temp = temp + np.random.uniform(2, 6)
        base_drift = base_drift + 0.01
    
    temp = max(-2, min(15, temp))
    
    door_open = 0
    if 8 <= hour <= 18:
        door_open = 1 if np.random.random() < 0.08 else 0
    else:
        door_open = 1 if np.random.random() < 0.02 else 0
    
    battery_drain = 0.01
    if door_open:
        battery_drain += 0.02
    if temp > 8:
        battery_drain += 0.01
    battery = max(0, min(100, 100 - i * battery_drain + np.random.normal(0, 2)))
    
    record = {
        "timestamp": timestamp.isoformat(),
        "device_id": f"DEV-{np.random.randint(1,6)}",
        "temperature_celsius": round(temp, 1),
        "humidity_percent": round(60 + np.random.normal(0, 10) + (temp - 5) * 2, 0),
        "battery_percent": round(battery, 0),
        "door_open": door_open,
        "has_failure": has_failure
    }
    records.append(record)

with open("data/raw/coldchain_data.ndjson", "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

failure_rate = sum(r['has_failure'] for r in records) / len(records) * 100
print(f"\nGenerated {len(records)} records")
print(f"Failure rate: {failure_rate:.1f}%")
print("File: data/raw/coldchain_data.ndjson")