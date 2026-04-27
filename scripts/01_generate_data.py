"""
Script 1: Enhanced Synthetic Data Generation with Realistic Failure Patterns
Produces data where failures are detectable but NOT obvious (prevents perfect 100% accuracy)
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

print("=" * 60)
print("GENERATING REALISTIC COLD CHAIN DATA")
print("=" * 60)

# Parameters
DURATION_DAYS = 7
INTERVAL_MINUTES = 5
TOTAL_RECORDS = int(DURATION_DAYS * 24 * 60 / INTERVAL_MINUTES)  # 2016 records
FAILURE_RATE = 0.02  # 2% of records are failures

print(f"Generating {TOTAL_RECORDS} records over {DURATION_DAYS} days...")

records = []
start_time = datetime.now() - timedelta(days=DURATION_DAYS)
base_drift = 0.0
failure_history = []  # Track recent failures for realistic patterns

for i in range(TOTAL_RECORDS):
    timestamp = start_time + timedelta(minutes=i * INTERVAL_MINUTES)
    hour = timestamp.hour
    
    # ----- TEMPERATURE GENERATION (Realistic daily cycle) -----
    # Base temperature for vaccine cold chain (2-8°C target)
    base_temp = 5.0
    
    # Daily cycle (cooler at night, warmer in afternoon)
    daily_variation = 1.2 * abs((hour - 14) / 12)
    
    # Gradual equipment degradation (drift increases over time)
    if i > 800:  # After ~2.8 days
        base_drift = min(1.5, base_drift + 0.001)
    
    # Random noise (sensor error, environmental factors)
    noise = np.random.normal(0, 0.3)
    
    # Calculate temperature
    temp = base_temp + base_drift + daily_variation + noise
    
    # ----- FAILURE INJECTION (Realistic - not too obvious) -----
    has_failure = 0
    failure_type = None
    
    if np.random.random() < FAILURE_RATE:
        has_failure = 1
        
        # Choose from realistic failure types
        failure_type = np.random.choice([
            "gradual_rise",      # Compressor slowly failing
            "sudden_spike",      # Door left open or power surge
            "battery_drain",     # Power supply issue
            "sensor_noise"       # Faulty sensor reading
        ], p=[0.35, 0.25, 0.25, 0.15])  # Probabilities
        
        if failure_type == "gradual_rise":
            # Subtle temperature increase (0.5-1.5°C over several readings)
            temp = temp + np.random.uniform(0.5, 1.5)
            
        elif failure_type == "sudden_spike":
            # Quick spike but not extreme (1.5-3.0°C)
            temp = temp + np.random.uniform(1.5, 3.0)
            # Also increase base drift to simulate damage
            base_drift = base_drift + np.random.uniform(0, 0.02)
            
        elif failure_type == "battery_drain":
            # Only affect battery, temperature changes normally
            pass
            
        elif failure_type == "sensor_noise":
            # Add extra random noise to reading
            temp = temp + np.random.normal(0, 1.5)
    
    # Clamp temperature to realistic range
    temp = max(-2, min(15, temp))
    
    # ----- DOOR OPEN EVENTS (Realistic pattern) -----
    door_open = 0
    if 8 <= hour <= 18:
        # Working hours: more door activity
        door_open = 1 if np.random.random() < 0.08 else 0
    else:
        # Night time: occasional checks only
        door_open = 1 if np.random.random() < 0.02 else 0
    
    # If there was a recent failure, increase door activity (operator checking)
    if len(failure_history) > 0 and i - failure_history[-1] < 20:
        door_open = door_open or (np.random.random() < 0.2)
    
    # ----- BATTERY LEVEL (Realistic drain) -----
    battery_drain = 0.008  # Base drain per reading
    
    # Additional factors
    if door_open:
        battery_drain += 0.015  # More drain when door open
    if has_failure and failure_type == "battery_drain":
        battery_drain += 0.05  # Battery failure scenario
    if temp > 8:
        battery_drain += 0.01  # Compressor working harder
    if temp < 2:
        battery_drain += 0.005  # Heater working
    
    # Calculate battery (with some randomness)
    battery = 100 - (i * battery_drain * 60 / INTERVAL_MINUTES) + np.random.normal(0, 1)
    battery = max(0, min(100, battery))
    
    # ----- HUMIDITY (Correlated with temperature) -----
    humidity = 60 - (temp - 5) * 2 + np.random.normal(0, 5)
    humidity = max(30, min(90, humidity))
    
    # Record failure for history
    if has_failure:
        failure_history.append(i)
        # Only keep last 10 failures
        if len(failure_history) > 10:
            failure_history.pop(0)
    
    # Create record
    record = {
        "timestamp": timestamp.isoformat(),
        "device_id": f"DEV-{np.random.randint(1, 6)}",
        "temperature_celsius": round(temp, 1),
        "humidity_percent": round(humidity, 0),
        "battery_percent": round(battery, 0),
        "door_open": door_open,
        "has_failure": has_failure,
        "failure_type": failure_type if has_failure else None
    }
    records.append(record)

# Save to file
output_path = "data/raw/coldchain_data.ndjson"
with open(output_path, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

# Calculate statistics
failure_count = sum(r['has_failure'] for r in records)
failure_rate = failure_count / len(records) * 100

# Calculate temperature gap between normal and failure
temps_normal = [r['temperature_celsius'] for r in records if r['has_failure'] == 0]
temps_failure = [r['temperature_celsius'] for r in records if r['has_failure'] == 1]

avg_temp_normal = sum(temps_normal) / len(temps_normal)
avg_temp_failure = sum(temps_failure) / len(temps_failure)
temp_gap = avg_temp_failure - avg_temp_normal

print("\n" + "=" * 60)
print("DATA GENERATION COMPLETE")
print("=" * 60)
print(f"✅ Total records: {len(records)}")
print(f"✅ Failure rate: {failure_rate:.1f}%")
print(f"✅ Temperature range: {min(r['temperature_celsius'] for r in records)}°C to {max(r['temperature_celsius'] for r in records)}°C")
print(f"✅ Avg temp (normal): {avg_temp_normal:.1f}°C")
print(f"✅ Avg temp (failure): {avg_temp_failure:.1f}°C")
print(f"✅ Temperature gap: {temp_gap:.1f}°C")

if temp_gap < 2.5:
    print("   ✅ GOOD: Temperature gap is realistic (not too obvious)")
else:
    print("   ⚠️ NOTE: Temperature gap is moderate - failures should be detectable but not trivial")

print(f"\n📁 Output saved to: {output_path}")
print("\nNext step: Run 02_preprocess_data.py")