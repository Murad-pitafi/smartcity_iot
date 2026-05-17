"""
Smart City IoT Sensor Simulator
Generates realistic time-series data for:
- Temperature sensors (HVAC, outdoor)
- Vibration sensors (structural/machinery)
- Energy consumption (kWh)
- Air quality (PM2.5)
With injected anomalies for LSTM detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_sensor_stream(n_points: int = 500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    now = datetime.utcnow()
    timestamps = [now - timedelta(minutes=(n_points - i)) for i in range(n_points)]

    t = np.linspace(0, 4 * np.pi, n_points)

    # Temperature: diurnal cycle + noise
    temperature = (
        28
        + 4 * np.sin(t)
        + np.random.normal(0, 0.4, n_points)
    )

    # Vibration: low baseline with mechanical rhythm
    vibration = (
        0.05
        + 0.02 * np.sin(3 * t)
        + np.abs(np.random.normal(0, 0.01, n_points))
    )

    # Energy consumption: peaks at working hours
    energy = (
        120
        + 40 * np.sin(t - np.pi / 4)
        + np.random.normal(0, 5, n_points)
    )

    # Air quality PM2.5: generally low, traffic spikes
    air_quality = (
        18
        + 5 * np.sin(t + np.pi / 3)
        + np.abs(np.random.normal(0, 2, n_points))
    )

    # --- Inject anomalies ---
    anomaly_labels = np.zeros(n_points, dtype=int)
    anomaly_types = ["none"] * n_points

    # Spike anomalies
    spike_indices = np.random.choice(range(50, n_points - 10), size=8, replace=False)
    for idx in spike_indices:
        sensor = np.random.choice(["temperature", "vibration", "energy", "air_quality"])
        if sensor == "temperature":
            temperature[idx : idx + 3] += np.random.uniform(6, 10)
            anomaly_types[idx] = "temp_spike"
        elif sensor == "vibration":
            vibration[idx : idx + 3] += np.random.uniform(0.15, 0.25)
            anomaly_types[idx] = "vibration_spike"
        elif sensor == "energy":
            energy[idx : idx + 3] += np.random.uniform(80, 120)
            anomaly_types[idx] = "energy_spike"
        else:
            air_quality[idx : idx + 3] += np.random.uniform(40, 70)
            anomaly_types[idx] = "air_quality_spike"
        anomaly_labels[idx : idx + 3] = 1

    # Drift anomaly
    drift_max = max(int(n_points * 0.4), int(n_points * 0.7) - 21)
    drift_start = np.random.randint(int(n_points * 0.4), max(int(n_points * 0.4) + 1, drift_max))
    drift_end = min(drift_start + 20, n_points)
    drift_len = drift_end - drift_start
    temperature[drift_start:drift_end] += np.linspace(0, 8, drift_len)
    anomaly_labels[drift_start:drift_end] = 1
    for i in range(drift_start, drift_end):
        anomaly_types[i] = "temp_drift"

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": np.round(temperature, 2),
            "vibration": np.round(vibration, 4),
            "energy_kwh": np.round(energy, 2),
            "air_quality_pm25": np.round(air_quality, 2),
            "anomaly": anomaly_labels,
            "anomaly_type": anomaly_types,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_sensor_stream()
    df.to_csv("sensor_data.csv", index=False)
    print(f"Generated {len(df)} rows, {df['anomaly'].sum()} anomaly points")
    print(df.tail())
