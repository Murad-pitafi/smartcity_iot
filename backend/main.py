"""
Smart City IoT Anomaly Detection API
FastAPI backend serving:
  GET /sensors/live      → latest readings + anomaly scores for all sensors
  GET /sensors/history   → full time-series (last N points)
  GET /sensors/stats     → summary statistics
  GET /anomalies/recent  → last 20 detected anomalies
  POST /train            → retrain models on fresh data
  GET /health            → service health check
"""

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.sensor_simulator import generate_sensor_stream
from backend.detector import AnomalyDetector

# ── Global state ──────────────────────────────────────────────────────────────
detector = AnomalyDetector()
df_global: pd.DataFrame = pd.DataFrame()
anomaly_log: List[Dict] = []
training_in_progress = False
training_error = None


def bootstrap():
    global df_global, anomaly_log, training_in_progress, training_error
    training_in_progress = True
    try:
        print("🔄 Generating sensor data...")
        df_global = generate_sensor_stream(n_points=500)
        print("🧠 Training LSTM models in background thread...")
        thresholds = detector.train(df_global)
        print(f"✅ Models trained. Thresholds: {thresholds}")

        # Pre-populate anomaly log from ground-truth labels
        anomaly_rows = df_global[df_global["anomaly"] == 1].copy()
        for _, row in anomaly_rows.iterrows():
            anomaly_log.append({
                "timestamp": row["timestamp"].isoformat(),
                "sensor": row["anomaly_type"].split("_")[0],
                "anomaly_type": row["anomaly_type"],
                "severity": "critical" if "spike" in row["anomaly_type"] else "warning",
            })
        print(f"📋 Loaded {len(anomaly_log)} historical anomaly events")
    except Exception as e:
        training_error = str(e)
        print(f"❌ Training failed: {e}")
    finally:
        training_in_progress = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=bootstrap, daemon=True)
    thread.start()
    yield


app = FastAPI(
    title="Smart City IoT Anomaly Detection API",
    description="LSTM-based real-time anomaly detection for NEOM-style smart city sensor networks",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def sensor_display_name(sensor: str) -> str:
    names = {
        "temperature": "Temperature",
        "vibration": "Structural Vibration",
        "energy_kwh": "Energy Consumption",
        "air_quality_pm25": "Air Quality (PM2.5)",
    }
    return names.get(sensor, sensor)


def sensor_unit(sensor: str) -> str:
    units = {
        "temperature": "°C",
        "vibration": "g",
        "energy_kwh": "kWh",
        "air_quality_pm25": "µg/m³",
    }
    return units.get(sensor, "")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "training" if training_in_progress else ("error" if training_error else "online"),
        "model_trained": detector.state.trained,
        "data_points": len(df_global),
        "anomalies_logged": len(anomaly_log),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/sensors/live")
def sensors_live():
    """Latest readings + LSTM anomaly scores for all 4 sensors."""
    if df_global.empty or not detector.state.trained:
        return {"error": "System initializing, try again in 30s"}

    latest = df_global.iloc[-1]
    scores = detector.score_latest(df_global)
    score_map = {r.sensor: r for r in scores}

    readings = []
    for sensor in detector.SENSORS:
        val = float(latest[sensor])
        result = score_map.get(sensor)
        readings.append({
            "sensor": sensor,
            "display_name": sensor_display_name(sensor),
            "unit": sensor_unit(sensor),
            "value": round(val, 3),
            "timestamp": latest["timestamp"].isoformat(),
            "anomaly": result.is_anomaly if result else False,
            "anomaly_score": result.score if result else 0,
            "threshold": result.threshold if result else 0,
            "severity": result.severity if result else "normal",
            "confidence": result.confidence if result else 0,
        })

    # Update anomaly log if new anomaly detected
    for r in scores:
        if r.is_anomaly:
            anomaly_log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "sensor": r.sensor,
                "anomaly_type": f"lstm_detected_{r.sensor}",
                "severity": r.severity,
                "score": r.score,
            })

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "readings": readings,
        "total_anomalies": len(anomaly_log),
    }


@app.get("/sensors/history")
def sensors_history(n: int = Query(default=200, ge=20, le=500)):
    """Full time-series for all sensors (last n points)."""
    if df_global.empty:
        return {"error": "No data available"}

    subset = df_global.tail(n).copy()
    history = {}
    for sensor in detector.SENSORS:
        history[sensor] = {
            "timestamps": [t.isoformat() for t in subset["timestamp"]],
            "values": subset[sensor].tolist(),
            "anomaly_flags": subset["anomaly"].tolist(),
            "unit": sensor_unit(sensor),
            "display_name": sensor_display_name(sensor),
        }
    return {"n": n, "sensors": history}


@app.get("/sensors/stats")
def sensors_stats():
    """Summary statistics for all sensors."""
    if df_global.empty:
        return {"error": "No data available"}

    stats = {}
    for sensor in detector.SENSORS:
        series = df_global[sensor]
        stats[sensor] = {
            "display_name": sensor_display_name(sensor),
            "unit": sensor_unit(sensor),
            "mean": round(float(series.mean()), 3),
            "std": round(float(series.std()), 3),
            "min": round(float(series.min()), 3),
            "max": round(float(series.max()), 3),
            "current": round(float(series.iloc[-1]), 3),
            "threshold": round(detector.state.thresholds.get(sensor, 0), 6),
            "anomaly_count": int(df_global[df_global["anomaly_type"].str.contains(sensor.split("_")[0])]["anomaly"].sum()),
        }
    return {"stats": stats, "total_points": len(df_global)}


@app.get("/anomalies/recent")
def anomalies_recent(limit: int = Query(default=20, le=100)):
    """Most recent detected anomalies."""
    recent = anomaly_log[-limit:][::-1]
    return {
        "count": len(recent),
        "anomalies": recent,
    }


@app.post("/train")
def retrain():
    """Retrain models on freshly generated sensor data."""
    global df_global, anomaly_log
    df_global = generate_sensor_stream(n_points=500, seed=np.random.randint(0, 9999))
    thresholds = detector.train(df_global)
    anomaly_log.clear()
    return {
        "status": "retrained",
        "thresholds": thresholds,
        "data_points": len(df_global),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
