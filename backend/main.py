"""
Smart City IoT Anomaly Detection API
  GET /                  → dashboard HTML (auto-detects host)
  GET /sensors/live      → latest readings + anomaly scores
  GET /sensors/history   → full time-series
  GET /sensors/stats     → summary statistics
  GET /anomalies/recent  → recent anomaly events
  POST /train            → retrain models
  GET /health            → health check
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import sys, os, asyncio, random
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.sensor_simulator import generate_sensor_stream
from backend.detector import AnomalyDetector

detector = AnomalyDetector()
df_global: pd.DataFrame = pd.DataFrame()
anomaly_log: List[Dict] = []


def bootstrap():
    global df_global, anomaly_log
    print("Generating sensor data...")
    df_global = generate_sensor_stream(n_points=500)
    print("Training LSTM models (~30s on CPU)...")
    detector.train(df_global)
    print("Models trained.")
    for _, row in df_global[df_global["anomaly"] == 1].iterrows():
        anomaly_log.append({
            "timestamp": row["timestamp"].isoformat(),
            "sensor": row["anomaly_type"].split("_")[0],
            "anomaly_type": row["anomaly_type"],
            "severity": "critical" if "spike" in row["anomaly_type"] else "warning",
        })


async def simulation_loop():
    """Tick every 3s — adds a new row with random walk + occasional spike."""
    global df_global
    await asyncio.sleep(8)
    while True:
        try:
            if not df_global.empty:
                last = df_global.iloc[-1].copy()
                last["timestamp"] = datetime.utcnow()
                last["temperature"]       = round(float(last["temperature"]) + random.uniform(-0.8, 0.8), 2)
                last["vibration"]         = round(max(0.01, float(last["vibration"]) + random.uniform(-0.005, 0.005)), 4)
                last["energy_kwh"]        = round(float(last["energy_kwh"]) + random.uniform(-3, 3), 2)
                last["air_quality_pm25"]  = round(max(5, float(last["air_quality_pm25"]) + random.uniform(-1.5, 1.5)), 2)
                last["anomaly"] = 0
                last["anomaly_type"] = "none"
                if random.random() < 0.05:
                    spike = random.choice(["temperature", "vibration", "energy_kwh", "air_quality_pm25"])
                    deltas = {"temperature": 7, "vibration": 0.15, "energy_kwh": 80, "air_quality_pm25": 45}
                    last[spike] = round(float(last[spike]) + deltas[spike], 4)
                    last["anomaly"] = 1
                    last["anomaly_type"] = f"{spike.split('_')[0]}_spike"
                    anomaly_log.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "sensor": spike,
                        "anomaly_type": last["anomaly_type"],
                        "severity": "critical",
                    })
                df_global = pd.concat([df_global, pd.DataFrame([last])], ignore_index=True).tail(600)
        except Exception as e:
            print(f"Sim error: {e}")
        await asyncio.sleep(3)


@asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap()
    asyncio.create_task(simulation_loop())
    yield


app = FastAPI(title="Smart City IoT API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def sensor_display_name(s):
    return {"temperature":"Temperature","vibration":"Structural Vibration","energy_kwh":"Energy Consumption","air_quality_pm25":"Air Quality (PM2.5)"}.get(s, s)

def sensor_unit(s):
    return {"temperature":"°C","vibration":"g","energy_kwh":"kWh","air_quality_pm25":"µg/m³"}.get(s, "")


@app.get("/", response_class=HTMLResponse)
def dashboard():
    html = (Path(__file__).parent.parent / "frontend" / "index.html").read_text()
    # Make API calls use relative URLs (same host) — works on any deployment
    html = html.replace(
        "window.location.hostname === 'localhost' ? 'http://localhost:8000' : ''",
        "''"
    )
    return HTMLResponse(content=html)


@app.get("/health")
def health():
    return {"status": "online", "model_trained": detector.state.trained,
            "data_points": len(df_global), "anomalies_logged": len(anomaly_log),
            "timestamp": datetime.utcnow().isoformat()}


@app.get("/sensors/live")
def sensors_live():
    if df_global.empty or not detector.state.trained:
        return {"error": "System initializing, try again in 30s"}
    latest = df_global.iloc[-1]
    scores = detector.score_latest(df_global)
    score_map = {r.sensor: r for r in scores}
    readings = []
    for sensor in detector.SENSORS:
        r = score_map.get(sensor)
        readings.append({
            "sensor": sensor,
            "display_name": sensor_display_name(sensor),
            "unit": sensor_unit(sensor),
            "value": round(float(latest[sensor]), 3),
            "timestamp": latest["timestamp"].isoformat() if hasattr(latest["timestamp"], "isoformat") else str(latest["timestamp"]),
            "anomaly": r.is_anomaly if r else False,
            "anomaly_score": r.score if r else 0,
            "threshold": r.threshold if r else 0,
            "severity": r.severity if r else "normal",
            "confidence": r.confidence if r else 0,
        })
    return {"timestamp": datetime.utcnow().isoformat(), "readings": readings, "total_anomalies": len(anomaly_log)}


@app.get("/sensors/history")
def sensors_history(n: int = Query(default=200, ge=20, le=500)):
    if df_global.empty:
        return {"error": "No data available"}
    subset = df_global.tail(n).copy()
    history = {}
    for sensor in detector.SENSORS:
        history[sensor] = {
            "timestamps": [t.isoformat() if hasattr(t, "isoformat") else str(t) for t in subset["timestamp"]],
            "values": subset[sensor].tolist(),
            "anomaly_flags": subset["anomaly"].tolist(),
            "unit": sensor_unit(sensor),
            "display_name": sensor_display_name(sensor),
        }
    return {"n": n, "sensors": history}


@app.get("/sensors/stats")
def sensors_stats():
    if df_global.empty:
        return {"error": "No data available"}
    stats = {}
    for sensor in detector.SENSORS:
        s = df_global[sensor]
        stats[sensor] = {
            "display_name": sensor_display_name(sensor), "unit": sensor_unit(sensor),
            "mean": round(float(s.mean()), 3), "std": round(float(s.std()), 3),
            "min": round(float(s.min()), 3), "max": round(float(s.max()), 3),
            "current": round(float(s.iloc[-1]), 3),
            "threshold": round(detector.state.thresholds.get(sensor, 0), 6),
        }
    return {"stats": stats, "total_points": len(df_global)}


@app.get("/anomalies/recent")
def anomalies_recent(limit: int = Query(default=20, le=100)):
    return {"count": len(anomaly_log), "anomalies": anomaly_log[-limit:][::-1]}


@app.post("/train")
def retrain():
    global df_global, anomaly_log
    df_global = generate_sensor_stream(n_points=500, seed=np.random.randint(0, 9999))
    thresholds = detector.train(df_global)
    anomaly_log.clear()
    return {"status": "retrained", "thresholds": thresholds, "data_points": len(df_global)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
