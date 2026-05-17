# Smart City IoT Anomaly Detection Dashboard
**By Muhammad Murad — AI/ML Engineer | AgriEngineering 2026 | IEEE YESIST12 Global Top 5**

A production-grade LSTM-based anomaly detection system for smart city sensor networks — built to showcase real-time AI inference aligned with Saudi Arabia's Vision 2030 and NEOM's cognitive city infrastructure.

---

## 🏙️ What This Does
- **Simulates 4 IoT sensor streams**: Temperature, Structural Vibration, Energy Consumption, Air Quality (PM2.5)
- **Trains LSTM autoencoders** on each sensor to learn normal patterns
- **Detects anomalies** in real-time using reconstruction error thresholding (μ + 2.5σ)
- **Serves results** via a FastAPI REST API
- **Visualizes live** in a dark industrial dashboard with Chart.js

---

## 📁 Project Structure
```
smartcity_iot/
├── backend/
│   ├── main.py          # FastAPI app (all endpoints)
│   └── detector.py      # LSTM Autoencoder + AnomalyDetector class
├── data/
│   └── sensor_simulator.py  # Realistic IoT data generator with injected anomalies
├── frontend/
│   └── index.html       # Live dashboard (pure HTML/CSS/JS)
├── requirements.txt
├── Dockerfile
├── railway.toml         # Free Railway deployment config
└── README.md
```

---

## ⚡ Run Locally (5 minutes)

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Start the API server**
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
> ⏳ First boot trains 4 LSTM models — takes ~30 seconds on CPU.

Then open `http://localhost:8000/` in your browser.

**3. Open the dashboard**
```
Open frontend/index.html in your browser
```
The dashboard auto-connects to `localhost:8000` and polls every 3 seconds.

## 📦 Push To GitHub Cleanly

Before pushing, keep the local environment out of Git:
- Do not commit `.venv/`
- Do not commit `__pycache__/` or `*.pyc`
- Do not commit `.env` files or secrets

The repo now includes a `.gitignore` for those files. If `.venv/` was already tracked in Git, remove it from the index once with:
```bash
git rm -r --cached .venv
```

Typical push flow:
```bash
git add .
git commit -m "Update dashboard and deployment docs"
git push origin main
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service status + model state |
| GET | `/sensors/live` | Latest readings + LSTM anomaly scores |
| GET | `/sensors/history?n=200` | Full time-series for all sensors |
| GET | `/sensors/stats` | Summary statistics |
| GET | `/anomalies/recent?limit=20` | Recent anomaly event log |
| POST | `/train` | Retrain models on fresh simulated data |

**Example:**
```bash
curl http://localhost:8000/sensors/live | python -m json.tool
```

---

## 🚀 Deploy Free on Railway

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Railway auto-detects `Dockerfile` and deploys
4. Update `API` variable in `frontend/index.html` to your Railway URL
5. Host `index.html` on GitHub Pages (free)

**That's it — fully deployed, zero cost.**

---

## 🧠 Model Architecture

```
Input: Sliding window of 20 timesteps (per sensor)
       ↓
LSTM Encoder (2 layers, hidden=32)
       ↓
LSTM Decoder (2 layers, hidden=32)
       ↓
Reconstruction error (MSE)
       ↓
Threshold: μ_val + 2.5 * σ_val
       ↓
Severity: normal / warning / critical
```

**Why LSTM Autoencoder?**
- Learns temporal patterns without labeled anomaly data (unsupervised)
- Generalizes to drift, spikes, and sustained shifts
- Lightweight enough to run on CPU / edge devices
- Directly applicable to NEOM, Aramco predictive maintenance

---

## 🎯 Vision 2030 Alignment

| KSA Priority | This Project |
|---|---|
| Smart city infrastructure (NEOM) | Real-time sensor monitoring |
| Industrial AI (Aramco) | Predictive maintenance via LSTM |
| AI-driven operations | Autonomous anomaly alerting |
| MLOps maturity | FastAPI + Docker + structured inference |

---

## 📬 Contact
**Muhammad Murad**
- Email: pitafimurad99@gmail.com
- LinkedIn: [linkedin.com/in/murad-pitafi](https://linkedin.com/in/murad-pitafi)
- GitHub: [github.com/murad-pitafi](https://github.com/murad-pitafi)
- Paper: Agentic AI Framework for Smart Agriculture — AgriEngineering, vol. 8, Jan 2026

---

## © Copyright & License

- **Copyright** (c) 2026 Muhammad Murad (Murad-pitafi). All rights reserved by the author.
- This repository may include third-party open-source components; those components remain subject to their own licenses.
- If you would like to make this project open-source under a permissive license (e.g. MIT, Apache-2.0) I can add a `LICENSE` file — tell me which license you prefer.

