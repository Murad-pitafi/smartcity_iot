"""
LSTM-based Anomaly Detection Engine
Architecture:
  - Sliding window of 20 timesteps
  - LSTM encoder learns normal patterns
  - Reconstruction error flags anomalies
  - Threshold set at mean + 2.5 * std of training errors
  - Per-sensor models for temperature, vibration, energy, air quality
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import json


@dataclass
class AnomalyResult:
    sensor: str
    is_anomaly: bool
    score: float          # reconstruction error (higher = more anomalous)
    threshold: float
    severity: str         # "normal", "warning", "critical"
    confidence: float     # 0–1


@dataclass 
class ModelState:
    thresholds: Dict[str, float] = field(default_factory=dict)
    trained: bool = False


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder
        self.encoder = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.1
        )
        # Decoder
        self.decoder = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=0.1
        )
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)
        _, (h, c) = self.encoder(x)
        # Repeat hidden state for decoder input
        decoder_input = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        decoded, _ = self.decoder(decoder_input, (h, c))
        return self.output_layer(decoded)


def make_windows(series: np.ndarray, window: int = 20) -> np.ndarray:
    """Slide a window over 1D series → (N, window, 1)"""
    out = []
    for i in range(len(series) - window):
        out.append(series[i : i + window])
    return np.array(out, dtype=np.float32).reshape(-1, window, 1)


def normalize(series: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mu, sigma = series.mean(), series.std() + 1e-8
    return (series - mu) / sigma, mu, sigma


class AnomalyDetector:
    SENSORS = ["temperature", "vibration", "energy_kwh", "air_quality_pm25"]
    WINDOW = 20
    EPOCHS = 25
    LR = 1e-3

    def __init__(self):
        self.models: Dict[str, LSTMAutoencoder] = {}
        self.norms: Dict[str, Tuple[float, float]] = {}
        self.state = ModelState()
        self.device = torch.device("cpu")

    def _train_one(self, sensor: str, series: np.ndarray) -> float:
        normed, mu, sigma = normalize(series)
        self.norms[sensor] = (mu, sigma)

        windows = make_windows(normed, self.WINDOW)
        # Train on first 80%
        n_train = int(len(windows) * 0.8)
        X_train = torch.tensor(windows[:n_train])

        model = LSTMAutoencoder()
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR)
        criterion = nn.MSELoss()

        dataset = TensorDataset(X_train, X_train)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model.train()
        for _ in range(self.EPOCHS):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

        # Compute threshold on validation set
        model.eval()
        X_val = torch.tensor(windows[n_train:])
        with torch.no_grad():
            recon = model(X_val)
            errors = ((recon - X_val) ** 2).mean(dim=(1, 2)).numpy()

        threshold = float(errors.mean() + 2.5 * errors.std())
        self.models[sensor] = model
        self.state.thresholds[sensor] = threshold
        return threshold

    def train(self, df) -> Dict[str, float]:
        """Train all sensor models. Returns thresholds."""
        thresholds = {}
        for sensor in self.SENSORS:
            series = df[sensor].values.astype(np.float32)
            thresh = self._train_one(sensor, series)
            thresholds[sensor] = thresh
        self.state.trained = True
        return thresholds

    def score_window(self, sensor: str, window: np.ndarray) -> AnomalyResult:
        """Score a single window of `WINDOW` recent readings."""
        if sensor not in self.models:
            raise ValueError(f"Model for {sensor} not trained")

        mu, sigma = self.norms[sensor]
        normed = (window - mu) / (sigma + 1e-8)
        x = torch.tensor(normed, dtype=torch.float32).reshape(1, self.WINDOW, 1)

        self.models[sensor].eval()
        with torch.no_grad():
            recon = self.models[sensor](x)
            score = float(((recon - x) ** 2).mean().item())

        threshold = self.state.thresholds[sensor]
        is_anomaly = score > threshold
        ratio = score / (threshold + 1e-8)

        if ratio < 0.7:
            severity = "normal"
        elif ratio < 1.0:
            severity = "warning"
        elif ratio < 2.0:
            severity = "critical"
        else:
            severity = "critical"

        confidence = min(1.0, ratio) if is_anomaly else max(0.0, 1.0 - ratio)

        return AnomalyResult(
            sensor=sensor,
            is_anomaly=is_anomaly,
            score=round(score, 6),
            threshold=round(threshold, 6),
            severity=severity,
            confidence=round(confidence, 3),
        )

    def score_latest(self, df, n_latest: int = 20) -> List[AnomalyResult]:
        """Score all sensors on the latest n_latest rows."""
        results = []
        for sensor in self.SENSORS:
            series = df[sensor].values[-n_latest:].astype(np.float32)
            if len(series) < self.WINDOW:
                continue
            results.append(self.score_window(sensor, series))
        return results
