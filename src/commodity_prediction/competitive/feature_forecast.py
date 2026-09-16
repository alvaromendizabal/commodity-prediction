"""Competitive feature-forecasting and prediction-prior research.

This module independently reimplements and adapts mechanisms described in public
MITSUI competition writeups. It deliberately preserves the repository's stricter
point-in-time validation contract instead of reproducing any ambiguous look-ahead
or leaderboard-tuned behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from commodity_prediction.metrics import correlation_sharpe, daily_rank_correlations


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def target_assets(pairs: pd.DataFrame) -> list[str]:
    assets: set[str] = set()
    for pair in pairs["pair"]:
        assets.update(pair.split(" - "))
    return sorted(assets)


@dataclass(frozen=True)
class ScaleState:
    means: np.ndarray
    scales: np.ndarray
    positive_floor: np.ndarray
    space: str

    def transform(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        if self.space == "log":
            x = np.log(np.maximum(x, self.positive_floor))
        x = np.where(np.isfinite(x), x, self.means)
        return ((x - self.means) / self.scales).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64) * self.scales + self.means
        if self.space == "log":
            x = np.exp(np.clip(x, -50, 50))
        return np.maximum(x, self.positive_floor)


def fit_scale(values: np.ndarray, space: str) -> ScaleState:
    if space not in {"raw", "log"}:
        raise ValueError(f"Unsupported forecast space: {space}")
    x = np.asarray(values, dtype=np.float64)
    finite_positive = np.where(np.isfinite(x) & (x > 0), x, np.nan)
    floor = np.nanpercentile(finite_positive, 0.1, axis=0)
    fallback_floor = np.nanmedian(finite_positive, axis=0)
    floor = np.where(np.isfinite(floor) & (floor > 0), floor * 0.1, fallback_floor * 1e-6)
    floor = np.where(np.isfinite(floor) & (floor > 0), floor, 1e-12)
    if space == "log":
        x = np.log(np.maximum(x, floor))
    means = np.nanmean(x, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    filled = np.where(np.isfinite(x), x, means)
    scales = filled.std(axis=0, ddof=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
    return ScaleState(means=means, scales=scales, positive_floor=floor, space=space)


def make_windows(z: np.ndarray, window: int, target_start: int, target_stop: int) -> tuple[np.ndarray, np.ndarray]:
    if window < 2 or target_start < window or target_stop <= target_start:
        raise ValueError("Invalid sequence window bounds")
    xs = np.stack([z[t - window : t] for t in range(target_start, target_stop)]).astype(np.float32)
    ys = z[target_start:target_stop].astype(np.float32)
    return xs, ys


def targets_from_forecast_path(path: np.ndarray, pairs: pd.DataFrame, assets: list[str]) -> np.ndarray:
    """Convert predicted d+1..d+5 asset prices to the official target definitions."""
    if path.ndim != 2 or path.shape[0] < 5:
        raise ValueError("Forecast path must contain at least five future rows")
    positions = {name: i for i, name in enumerate(assets)}
    out = np.zeros(len(pairs), dtype=np.float64)
    for j, row in enumerate(pairs.itertuples(index=False)):
        horizon = int(row.lag)
        if not 1 <= horizon <= 4:
            raise ValueError("Expected target horizons 1..4")
        components = row.pair.split(" - ")
        returns = []
        for asset in components:
            i = positions[asset]
            start = max(float(path[0, i]), 1e-12)
            end = max(float(path[horizon, i]), 1e-12)
            returns.append(math.log(end) - math.log(start))
        out[j] = returns[0] if len(returns) == 1 else returns[0] - returns[1]
    return out


def released_label_signal(
    y: pd.DataFrame,
    pairs: pd.DataFrame,
    prediction_positions: np.ndarray,
    trailing: int = 5,
) -> pd.DataFrame:
    """Mean only labels that would have been released by each prediction date.

    For a target with lag h, label t is available at prediction date t+h+1.
    This mirrors the repository's verified release calendar and never reaches into
    unreleased validation labels.
    """
    if trailing < 1:
        raise ValueError("trailing must be positive")
    output = np.zeros((len(prediction_positions), y.shape[1]), dtype=np.float64)
    for j, row in enumerate(pairs.itertuples(index=False)):
        delay = int(row.lag) + 1
        values = y.iloc[:, j].to_numpy(dtype=float)
        historical_mean = float(np.nanmean(values[: max(1, int(prediction_positions[0]) - delay + 1)]))
        if not np.isfinite(historical_mean):
            historical_mean = 0.0
        for i, pos in enumerate(prediction_positions):
            available_stop = int(pos) - delay + 1
            start = max(0, available_stop - trailing)
            recent = values[start:available_stop]
            recent = recent[np.isfinite(recent)]
            output[i, j] = float(recent.mean()) if len(recent) else historical_mean
    return pd.DataFrame(output, index=y.index[prediction_positions], columns=y.columns)


def rank_standardize_rows(y: pd.DataFrame) -> np.ndarray:
    values = y.to_numpy(dtype=float)
    z = np.zeros_like(values, dtype=np.float64)
    for i, row in enumerate(values):
        observed = np.isfinite(row)
        if observed.sum() < 2:
            continue
        ranks = pd.Series(row[observed]).rank(method="average").to_numpy(dtype=float, copy=True)
        ranks -= ranks.mean()
        std = ranks.std(ddof=0)
        if std > 1e-12:
            ranks /= std
        z[i, observed] = ranks
    return z


def mean_rank_prior(train_y: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    ranked = rank_standardize_rows(train_y)
    vector = ranked.mean(axis=0)
    if np.std(vector) <= 1e-12:
        vector = np.arange(train_y.shape[1], dtype=float)
    return pd.DataFrame(np.tile(vector, (len(index), 1)), index=index, columns=train_y.columns)


def regularized_kelly_rank_prior(train_y: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Training-only covariance-regularized cross-sectional rank direction.

    Ledoit-Wolf shrinkage replaces hand-tuning a covariance regularization weight on
    the outer validation period. This is an adaptation of the public #10 idea, not
    a copy of its implementation.
    """
    ranked = rank_standardize_rows(train_y)
    mu = ranked.mean(axis=0)
    covariance = LedoitWolf().fit(ranked)
    direction = covariance.precision_ @ mu
    direction -= direction.mean()
    std = direction.std(ddof=0)
    if not np.isfinite(direction).all() or std <= 1e-12:
        raise ValueError("Degenerate regularized rank prior")
    direction /= std
    return pd.DataFrame(np.tile(direction, (len(index), 1)), index=index, columns=train_y.columns)


def score_prediction(truth: pd.DataFrame, prediction: pd.DataFrame) -> dict:
    daily = daily_rank_correlations(truth, prediction)
    return {
        "official_metric": correlation_sharpe(daily),
        "mean_daily_rank_correlation": float(daily.mean()),
        "std_daily_rank_correlation": float(daily.std(ddof=0)),
        "positive_correlation_fraction": float((daily > 0).mean()),
        "daily_rank_correlations": daily.tolist(),
    }


def blend(a: pd.DataFrame, b: pd.DataFrame, weight_a: float) -> pd.DataFrame:
    if not a.index.equals(b.index) or not a.columns.equals(b.columns):
        raise ValueError("Blend inputs must have identical schemas")
    return a * float(weight_a) + b * (1.0 - float(weight_a))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(directory: Path, files: list[str], payload: dict) -> None:
    data = {"payload": payload, "files": {name: sha256(directory / name) for name in files}}
    (directory / "manifest.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def verify_manifest(directory: Path, expected_payload: dict) -> bool:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        data = json.loads(manifest_path.read_text())
        if data.get("payload") != expected_payload:
            return False
        return all((directory / name).exists() and sha256(directory / name) == digest for name, digest in data["files"].items())
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _device_name() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def train_feature_forecaster(
    values: np.ndarray,
    train_stop: int,
    config: dict,
    space: str,
    log: Callable[[str, dict], None],
) -> tuple[object, ScaleState, dict]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    seed = int(config["seed"])
    set_deterministic_seed(seed)
    window = int(config["window_days"])
    inner_stop = max(window + 25, int(train_stop * (1.0 - float(config["inner_validation_fraction"]))))
    if inner_stop >= train_stop - 10:
        inner_stop = train_stop - 10
    inner_scale = fit_scale(values[:inner_stop], space)
    z_inner_all = inner_scale.transform(values[:train_stop])
    x_train, y_train = make_windows(z_inner_all, window, window, inner_stop)
    x_valid, y_valid = make_windows(z_inner_all, window, inner_stop, train_stop)

    class Forecaster(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=dim,
                hidden_size=int(config["hidden_dim"]),
                num_layers=int(config["num_layers"]),
                dropout=float(config["dropout"]) if int(config["num_layers"]) > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Linear(int(config["hidden_dim"]), dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output, _ = self.lstm(x)
            return self.head(output[:, -1])

    device = torch.device(_device_name())

    def fit_once(xa: np.ndarray, ya: np.ndarray, epochs: int, validation: tuple[np.ndarray, np.ndarray] | None) -> tuple[nn.Module, int, list[dict]]:
        set_deterministic_seed(seed)
        model = Forecaster(xa.shape[-1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
        loss_fn = nn.MSELoss()
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(xa), torch.from_numpy(ya)),
            batch_size=int(config["batch_size"]),
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        best_state = None
        best_epoch = epochs
        best_loss = float("inf")
        stale = 0
        history = []
        if validation is not None:
            vx = torch.from_numpy(validation[0]).to(device)
            vy = torch.from_numpy(validation[1]).to(device)
        for epoch in range(1, epochs + 1):
            model.train()
            total = 0.0
            count = 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach().cpu()) * len(xb)
                count += len(xb)
            train_loss = total / max(count, 1)
            valid_loss = None
            if validation is not None:
                model.eval()
                with torch.no_grad():
                    valid_loss = float(loss_fn(model(vx), vy).detach().cpu())
                if valid_loss < best_loss - 1e-8:
                    best_loss = valid_loss
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
            history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
            log("epoch", {"space": space, "epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
            if validation is not None and stale >= int(config["early_stopping_patience"]):
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, best_epoch, history

    _, best_epoch, selection_history = fit_once(
        x_train,
        y_train,
        int(config["max_epochs"]),
        (x_valid, y_valid),
    )

    final_scale = fit_scale(values[:train_stop], space)
    z_all = final_scale.transform(values[:train_stop])
    x_all, y_all = make_windows(z_all, window, window, train_stop)
    final_model, _, refit_history = fit_once(x_all, y_all, best_epoch, None)
    metadata = {
        "device": str(device),
        "space": space,
        "inner_stop": inner_stop,
        "train_stop": train_stop,
        "best_epoch": best_epoch,
        "selection_history": selection_history,
        "refit_history": refit_history,
        "training_windows": len(x_all),
        "parameters": sum(p.numel() for p in final_model.parameters()),
    }
    return final_model, final_scale, metadata


def recursive_predict(
    model: object,
    scale: ScaleState,
    observed_values: np.ndarray,
    prediction_positions: np.ndarray,
    config: dict,
    pairs: pd.DataFrame,
    assets: list[str],
    log: Callable[[str, dict], None],
) -> pd.DataFrame:
    import torch

    device = torch.device(_device_name())
    model = model.to(device)
    model.eval()
    window = int(config["window_days"])
    steps = int(config["forecast_steps"])
    rows = []
    started = time.monotonic()
    with torch.no_grad():
        for n, pos in enumerate(prediction_positions, start=1):
            if pos - window + 1 < 0:
                raise ValueError("Not enough observed history for inference window")
            context = scale.transform(observed_values[pos - window + 1 : pos + 1])
            seq = context.copy()
            future_z = []
            for _ in range(steps):
                xb = torch.from_numpy(seq[-window:][None, :, :]).to(device)
                next_z = model(xb).detach().cpu().numpy()[0]
                future_z.append(next_z)
                seq = np.vstack([seq, next_z])
            future_prices = scale.inverse(np.asarray(future_z))
            rows.append(targets_from_forecast_path(future_prices, pairs, assets))
            if n == 1 or n == len(prediction_positions) or n % 30 == 0:
                log(
                    "prediction_progress",
                    {
                        "space": scale.space,
                        "completed": n,
                        "total": len(prediction_positions),
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    },
                )
    return pd.DataFrame(rows, index=prediction_positions, columns=pairs["target"].tolist())
