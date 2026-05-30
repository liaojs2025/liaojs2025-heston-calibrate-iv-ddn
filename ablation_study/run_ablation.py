#!/usr/bin/env python3
"""
Run the IV-DDN ablation study.

This script is self-contained and writes all outputs under ablation_study/results.
It reuses the repository's existing data files and Heston pricing utilities, but it
does not modify or execute the original notebooks/training scripts.

Default full run:
    python run_ablation.py

Useful smoke test:
    python run_ablation.py --epochs 1 --train-limit 2000 --calibration-starts 1 --calibration-steps 2 --force-train
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
HESTON_PROJECT = REPO_ROOT / "Heston_Project"
RESULTS_DIR = SCRIPT_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
CACHE_DIR = RESULTS_DIR / "cache"


def load_heston_pricing_functions():
    pricing_path = HESTON_PROJECT / "modules" / "pricing.py"
    spec = importlib.util.spec_from_file_location("heston_project_pricing", pricing_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Heston pricing utilities from {pricing_path}")
    pricing_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pricing_module)
    return pricing_module.calculate_heston_price, pricing_module.compute_iv_vega_batch


calculate_heston_price, compute_iv_vega_batch = load_heston_pricing_functions()


FEATURE_COLS = ["kappa", "lambda", "sigma", "rho", "v0", "r", "tau", "log_K_S0"]
PRICE_GRAD_COLS = ["d_kappa", "d_lambda", "d_sigma", "d_rho", "d_v0"]
DIV_GRAD_COLS = ["div_kappa", "div_lambda", "div_sigma", "div_rho", "div_v0"]
PARAM_NAMES = ["kappa", "lambda", "sigma", "rho", "v0"]
PARAM_BOUNDS = np.array(
    [
        [0.01, 5.00],
        [0.01, 1.00],
        [0.10, 1.00],
        [-0.90, -0.05],
        [0.01, 1.00],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class VariantConfig:
    key: str
    label: str
    derivative_mode: str
    derivative_loss: str
    activation: str
    calibration_vega_weighted: bool
    lambda_feller: float


@dataclass(frozen=True)
class CalibrationSettings:
    n_starts: int
    lr: float
    max_steps: int
    patience: int


VARIANTS = [
    VariantConfig("A", "IV-NN (no derivative loss)", "none", "smooth_l1", "softplus", False, 0.0),
    VariantConfig("B", "IV-DDN (unweighted derivative loss)", "unweighted", "smooth_l1", "softplus", False, 0.0),
    VariantConfig("C", "IV-DDN (Vega-weighted derivative loss)", "vega_weighted", "smooth_l1", "softplus", True, 0.0),
    VariantConfig("D", "IV-DDN + Vega weighting + Feller penalty", "vega_weighted", "smooth_l1", "softplus", True, 10.0),
    VariantConfig("E", "IV-DDN + Vega weighting + Feller penalty + MSE derivative loss", "vega_weighted", "mse", "softplus", True, 10.0),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class AblationDDN(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        heston_param_dim: int = 5,
        hidden_layers: int = 6,
        neurons_per_layer: int = 150,
        activation: str = "softplus",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.heston_dim = heston_param_dim
        self.activation = activation
        self.register_buffer("X_min", torch.zeros(input_dim))
        self.register_buffer("X_max", torch.ones(input_dim))
        self.register_buffer("y_min", torch.zeros(1))
        self.register_buffer("y_max", torch.ones(1))
        self.is_fitted = False

        act_cls = nn.Softplus if activation == "softplus" else nn.ReLU
        layers: list[nn.Module] = [nn.Linear(input_dim, neurons_per_layer), act_cls()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(neurons_per_layer, neurons_per_layer), act_cls()]
        layers += [nn.Linear(neurons_per_layer, 1), act_cls()]
        self.network = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def fit_scalers(self, X_raw: torch.Tensor, y_raw: torch.Tensor) -> None:
        self.X_min = X_raw.min(dim=0)[0].detach().clone()
        self.X_max = X_raw.max(dim=0)[0].detach().clone()
        self.y_min = y_raw.min(dim=0)[0].detach().clone()
        self.y_max = y_raw.max(dim=0)[0].detach().clone()
        self.is_fitted = True

    def _normalize_X(self, X_raw: torch.Tensor) -> torch.Tensor:
        return (X_raw - self.X_min) / (self.X_max - self.X_min + 1e-8)

    def _normalize_y(self, y_raw: torch.Tensor) -> torch.Tensor:
        return (y_raw - self.y_min) / (self.y_max - self.y_min + 1e-8)

    def _denormalize_y(self, y_scaled: torch.Tensor) -> torch.Tensor:
        return y_scaled * (self.y_max - self.y_min) + self.y_min

    def forward(self, X_raw: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("fit_scalers must be called before forward.")
        return self._denormalize_y(self.network(self._normalize_X(X_raw)))

    def compute_loss(
        self,
        X_raw: torch.Tensor,
        y_iv_raw: torch.Tensor,
        true_vega: torch.Tensor,
        true_grad_price: torch.Tensor,
        true_div: torch.Tensor,
        derivative_mode: str,
        derivative_loss: str,
        lambda_deriv: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_fitted:
            raise RuntimeError("fit_scalers must be called before training.")

        X_scaled = self._normalize_X(X_raw)
        X_scaled.requires_grad_(lambda_deriv > 0 and derivative_mode != "none")
        y_true_scaled = self._normalize_y(y_iv_raw)
        y_pred_scaled = self.network(X_scaled)
        loss_iv = F.mse_loss(y_pred_scaled, y_true_scaled)

        if lambda_deriv <= 0 or derivative_mode == "none":
            zero = torch.zeros((), dtype=loss_iv.dtype, device=loss_iv.device)
            return loss_iv, loss_iv, zero

        pred_grads_full = torch.autograd.grad(
            outputs=y_pred_scaled,
            inputs=X_scaled,
            grad_outputs=torch.ones_like(y_pred_scaled),
            create_graph=True,
            retain_graph=True,
        )[0]
        pred_grads_norm = pred_grads_full[:, : self.heston_dim]
        delta_theta = self.X_max[: self.heston_dim] - self.X_min[: self.heston_dim] + 1e-8
        delta_iv = self.y_max - self.y_min + 1e-8
        pred_div = pred_grads_norm * (delta_iv / delta_theta)

        if derivative_mode == "vega_weighted":
            # Match Heston_Project/modules/model.py:
            # compare (dIV_pred/dtheta) * Vega / 100 against d_xxx,
            # where d_xxx is the stored price-space gradient (dP/dtheta)/S0.
            pred_term = pred_div * true_vega / 100.0
            true_term = true_grad_price
        elif derivative_mode == "unweighted":
            pred_term = pred_div
            true_term = true_div
        else:
            raise ValueError(f"Unknown derivative_mode: {derivative_mode}")

        if derivative_loss == "smooth_l1":
            loss_deriv = F.smooth_l1_loss(pred_term, true_term)
        elif derivative_loss == "mse":
            loss_deriv = F.mse_loss(pred_term, true_term)
        else:
            raise ValueError(f"Unknown derivative_loss: {derivative_loss}")

        total = loss_iv + lambda_deriv * loss_deriv
        return total, loss_iv, loss_deriv


def feller_violation(theta: Iterable[float]) -> float:
    kappa, lam, sigma = [float(x) for x in list(theta)[:3]]
    return max(0.0, sigma * sigma - 2.0 * kappa * lam)


def load_training_dataframe(train_limit: int | None, seed: int, refresh_cache: bool) -> pd.DataFrame:
    data_path = REPO_ROOT / "data" / "heston_dataset_200k.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing training dataset: {data_path}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "heston_dataset_with_iv_vega_div.csv"
    use_cache = train_limit is None

    if use_cache and cache_path.exists() and not refresh_cache:
        df = pd.read_csv(cache_path)
    else:
        df = pd.read_csv(data_path)
        if train_limit is not None and train_limit > 0 and train_limit < len(df):
            df = df.sample(n=train_limit, random_state=seed).reset_index(drop=True)
        missing = [col for col in FEATURE_COLS + PRICE_GRAD_COLS + ["S0", "K", "Price"] if col not in df.columns]
        if missing:
            raise ValueError(f"Training CSV is missing required columns: {missing}")

        if "iv" not in df.columns or "vega" not in df.columns:
            iv_arr, vega_arr = compute_iv_vega_batch(
                df["Price"].to_numpy(dtype=np.float64),
                df["S0"].to_numpy(dtype=np.float64),
                df["K"].to_numpy(dtype=np.float64),
                df["tau"].to_numpy(dtype=np.float64),
                df["r"].to_numpy(dtype=np.float64),
            )
            df["iv"] = iv_arr
            df["vega"] = vega_arr

        for price_col, div_col in zip(PRICE_GRAD_COLS, DIV_GRAD_COLS):
            if div_col not in df.columns:
                df[div_col] = df[price_col].astype(float) * df["S0"].astype(float) / df["vega"].astype(float)

        clean_cols = FEATURE_COLS + PRICE_GRAD_COLS + DIV_GRAD_COLS + ["iv", "vega", "S0"]
        df[clean_cols] = df[clean_cols].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=clean_cols).copy()
        df = df[(df["iv"] > 0.0) & (df["vega"] > 0.0)].reset_index(drop=True)
        if use_cache:
            df.to_csv(cache_path, index=False)
    return df


def split_training_data(df: pd.DataFrame, seed: int, device: torch.device) -> dict[str, tuple[torch.Tensor, ...]]:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_total = len(df)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    splits = {
        "train": df.iloc[:n_train].copy(),
        "val": df.iloc[n_train : n_train + n_val].copy(),
        "test": df.iloc[n_train + n_val :].copy(),
    }

    out: dict[str, tuple[torch.Tensor, ...]] = {}
    for name, part in splits.items():
        X = torch.tensor(part[FEATURE_COLS].to_numpy(dtype=np.float32), device=device)
        y = torch.tensor(part["iv"].to_numpy(dtype=np.float32), device=device).view(-1, 1)
        vega = torch.tensor(part["vega"].to_numpy(dtype=np.float32), device=device).view(-1, 1)
        price_grad = torch.tensor(part[PRICE_GRAD_COLS].to_numpy(dtype=np.float32), device=device)
        div = torch.tensor(part[DIV_GRAD_COLS].to_numpy(dtype=np.float32), device=device)
        out[name] = (X, y, vega, price_grad, div)
    return out


def checkpoint_path(config: VariantConfig) -> Path:
    return MODELS_DIR / f"variant_{config.key}_hestonloss_v2.pth"


def save_model(model: AblationDDN, config: VariantConfig, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "is_fitted": model.is_fitted,
            "input_dim": model.input_dim,
            "heston_dim": model.heston_dim,
            "activation": model.activation,
            "variant": asdict(config),
            "metadata": metadata,
        },
        path,
    )


def load_model(config: VariantConfig, path: Path, device: torch.device) -> AblationDDN:
    ckpt = torch.load(path, map_location=device)
    model = AblationDDN(
        input_dim=int(ckpt.get("input_dim", 8)),
        heston_param_dim=int(ckpt.get("heston_dim", 5)),
        activation=str(ckpt.get("activation", config.activation)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.is_fitted = bool(ckpt.get("is_fitted", True))
    model.eval()
    return model


def train_or_load_model(
    config: VariantConfig,
    tensors: dict[str, tuple[torch.Tensor, ...]],
    args: argparse.Namespace,
    device: torch.device,
) -> AblationDDN:
    path = checkpoint_path(config)
    if path.exists() and not args.force_train:
        print(f"[{config.key}] Loading existing checkpoint: {path}")
        return load_model(config, path, device)

    print(f"[{config.key}] Training: {config.label}")
    set_seed(args.seed)
    X_train, y_train, vega_train, price_grad_train, div_train = tensors["train"]
    X_val, y_val, _, _, _ = tensors["val"]

    model = AblationDDN(activation=config.activation).to(device)
    model.fit_scalers(X_train, y_train)

    dataset = TensorDataset(X_train, y_train, vega_train, price_grad_train, div_train)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_gamma)

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        lambda_deriv = 0.0 if epoch <= args.warmup_epochs else args.lambda_deriv
        if config.derivative_mode == "none":
            lambda_deriv = 0.0

        model.train()
        total_loss = 0.0
        total_iv = 0.0
        total_deriv = 0.0
        for batch_X, batch_y, batch_vega, batch_price_grad, batch_div in loader:
            optimizer.zero_grad()
            loss, loss_iv, loss_deriv = model.compute_loss(
                batch_X,
                batch_y,
                batch_vega,
                batch_price_grad,
                batch_div,
                derivative_mode=config.derivative_mode,
                derivative_loss=config.derivative_loss,
                lambda_deriv=lambda_deriv,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.item())
            total_iv += float(loss_iv.item())
            total_deriv += float(loss_deriv.item())
        scheduler.step()

        if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
            model.eval()
            with torch.no_grad():
                pred = model(X_val)
                val_mre = torch.mean(torch.abs(pred - y_val) / (torch.abs(y_val) + 1e-8)).item()
            denom = max(1, len(loader))
            print(
                f"[{config.key}] epoch={epoch:04d} "
                f"loss={total_loss / denom:.6e} "
                f"iv={total_iv / denom:.6e} "
                f"deriv={total_deriv / denom:.6e} "
                f"val_mre={100.0 * val_mre:.3f}%"
            )

    metadata = {
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "lambda_deriv": args.lambda_deriv,
        "batch_size": args.batch_size,
        "train_count": int(X_train.shape[0]),
        "elapsed_sec": time.time() - start,
    }
    save_model(model, config, path, metadata)
    return model


def predict_iv(model: AblationDDN, X: torch.Tensor, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            preds.append(model(X[start : start + batch_size]).detach().cpu().numpy())
    return np.concatenate(preds, axis=0).reshape(-1)


def prediction_metrics(model: AblationDDN, tensors: dict[str, tuple[torch.Tensor, ...]]) -> dict[str, float]:
    X_test, y_test, _, _, _ = tensors["test"]
    pred = predict_iv(model, X_test)
    true = y_test.detach().cpu().numpy().reshape(-1)
    valid = np.isfinite(pred) & np.isfinite(true) & (np.abs(true) > 1e-12)
    err = pred[valid] - true[valid]
    rel = np.abs(err) / np.abs(true[valid])
    return {
        "test_iv_mae": float(np.mean(np.abs(err))),
        "test_iv_rmse": float(np.sqrt(np.mean(err * err))),
        "test_iv_mre": float(np.mean(rel)),
    }


def market_dict(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "r": df["r"].to_numpy(dtype=np.float32),
        "tau": df["tau"].to_numpy(dtype=np.float32),
        "S0": df["S0"].to_numpy(dtype=np.float32),
        "K": df["K"].to_numpy(dtype=np.float32),
        "iv_market": df["iv_market"].to_numpy(dtype=np.float32),
        "vega": df["vega"].to_numpy(dtype=np.float32),
    }


def ddn_iv_for_theta(model: AblationDDN, device: torch.device, theta: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    theta_t = torch.tensor(theta, dtype=torch.float32, device=device).unsqueeze(0).expand(n, -1)
    r_t = torch.tensor(df["r"].to_numpy(dtype=np.float32), device=device)
    tau_t = torch.tensor(df["tau"].to_numpy(dtype=np.float32), device=device)
    lks_t = torch.tensor(df["log_K_S0"].to_numpy(dtype=np.float32), device=device)
    X = torch.cat([theta_t, torch.stack([r_t, tau_t, lks_t], dim=1)], dim=1)
    return predict_iv(model, X)


class FlexibleCalibrator:
    def __init__(self, model: AblationDDN, device: torch.device, config: VariantConfig, seed: int) -> None:
        self.model = model
        self.device = device
        self.config = config
        self.rng = np.random.default_rng(seed)

    def _prepare(self, data: dict[str, np.ndarray]) -> None:
        S0 = np.asarray(data["S0"], dtype=np.float64)
        K = np.asarray(data["K"], dtype=np.float64)
        iv = np.asarray(data["iv_market"], dtype=np.float64)
        vega = np.asarray(data["vega"], dtype=np.float64)
        valid = np.isfinite(iv) & np.isfinite(vega) & (iv > 0.0) & (vega > 0.0)
        if valid.sum() == 0:
            raise ValueError("No valid IV/Vega points for calibration.")

        self.r_t = torch.tensor(np.asarray(data["r"], dtype=np.float32)[valid], device=self.device)
        self.tau_t = torch.tensor(np.asarray(data["tau"], dtype=np.float32)[valid], device=self.device)
        self.log_k_s0_t = torch.tensor(np.log(K[valid] / S0[valid]).astype(np.float32), device=self.device)
        self.iv_t = torch.tensor(iv[valid].astype(np.float32), device=self.device).view(-1, 1)
        vega_t = torch.tensor(vega[valid].astype(np.float32), device=self.device).view(-1, 1)
        if self.config.calibration_vega_weighted:
            self.weights_t = vega_t / (vega_t.mean() + 1e-8)
        else:
            self.weights_t = torch.ones_like(vega_t)
        self.M = int(valid.sum())

    def _loss(self, theta: torch.Tensor) -> torch.Tensor:
        theta_exp = theta.unsqueeze(0).expand(self.M, -1)
        mkt = torch.stack([self.r_t, self.tau_t, self.log_k_s0_t], dim=1)
        X = torch.cat([theta_exp, mkt], dim=1)
        pred = self.model(X)
        iv_loss = torch.mean(self.weights_t * (pred - self.iv_t) ** 2)
        if self.config.lambda_feller > 0.0:
            kappa, lam, sigma = theta[0], theta[1], theta[2]
            penalty = torch.relu(sigma * sigma - 2.0 * kappa * lam + 1e-6)
            return iv_loss + self.config.lambda_feller * penalty
        return iv_loss

    def calibrate(
        self,
        data: dict[str, np.ndarray],
        n_starts: int,
        lr: float,
        max_steps: int,
        patience: int,
    ) -> tuple[np.ndarray, float, list[tuple[np.ndarray, float]]]:
        self._prepare(data)
        starts = self.rng.uniform(PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1], size=(n_starts, 5))
        lb = torch.tensor(PARAM_BOUNDS[:, 0], dtype=torch.float32, device=self.device)
        ub = torch.tensor(PARAM_BOUNDS[:, 1], dtype=torch.float32, device=self.device)

        all_results: list[tuple[np.ndarray, float]] = []
        best_theta: np.ndarray | None = None
        best_loss = float("inf")

        for x0 in starts:
            theta = torch.tensor(x0.astype(np.float32), dtype=torch.float32, device=self.device, requires_grad=True)
            optimizer = optim.Adam([theta], lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=patience, min_lr=1e-6
            )
            local_best = x0.copy()
            local_loss = float("inf")
            no_improve = 0

            for _ in range(max_steps):
                optimizer.zero_grad()
                loss = self._loss(theta)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    theta.clamp_(lb, ub)
                loss_value = float(loss.item())
                scheduler.step(loss_value)
                if loss_value < local_loss:
                    local_loss = loss_value
                    local_best = theta.detach().cpu().numpy().astype(np.float64)
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve > patience or optimizer.param_groups[0]["lr"] <= 1.01e-6:
                    break

            all_results.append((local_best, local_loss))
            if local_loss < best_loss:
                best_loss = local_loss
                best_theta = local_best

        if best_theta is None:
            raise RuntimeError("Calibration failed to produce a parameter set.")
        return best_theta, best_loss, all_results


def calc_mre(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(true) & (np.abs(true) > 1e-12)
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(pred[valid] - true[valid]) / np.abs(true[valid])))


def calc_max_re(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(true) & (np.abs(true) > 1e-12)
    if not valid.any():
        return float("nan")
    return float(np.max(np.abs(pred[valid] - true[valid]) / np.abs(true[valid])))


def build_synthetic_chain() -> pd.DataFrame:
    true = np.array([2.0, 0.1, 0.4, -0.6, 0.05], dtype=np.float64)
    true_r = 0.05
    true_S0 = 1000.0
    taus = [0.3, 0.4, 0.5, 0.6, 0.7]
    moneyness = np.linspace(0.6, 1.5, 20)
    rows = []
    for tau in taus:
        for m in moneyness:
            K = true_S0 * m
            price = calculate_heston_price(*true, true_r, tau, true_S0, K)
            if np.isfinite(price) and price > 0.01:
                rows.append({"r": true_r, "tau": tau, "S0": true_S0, "K": K, "P_mkt": price})
    df = pd.DataFrame(rows)
    iv, vega = compute_iv_vega_batch(
        df["P_mkt"].to_numpy(dtype=np.float64),
        df["S0"].to_numpy(dtype=np.float64),
        df["K"].to_numpy(dtype=np.float64),
        df["tau"].to_numpy(dtype=np.float64),
        df["r"].to_numpy(dtype=np.float64),
    )
    df["iv_market"] = iv
    df["vega"] = vega
    df["log_K_S0"] = np.log(df["K"] / df["S0"])
    return df.dropna(subset=["iv_market", "vega"]).query("iv_market > 0 and vega > 0").reset_index(drop=True)


def matched_interest_rate(yield_data: pd.DataFrame, quote_date: str, tau: float) -> float:
    date = pd.to_datetime(quote_date).strftime("%m/%d/%Y")
    row = yield_data[yield_data["date"] == date]
    if row.empty:
        raise ValueError(f"No yield curve row for {date}")
    maturities = np.array([1 / 12, 2 / 12, 3 / 12, 6 / 12, 1.0], dtype=np.float64)
    par_rates = row[["1 mo", "2 mo", "3 mo", "6 mo", "1 yr"]].to_numpy(dtype=np.float64).flatten() / 100.0
    continuous = np.log(1.0 + par_rates * maturities) / maturities
    curve = interp1d(maturities, continuous, kind="cubic", fill_value="extrapolate")
    return float(curve(tau))


def load_and_clean_spy(label: str, n_take: int = 100) -> pd.DataFrame:
    csv_path = REPO_ROOT / "data" / f"spy_{label.replace('-', '_')}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing SPY file: {csv_path}")
    yield_data = pd.read_csv(REPO_ROOT / "data" / "par-yield-curve-rates-2020-2023.csv")

    raw = pd.read_csv(csv_path)
    raw.columns = [c.strip().strip("[]") for c in raw.columns]
    for col in ["UNDERLYING_LAST", "DTE", "STRIKE", "C_LAST", "C_BID", "C_ASK", "C_VOLUME"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    df = raw.dropna(subset=["UNDERLYING_LAST", "DTE", "STRIKE", "C_LAST", "C_BID", "C_ASK"]).copy()
    df = df[df["C_LAST"] > 0.01].copy()
    df = df[df["UNDERLYING_LAST"] > 0].copy()
    df = df[df["C_VOLUME"].fillna(0) > 0].copy()
    df = df[(df["DTE"] >= 40) & (df["DTE"] <= 300)].copy()
    df["S0"] = df["UNDERLYING_LAST"]
    df["K"] = df["STRIKE"]
    df["P_mkt"] = (df["C_BID"] + df["C_ASK"]) * 0.5
    df["tau"] = df["DTE"] / 365.0
    df["r"] = df["tau"].map(lambda t: matched_interest_rate(yield_data, label, float(t)))
    df["log_K_S0"] = np.log(df["K"] / df["S0"])
    df = df[(df["log_K_S0"] >= -0.15) & (df["log_K_S0"] <= 0.20)].copy()

    iv, vega = compute_iv_vega_batch(
        df["P_mkt"].to_numpy(dtype=np.float64),
        df["S0"].to_numpy(dtype=np.float64),
        df["K"].to_numpy(dtype=np.float64),
        df["tau"].to_numpy(dtype=np.float64),
        df["r"].to_numpy(dtype=np.float64),
    )
    df["iv_market"] = iv
    df["vega"] = vega
    out = df.dropna(subset=["iv_market", "vega"]).copy()
    out = out[(out["iv_market"] > 0.0) & (out["vega"] > 0.0)].reset_index(drop=True)

    n_actual = min(n_take, len(out))
    mid = len(out) // 2
    start = max(0, mid - n_actual // 2)
    return out.iloc[start : start + n_actual][
        ["r", "tau", "S0", "K", "P_mkt", "log_K_S0", "iv_market", "vega"]
    ].reset_index(drop=True)


def load_and_clean_nvda(label: str, n_take: int = 100) -> pd.DataFrame:
    selected_path = REPO_ROOT / "experiments result" / f"nvda_selected_{label}.csv"
    if selected_path.exists():
        df = pd.read_csv(selected_path)
        return df[["r", "tau", "S0", "K", "P_mkt", "log_K_S0", "iv_market", "vega"]].head(n_take).reset_index(drop=True)

    csv_path = REPO_ROOT / "data" / f"nvda_{label}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing NVDA file: {csv_path}")
    yield_data = pd.read_csv(REPO_ROOT / "data" / "par-yield-curve-rates-2020-2023.csv")
    raw = pd.read_csv(csv_path)
    raw.columns = [c.strip().strip("[]") for c in raw.columns]
    for col in ["UNDERLYING_LAST", "DTE", "STRIKE", "C_LAST", "C_BID", "C_ASK", "C_VOLUME"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    df = raw.dropna(subset=["UNDERLYING_LAST", "DTE", "STRIKE", "C_LAST", "C_BID", "C_ASK"]).copy()
    df = df[df["C_LAST"] > 0.01].copy()
    df = df[df["UNDERLYING_LAST"] > 0].copy()
    df = df[df["C_VOLUME"].fillna(0) > 0].copy()
    df = df[(df["DTE"] >= 40) & (df["DTE"] <= 300)].copy()
    df["S0"] = df["UNDERLYING_LAST"]
    df["K"] = df["STRIKE"]
    df["P_mkt"] = (df["C_BID"] + df["C_ASK"]) * 0.5
    df["tau"] = df["DTE"] / 365.0
    df["r"] = df["tau"].map(lambda t: matched_interest_rate(yield_data, label, float(t)))
    df["log_K_S0"] = np.log(df["K"] / df["S0"])
    df = df[(df["log_K_S0"] >= -0.15) & (df["log_K_S0"] <= 0.20)].copy()

    iv, vega = compute_iv_vega_batch(
        df["P_mkt"].to_numpy(dtype=np.float64),
        df["S0"].to_numpy(dtype=np.float64),
        df["K"].to_numpy(dtype=np.float64),
        df["tau"].to_numpy(dtype=np.float64),
        df["r"].to_numpy(dtype=np.float64),
    )
    df["iv_market"] = iv
    df["vega"] = vega
    df = df.dropna(subset=["iv_market", "vega"]).copy()
    df = df[(df["iv_market"] > 0.0) & (df["vega"] > 0.0)].copy()
    df["rel_spread"] = (df["C_ASK"] - df["C_BID"]) / df["P_mkt"].replace(0, np.nan)
    df["abs_log_K_S0"] = df["log_K_S0"].abs()
    df = df.sort_values(
        ["abs_log_K_S0", "rel_spread", "C_VOLUME", "DTE", "K"],
        ascending=[True, True, False, True, True],
    ).head(n_take)
    df = df.sort_values(["DTE", "K"]).reset_index(drop=True)
    return df[["r", "tau", "S0", "K", "P_mkt", "log_K_S0", "iv_market", "vega"]]


def run_calibration_case(
    name: str,
    model: AblationDDN,
    config: VariantConfig,
    device: torch.device,
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object]:
    settings = calibration_settings_for_case(name, args)
    calibrator = FlexibleCalibrator(model, device, config, seed=calibration_seed_for_case(name))
    theta, loss, all_results = calibrator.calibrate(
        market_dict(df),
        n_starts=settings.n_starts,
        lr=settings.lr,
        max_steps=settings.max_steps,
        patience=settings.patience,
    )
    pred_iv = ddn_iv_for_theta(model, device, theta, df)
    true_iv = df["iv_market"].to_numpy(dtype=np.float64)
    return {
        "theta": theta,
        "loss": float(loss),
        "all_results": all_results,
        "iv_mre": calc_mre(pred_iv, true_iv),
        "iv_max_re": calc_max_re(pred_iv, true_iv),
        "feller_violation": feller_violation(theta),
    }


def calibration_settings_for_case(name: str, args: argparse.Namespace) -> CalibrationSettings:
    if name == "synthetic":
        defaults = CalibrationSettings(n_starts=10, lr=5e-3, max_steps=500, patience=40)
    elif name in {"spy", "nvda"}:
        defaults = CalibrationSettings(n_starts=10, lr=5e-3, max_steps=300, patience=50)
    else:
        raise ValueError(f"Unknown calibration case: {name}")

    return CalibrationSettings(
        n_starts=defaults.n_starts if args.calibration_starts is None else args.calibration_starts,
        lr=defaults.lr if args.calibration_lr is None else args.calibration_lr,
        max_steps=defaults.max_steps if args.calibration_steps is None else args.calibration_steps,
        patience=defaults.patience if args.calibration_patience is None else args.calibration_patience,
    )


def calibration_seed_for_case(name: str) -> int:
    if name == "synthetic":
        return 0
    if name in {"spy", "nvda"}:
        return 42
    raise ValueError(f"Unknown calibration case: {name}")


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def pct(x: float) -> float:
    return 100.0 * float(x)


def write_latex_table(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Ablation Study of IV-DDN}",
        "\\label{tab:ablation_study}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model Variant & Test IV MRE & Synthetic IV MRE & Same-Day IV MRE & Feller Violation \\% \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(str(row['model_variant']))} & "
            f"{pct(float(row['test_iv_mre'])):.2f} & "
            f"{pct(float(row['synthetic_iv_mre'])):.2f} & "
            f"{pct(float(row['same_day_iv_mre'])):.2f} & "
            f"{pct(float(row['feller_violation_pct'])):.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def best_row(rows: list[dict[str, object]], key: str) -> dict[str, object]:
    valid = [row for row in rows if math.isfinite(float(row[key]))]
    return min(valid, key=lambda row: float(row[key]))


def row_by_variant(rows: list[dict[str, object]], key: str) -> dict[str, object]:
    return next(row for row in rows if row["variant_id"] == key)


def write_analysis(rows: list[dict[str, object]], path: Path) -> None:
    A = row_by_variant(rows, "A")
    B = row_by_variant(rows, "B")
    C = row_by_variant(rows, "C")
    D = row_by_variant(rows, "D")
    E = row_by_variant(rows, "E")
    best_same = best_row(rows, "same_day_iv_mre")
    best_feller = best_row(rows, "feller_violation_pct")

    def value(row: dict[str, object], key: str) -> str:
        return f"{pct(float(row[key])):.2f}\\%"

    text = rf"""\subsection{{Ablation Study}}

Table~\ref{{tab:ablation_study}} reports the ablation results for the main components of the IV-DDN calibration pipeline. The IV-only baseline (Variant A) obtains a test IV MRE of {value(A, "test_iv_mre")} and a same-day IV MRE of {value(A, "same_day_iv_mre")}. Adding derivative supervision without Vega weighting (Variant B) changes these values to {value(B, "test_iv_mre")} and {value(B, "same_day_iv_mre")}, respectively. This comparison suggests that derivative learning can affect both interpolation accuracy and calibration behavior, although the direction and magnitude of the effect should be interpreted together with the calibration objective.

The Vega-weighted derivative variant (Variant C) records a test IV MRE of {value(C, "test_iv_mre")}, a synthetic IV MRE of {value(C, "synthetic_iv_mre")}, and a same-day IV MRE of {value(C, "same_day_iv_mre")}. Relative to Variant B, these results indicate whether weighting derivative information by Vega improves stability in the regions of the option surface that carry stronger price sensitivity. In this run, the lowest same-day IV MRE is obtained by {latex_escape(str(best_same["model_variant"]))} at {value(best_same, "same_day_iv_mre")}.

Introducing the Feller penalty in Variant D gives a Feller violation rate of {value(D, "feller_violation_pct")}, compared with {value(C, "feller_violation_pct")} for Variant C. This comparison directly measures the role of the feasibility regularizer during calibration. The smallest violation rate in the table is achieved by {latex_escape(str(best_feller["model_variant"]))} with {value(best_feller, "feller_violation_pct")}. These values should be read as evidence about calibrated parameter feasibility rather than as a guarantee that every optimization path remains feasible.

Variant E replaces the Huber derivative loss with an MSE derivative loss. Its test IV MRE is {value(E, "test_iv_mre")} and its same-day IV MRE is {value(E, "same_day_iv_mre")}, while Variant D gives {value(D, "test_iv_mre")} and {value(D, "same_day_iv_mre")}. The comparison is useful for assessing whether the robustness of the Huber loss matters when derivative labels contain large local sensitivities.
"""
    path.write_text(text + "\n", encoding="utf-8")


def write_config(args: argparse.Namespace) -> None:
    serializable = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (RESULTS_DIR / "run_config.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IV-DDN ablation study.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--lambda-deriv", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--decay-step", type=int, default=20)
    parser.add_argument("--decay-gamma", type=float, default=0.9)
    parser.add_argument("--train-limit", type=int, default=0, help="0 means use all available rows.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--calibration-starts", type=int, default=None)
    parser.add_argument("--calibration-steps", type=int, default=None)
    parser.add_argument("--calibration-patience", type=int, default=None)
    parser.add_argument("--calibration-lr", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    write_config(args)
    set_seed(args.seed)
    device = pick_device(args.device)
    train_limit = None if args.train_limit == 0 else args.train_limit

    print(f"Repository root: {REPO_ROOT}")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Device: {device}")

    df_train_all = load_training_dataframe(train_limit=train_limit, seed=args.seed, refresh_cache=args.refresh_cache)
    print(f"Training/evaluation rows: {len(df_train_all):,}")
    tensors = split_training_data(df_train_all, seed=args.seed, device=device)

    synthetic_df = build_synthetic_chain()
    spy_df = load_and_clean_spy("2022-09-02", n_take=100)
    nvda_df = load_and_clean_nvda("2021-11-02", n_take=100)
    print(f"Synthetic contracts: {len(synthetic_df)}")
    print(f"SPY same-day contracts: {len(spy_df)}")
    print(f"NVDA same-day contracts: {len(nvda_df)}")

    rows: list[dict[str, object]] = []
    for config in VARIANTS:
        model = train_or_load_model(config, tensors, args, device)
        pred_metrics = prediction_metrics(model, tensors)

        print(f"[{config.key}] Synthetic calibration")
        synthetic = run_calibration_case("synthetic", model, config, device, synthetic_df, args)
        print(f"[{config.key}] SPY same-day calibration")
        spy = run_calibration_case("spy", model, config, device, spy_df, args)
        print(f"[{config.key}] NVDA same-day calibration")
        nvda = run_calibration_case("nvda", model, config, device, nvda_df, args)

        feller_values = [
            float(synthetic["feller_violation"]),
            float(spy["feller_violation"]),
            float(nvda["feller_violation"]),
        ]
        feller_violation_pct = float(np.mean([v > 0.0 for v in feller_values]))
        feller_violation_avg = float(np.mean(feller_values))
        same_day_mre = float(np.mean([float(spy["iv_mre"]), float(nvda["iv_mre"])]))

        row = {
            "variant_id": config.key,
            "model_variant": config.label,
            "derivative_mode": config.derivative_mode,
            "derivative_loss": config.derivative_loss,
            "activation": config.activation,
            "calibration_vega_weighted": config.calibration_vega_weighted,
            "lambda_feller": config.lambda_feller,
            **pred_metrics,
            "synthetic_iv_mre": float(synthetic["iv_mre"]),
            "synthetic_iv_max_re": float(synthetic["iv_max_re"]),
            "spy_native_iv_mre": float(spy["iv_mre"]),
            "nvda_native_iv_mre": float(nvda["iv_mre"]),
            "same_day_iv_mre": same_day_mre,
            "synthetic_feller_violation": float(synthetic["feller_violation"]),
            "spy_feller_violation": float(spy["feller_violation"]),
            "nvda_feller_violation": float(nvda["feller_violation"]),
            "feller_violation_pct": feller_violation_pct,
            "feller_violation_avg": feller_violation_avg,
            "synthetic_theta": json.dumps(np.asarray(synthetic["theta"]).tolist()),
            "spy_theta": json.dumps(np.asarray(spy["theta"]).tolist()),
            "nvda_theta": json.dumps(np.asarray(nvda["theta"]).tolist()),
        }
        rows.append(row)

        pd.DataFrame(rows).to_csv(RESULTS_DIR / "ablation_results.csv", index=False)
        write_latex_table(rows, RESULTS_DIR / "ablation_table.tex")
        if len(rows) == len(VARIANTS):
            write_analysis(rows, RESULTS_DIR / "ablation_analysis.tex")
        print(
            f"[{config.key}] test_mre={100.0 * row['test_iv_mre']:.3f}% "
            f"synthetic_mre={100.0 * row['synthetic_iv_mre']:.3f}% "
            f"same_day_mre={100.0 * row['same_day_iv_mre']:.3f}% "
            f"feller_viol={100.0 * row['feller_violation_pct']:.2f}%"
        )

    out_csv = RESULTS_DIR / "ablation_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    write_latex_table(rows, RESULTS_DIR / "ablation_table.tex")
    write_analysis(rows, RESULTS_DIR / "ablation_analysis.tex")
    print(f"Saved: {out_csv}")
    print(f"Saved: {RESULTS_DIR / 'ablation_table.tex'}")
    print(f"Saved: {RESULTS_DIR / 'ablation_analysis.tex'}")


if __name__ == "__main__":
    main()
