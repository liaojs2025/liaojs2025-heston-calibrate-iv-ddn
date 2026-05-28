"""
pricing.py
QuantLib Heston 定价、数值梯度计算、LHS 训练数据生成。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import QuantLib as ql
from scipy.stats import qmc


# ── 单点定价 ─────────────────────────────────────────────────────────

def calculate_heston_price(
    kappa: float, theta: float, sigma: float, rho: float, v0: float,
    r: float, tau: float, S0: float, K: float,
) -> float:
    """使用 QuantLib 计算 Heston 模型下欧式看涨期权价格。失败返回 np.nan。"""
    if kappa <= 0 or theta < 0 or sigma <= 0 or v0 < 0 or tau <= 0 or S0 <= 0 or K <= 0:
        return np.nan
    if rho <= -1 or rho >= 1:
        return np.nan
    try:
        today = ql.Date(1, 1, 2024)
        ql.Settings.instance().evaluationDate = today
        day_count = ql.Actual365Fixed()
        maturity_date = today + max(1, int(tau * 365))

        spot_handle  = ql.QuoteHandle(ql.SimpleQuote(S0))
        flat_ts      = ql.YieldTermStructureHandle(ql.FlatForward(today, r,   day_count))
        dividend_ts  = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, day_count))

        process = ql.HestonProcess(flat_ts, dividend_ts, spot_handle, v0, kappa, theta, sigma, rho)
        model   = ql.HestonModel(process)
        engine  = ql.AnalyticHestonEngine(model)

        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Call, K),
            ql.EuropeanExercise(maturity_date),
        )
        option.setPricingEngine(engine)
        return option.NPV()
    except Exception:
        return np.nan


# ── 数值梯度 ─────────────────────────────────────────────────────────

def calculate_numerical_gradients(
    params: list[float], bump: float = 1e-4
) -> list[float]:
    """
    中心差分计算 Price 对前5个 Heston 参数的偏导数。
    params: [kappa, lambda, sigma, rho, v0, r, tau, S0, K]
    返回: [d/dkappa, d/dlambda, d/dsigma, d/drho, d/dv0]，失败处返回 np.nan
    """
    grads: list[float] = []
    if np.isnan(calculate_heston_price(*params)):
        return [np.nan] * 5

    for i in range(5):
        up, dn = list(params), list(params)
        up[i] += bump
        dn[i] -= bump
        try:
            p_up = calculate_heston_price(*up)
            p_dn = calculate_heston_price(*dn)
            grads.append(np.nan if (np.isnan(p_up) or np.isnan(p_dn))
                         else (p_up - p_dn) / (2 * bump))
        except Exception:
            grads.append(np.nan)
    return grads


# ── 训练数据生成 ──────────────────────────────────────────────────────

def generate_training_data(num_samples: int = 5) -> pd.DataFrame:
    """
    LHS 采样生成训练数据集。

    输入特征顺序（9维）:
      kappa, lambda, sigma, rho, v0, r, tau, S0, log_K_S0

    目标列:
      price_norm = Price / S0         （一阶齐次化）

    梯度标签（5维，已除以 S0）:
      d_kappa, d_lambda, d_sigma, d_rho, d_v0

    备用列: K（绝对行权价）, Price（绝对期权价格）
    """
    # 参数边界；第9维直接采样 log(K/S0) ∈ [-1, 1]
    l_bounds = [0.01, 0.01, 0.10, -0.90, 0.01, 0.001, 0.05,   10.0, -1.0]
    u_bounds = [5.00, 1.00, 1.00, -0.05, 1.00, 0.100, 1.00, 6000.0,  1.0]

    sampler  = qmc.LatinHypercube(d=9, seed=42)
    samples  = qmc.scale(sampler.random(n=num_samples), l_bounds, u_bounds)

    dataset: list[dict] = []
    failed = 0

    for idx, row in enumerate(samples):
        kappa, theta, sigma, rho, v0, r, tau, S0, log_k_s0 = row
        K      = S0 * np.exp(log_k_s0)
        params = [kappa, theta, sigma, rho, v0, r, tau, S0, K]

        price = calculate_heston_price(*params)
        if np.isnan(price) or price <= 0:
            failed += 1
            continue

        grads = calculate_numerical_gradients(params)
        if any(np.isnan(g) for g in grads):
            failed += 1
            continue

        dataset.append({
            # 输入特征
            "kappa":    kappa,
            "lambda":   theta,
            "sigma":    sigma,
            "rho":      rho,
            "v0":       v0,
            "r":        r,
            "tau":      tau,
            "S0":       S0,
            "log_K_S0": log_k_s0,
            # 备用
            "K":        K,
            "Price":    price,
            # 目标
            "price_norm": price / S0,
            # 梯度标签（已 /S0）
            "d_kappa":  grads[0] / S0,
            "d_lambda": grads[1] / S0,
            "d_sigma":  grads[2] / S0,
            "d_rho":    grads[3] / S0,
            "d_v0":     grads[4] / S0,
        })

        if (idx + 1) % 200 == 0:
            print(f"  [{idx+1:>7d}/{num_samples}] 有效 {len(dataset)}，失败 {failed}")

    df = pd.DataFrame(dataset)
    print(f"  完成: 有效 {len(df):,} / {num_samples:,}，失败 {failed}")
    return df
