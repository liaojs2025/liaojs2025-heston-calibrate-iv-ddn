"""
pricing.py
QuantLib Heston 定价、BS 隐含波动率 / Vega 计算、LHS 训练数据生成。

升级点:
  1. 新增 compute_iv_vega_batch() — 利用 py_vollib_vectorized 向量化计算 IV 与 Vega
  2. 新增 check_feller() — 验证 Feller 条件 2κθ > σ²
  3. generate_training_data() 目标从 price_norm → IV，并过滤不满足 Feller 条件的样本
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import QuantLib as ql
from scipy.stats import qmc

# ── py_vollib_vectorized：向量化 BS IV & Vega ───────────────────────
import py_vollib_vectorized as pvv


# =====================================================================
#  BS 隐含波动率 & Vega 向量化计算
# =====================================================================

def compute_iv_vega_batch(
    price: np.ndarray,
    S0: np.ndarray,
    K: np.ndarray,
    tau: np.ndarray,
    r: np.ndarray,
    flag: str = "c",
) -> tuple[np.ndarray, np.ndarray]:
    """
    利用 py_vollib_vectorized 从期权价格批量反算 BS 隐含波动率 (IV) 和 Vega。

    参数:
      price : 期权价格数组
      S0    : 标的价格数组
      K     : 行权价数组
      tau   : 剩余到期时间（年化）
      r     : 无风险利率
      flag  : 'c' = Call, 'p' = Put

    返回:
      iv   : BS 隐含波动率数组，计算失败处为 NaN
      vega : BS Vega 数组（期权价格对 IV 的敏感度），NaN 处同为 NaN

    注意: 违反无套利下界 (price < max(S-Ke^{-rτ},0)) 的期权将返回 NaN。
    """
    # py_vollib_vectorized 使用 pandas Series 接口
    price_s = pd.Series(price.astype(np.float64))
    S_s     = pd.Series(S0.astype(np.float64))
    K_s     = pd.Series(K.astype(np.float64))
    tau_s   = pd.Series(tau.astype(np.float64))
    r_s     = pd.Series(r.astype(np.float64))
    flag_s  = pd.Series([flag] * len(price))

    # ── 计算 IV（向量化，内部使用 Let's Be Rational 算法）
    iv_s = pvv.vectorized_implied_volatility(
        price_s, S_s, K_s, tau_s, r_s, flag_s,
        q=0.0,  # 股息率
        model="black_scholes_merton",
        return_as="series",
    )

    # ── 计算 Vega（∂Price/∂σ）——使用顶层 vectorized_vega 函数
    # 参数 sigma 传入刚刚计算出的 IV
    vega_s = pvv.vectorized_vega(
        flag_s, S_s, K_s, tau_s, r_s, iv_s,
        q=0.0,
        model="black_scholes_merton",
        return_as="series",
    )

    iv_arr   = iv_s.values.astype(np.float64)
    vega_arr = vega_s.values.astype(np.float64)
    return iv_arr, vega_arr


# =====================================================================
#  Feller 条件验证
# =====================================================================

def check_feller(kappa: float, theta: float, sigma: float) -> bool:
    """
    检查 Heston 模型 Feller 条件：2κθ > σ²
    满足 Feller 条件可保证方差过程 v(t) 始终为正。
    """
    return 2.0 * kappa * theta > sigma ** 2


# =====================================================================
#  QuantLib 单点定价
# =====================================================================

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

        spot_handle  = ql.QuoteHandle(ql.SimpleQuote(float(S0)))
        flat_ts      = ql.YieldTermStructureHandle(ql.FlatForward(today, float(r),   day_count))
        dividend_ts  = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, day_count))

        process = ql.HestonProcess(flat_ts, dividend_ts, spot_handle,
                                   float(v0), float(kappa), float(theta),
                                   float(sigma), float(rho))
        model   = ql.HestonModel(process)
        engine  = ql.AnalyticHestonEngine(model)

        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Call, float(K)),
            ql.EuropeanExercise(maturity_date),
        )
        option.setPricingEngine(engine)
        return option.NPV()
    except Exception:
        return np.nan


# =====================================================================
#  数值梯度（对 5 个 Heston 参数）
# =====================================================================

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


# =====================================================================
#  IV 空间数值梯度（对 5 个 Heston 参数求 ∂IV/∂θ_i）
# =====================================================================

def calculate_iv_numerical_gradients(
    params: list[float], bump: float = 1e-4
) -> list[float]:
    """
    中心差分计算 IV 对前5个 Heston 参数的偏导数。
    params: [kappa, lambda, sigma, rho, v0, r, tau, S0, K]
    返回: [dIV/dkappa, dIV/dlambda, dIV/dsigma, dIV/drho, dIV/dv0]

    计算方式：先算 bump 前后的 price，再用 compute_iv_vega_batch 算 IV，中心差分。
    """
    kappa, theta, sig, rho, v0, r, tau, S0, K = params
    base_price = calculate_heston_price(*params)
    if np.isnan(base_price) or base_price <= 0:
        return [np.nan] * 5

    grads = []
    for i in range(5):
        up, dn = list(params), list(params)
        up[i] += bump
        dn[i] -= bump
        try:
            p_up = calculate_heston_price(*up)
            p_dn = calculate_heston_price(*dn)
            if np.isnan(p_up) or np.isnan(p_dn) or p_up <= 0 or p_dn <= 0:
                grads.append(np.nan)
                continue
            # 将两个价格转换为 IV
            prices = np.array([p_up, p_dn])
            S_arr  = np.array([S0, S0])
            K_arr  = np.array([K, K])
            tau_arr = np.array([tau, tau])
            r_arr  = np.array([r, r])
            iv_arr, _ = compute_iv_vega_batch(prices, S_arr, K_arr, tau_arr, r_arr)
            iv_up, iv_dn = iv_arr[0], iv_arr[1]
            if np.isnan(iv_up) or np.isnan(iv_dn):
                grads.append(np.nan)
            else:
                grads.append((iv_up - iv_dn) / (2 * bump))
        except Exception:
            grads.append(np.nan)
    return grads


# =====================================================================
#  训练数据生成（IV 目标 + Feller 条件过滤）
# =====================================================================

def generate_training_data(num_samples: int = 5) -> pd.DataFrame:
    """
    LHS 采样生成训练数据集 (IV 目标版本)。

    输入特征顺序（8维，利用期权定价齐次性移除 S0）:
      kappa, lambda, sigma, rho, v0, r, tau, log_K_S0

    利用期权定价的齐次性:
      BS 隐含波动率仅取决于 log-moneyness log(K/S0)，而不取决于 S0 的绝对水平。
      因此 S0 不再作为网络输入特征。QuantLib 定价时固定 S0=100。

    目标列:
      iv   — BS 隐含波动率（由 Heston 价格反算）
      vega — BS Vega（用于校准阶段加权）

    梯度标签（5维，IV 空间）:
      div_kappa, div_lambda, div_sigma, div_rho, div_v0

    保留列: S0, K, Price, price_norm（兼容旧代码）

    数据质量保证:
      1. 过滤不满足 Feller 条件 (2κθ > σ²) 的参数组合
      2. 过滤 IV 计算为 NaN 的脏数据（违反无套利边界）
      3. 过滤 Vega ≤ 0 的数据（极端深度 OTM/ITM）
    """
    # 参数边界：8维（移除 S0，利用齐次性）
    # kappa, lambda, sigma, rho, v0, r, tau, log_K_S0
    l_bounds = [0.01, 0.01, 0.10, -0.90, 0.01, 0.001, 0.05, -1.0]
    u_bounds = [5.00, 1.00, 1.00, -0.05, 1.00, 0.100, 1.00,  1.0]

    # 固定 S0=100（IV 不依赖 S0 绝对水平）
    S0_FIXED = 100.0

    sampler  = qmc.LatinHypercube(d=8, seed=42)
    samples  = qmc.scale(sampler.random(n=num_samples), l_bounds, u_bounds)

    dataset: list[dict] = []
    failed = 0
    feller_reject = 0

    for idx, row in enumerate(samples):
        kappa, theta, sigma, rho, v0, r, tau, log_k_s0 = row
        S0 = S0_FIXED  # 固定 S0=100，IV 不依赖 S0 绝对水平

        # ── Feller 条件过滤：2κθ > σ²，不满足则跳过 ──
        if not check_feller(kappa, theta, sigma):
            feller_reject += 1
            failed += 1
            continue

        K      = S0 * np.exp(log_k_s0)
        params = [kappa, theta, sigma, rho, v0, r, tau, S0, K]

        price = calculate_heston_price(*params)
        if np.isnan(price) or price <= 0:
            failed += 1
            continue

        # ── 计算 BS 隐含波动率和 Vega ──
        iv_arr, vega_arr = compute_iv_vega_batch(
            np.array([price]), np.array([S0]), np.array([K]),
            np.array([tau]),   np.array([r]),
        )
        iv_val   = iv_arr[0]
        vega_val = vega_arr[0]

        # 过滤 IV 为 NaN（违反无套利边界）或 Vega ≤ 0 的脏数据
        if np.isnan(iv_val) or np.isnan(vega_val) or iv_val <= 0 or vega_val <= 0:
            failed += 1
            continue

        # ── 计算 IV 空间的数值梯度 dIV/dθ_i ──
        iv_grads = calculate_iv_numerical_gradients(params)
        if any(np.isnan(g) for g in iv_grads):
            failed += 1
            continue

        # ── 同时保留价格空间的梯度（兼容） ──
        price_grads = calculate_numerical_gradients(params)

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
            # 备用列
            "K":        K,
            "Price":    price,
            "price_norm": price / S0,
            # ===== 新目标: IV 和 Vega =====
            "iv":       iv_val,          # BS 隐含波动率（训练目标）
            "vega":     vega_val,        # BS Vega（校准阶段加权用）
            # IV 空间梯度标签
            "div_kappa":  iv_grads[0],
            "div_lambda": iv_grads[1],
            "div_sigma":  iv_grads[2],
            "div_rho":    iv_grads[3],
            "div_v0":     iv_grads[4],
            # 价格空间梯度标签（兼容旧代码）⚠️注意此处除以S0做了nomalization
            "d_kappa":  price_grads[0] / S0 if not np.isnan(price_grads[0]) else np.nan,
            "d_lambda": price_grads[1] / S0 if not np.isnan(price_grads[1]) else np.nan,
            "d_sigma":  price_grads[2] / S0 if not np.isnan(price_grads[2]) else np.nan,
            "d_rho":    price_grads[3] / S0 if not np.isnan(price_grads[3]) else np.nan,
            "d_v0":     price_grads[4] / S0 if not np.isnan(price_grads[4]) else np.nan,
        })

        if (idx + 1) % 200 == 0:
            print(f"  [{idx+1:>7d}/{num_samples}] 有效 {len(dataset)}，"
                  f"失败 {failed}（Feller 拒绝 {feller_reject}）")

    df = pd.DataFrame(dataset)
    # 最终清洗：移除任何含 NaN 的行
    df = df.dropna().reset_index(drop=True)
    print(f"  完成: 有效 {len(df):,} / {num_samples:,}，失败 {failed}"
          f"（Feller 拒绝 {feller_reject}）")
    return df
