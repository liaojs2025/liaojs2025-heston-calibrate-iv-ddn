"""
calibration.py
HestonCalibrator — Multi-start Adam 校准器（IV 空间 + Feller 约束版本）。

升级说明:
  1. 校准目标从 price MSE 改为 Vega 加权的 IV MSE
  2. 引入 PINN 思想：Feller 条件 (2κθ > σ²) 作为惩罚项加入损失函数
  3. 校准参数 kappa, lambda(theta), sigma 的梯度可正确反向传播 Feller penalty
"""

from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim


class HestonCalibrator:
    """
    Multi-start Adam 校准器 (IV 空间 + Feller 约束)。

    策略：
      1. 在参数可行域内随机生成 n_starts 个初始点
      2. 每个初始点独立运行 Adam + ReduceLROnPlateau + Box Projection
      3. 损失函数 = Vega 加权 IV MSE + lambda_feller * Feller 惩罚
      4. 取所有起始点中最终 loss 最小的结果输出

    IV 接口：
      - 第8个输入特征: log(K/S0)（利用齐次性，S0 不作为网络输入）
      - DDN 输出: IV (隐含波动率)
      - 校准目标: 使 DDN 预测的 IV 拟合市场 IV
    """

    # Heston 参数可行域（与训练数据边界一致）
    PARAM_BOUNDS = np.array([
        [0.01, 5.00],    # kappa
        [0.01, 1.00],    # lambda (theta)
        [0.10, 1.00],    # sigma
        [-0.90, -0.05],  # rho
        [0.01, 1.00],    # v0
    ])  # shape (5, 2)

    def __init__(self, model, device, seed: int = 42):
        self.model  = model
        self.device = device
        self.model.eval()
        self.rng = np.random.default_rng(seed)

    # ── Feller 惩罚项（PINN 约束） ───────────────────────────────────

    @staticmethod
    def feller_penalty(theta_t: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """
        Feller 条件惩罚项：当 σ² > 2κθ 时产生正向 Loss，满足条件时惩罚为 0。

        Feller 条件要求: 2κθ > σ²  (保证方差过程 v(t) 严格为正)
        惩罚定义: penalty = ReLU(σ² - 2κθ + ε)

        参数:
          theta_t : Tensor (5,)，顺序 [kappa, lambda(theta), sigma, rho, v0]
          epsilon : 小正数，增加一点 margin 使优化器倾向于严格满足条件

        返回:
          标量 Tensor，Feller 惩罚值
          - 满足 2κθ > σ² 时返回 ≈ 0
          - 违反时返回 σ² - 2κθ + ε > 0

        注意:
          kappa, lambda, sigma 均为 nn.Parameter / requires_grad=True，
          ReLU 对其导数是分段线性的，梯度可以正确反向传播。
        """
        kappa  = theta_t[0]    # κ: 均值回复速度
        lam    = theta_t[1]    # θ (lambda): 长期方差均值
        sigma  = theta_t[2]    # σ: 波动率的波动率 (vol of vol)
        # penalty = max(0, σ² - 2κθ + ε)
        return torch.relu(sigma ** 2 - 2.0 * kappa * lam + epsilon)

    # ── 市场数据预处理 ────────────────────────────────────────────────

    def _prepare_market_tensors(self, market_data: dict) -> None:
        """
        将市场数据转为 Tensor 并缓存在 self 上。

        market_data keys:
          'r', 'tau', 'S0', 'K'   — 市场参数（绝对值数组）
          'iv_market'              — 市场 BS 隐含波动率数组（新增）
          'vega'                   — 市场 BS Vega 数组（新增，用于加权）

        兼容旧接口: 若包含 'P_mkt' 但无 'iv_market'，则自动计算。
        """
        S0_arr = np.asarray(market_data["S0"], dtype=np.float64)
        K_arr  = np.asarray(market_data["K"],  dtype=np.float64)

        def _t(arr):
            return torch.tensor(arr, dtype=torch.float32, device=self.device)

        self.r_t        = _t(np.asarray(market_data["r"],   dtype=np.float32))
        self.tau_t      = _t(np.asarray(market_data["tau"], dtype=np.float32))
        self.S0_t       = _t(S0_arr)
        self.log_k_s0_t = _t(np.log(K_arr / S0_arr))
        self.M          = len(S0_arr)

        # ── 处理 IV 和 Vega ──
        if "iv_market" in market_data and "vega" in market_data:
            # 直接使用传入的 IV 和 Vega
            iv_arr   = np.asarray(market_data["iv_market"], dtype=np.float64)
            vega_arr = np.asarray(market_data["vega"],      dtype=np.float64)
        elif "P_mkt" in market_data:
            # 兼容旧接口: 从期权价格自动计算 IV 和 Vega
            from modules.pricing import compute_iv_vega_batch
            P_arr = np.asarray(market_data["P_mkt"], dtype=np.float64)
            iv_arr, vega_arr = compute_iv_vega_batch(
                P_arr, S0_arr, K_arr,
                np.asarray(market_data["tau"], dtype=np.float64),
                np.asarray(market_data["r"],   dtype=np.float64),
            )
        else:
            raise ValueError("market_data 必须包含 'iv_market'+'vega' 或 'P_mkt'")

        # ── 清洗: 去除 IV 或 Vega 为 NaN/≤0 的期权 ──
        valid = np.isfinite(iv_arr) & np.isfinite(vega_arr) & (iv_arr > 0) & (vega_arr > 0)
        if valid.sum() < self.M:
            n_drop = self.M - valid.sum()
            print(f"  ⚠️  Vega/IV 清洗: 丢弃 {n_drop} 条无效期权，保留 {valid.sum()} 条")
            self.r_t        = self.r_t[valid]
            self.tau_t      = self.tau_t[valid]
            self.S0_t       = self.S0_t[valid]
            self.log_k_s0_t = self.log_k_s0_t[valid]
            iv_arr   = iv_arr[valid]
            vega_arr = vega_arr[valid]
            self.M   = int(valid.sum())

        # 存储市场 IV (目标) 和 Vega 权重
        self.iv_mkt_t = _t(iv_arr).view(-1, 1)            # (M, 1) 市场 IV

        # ── Vega 加权系数: 归一化 Vega 使权重和为 M ──
        # 这样加权后的 MSE 量纲与未加权时一致，方便调参
        vega_t = _t(vega_arr).view(-1, 1)                  # (M, 1)
        self.vega_weights_t = vega_t / (vega_t.mean() + 1e-8)  # 归一化 Vega 权重

    # ── 前向损失（Vega 加权 IV MSE + Feller 惩罚） ────────────────────

    def _forward_loss(
        self,
        theta_t: torch.Tensor,
        lambda_feller: float = 10.0,
    ) -> torch.Tensor:
        """
        校准损失函数（核心重构）:

        Calib_Loss = Mean( Vega_weights * (DDN_IV − Market_IV)² )
                   + lambda_feller * feller_penalty

        参数:
          theta_t       : Tensor (5,)，当前 Heston 参数猜测值
          lambda_feller : Feller 惩罚权重（默认 10.0）

        返回:
          标量 loss Tensor（梯度可回传至 theta_t 中的 kappa, lambda, sigma）
        """
        # ── 构建 DDN 输入（8维: 5个Heston参数 + r, tau, log_K_S0） ──
        theta_exp = theta_t.unsqueeze(0).expand(self.M, -1)          # (M, 5)
        mkt_feat  = torch.stack(
            [self.r_t, self.tau_t, self.log_k_s0_t], dim=1
        )                                                             # (M, 3)
        X_input   = torch.cat([theta_exp, mkt_feat], dim=1)          # (M, 8)

        # ── DDN 预测 IV ──
        iv_pred = self.model(X_input)                                 # (M, 1)

        # ── Vega 加权 IV MSE ──
        # 含义: IV 误差按 Vega 大小加权，ATM 附近（Vega 大）权重高，
        #        深度 OTM（Vega 小）权重低，符合金融直觉
        iv_mse = torch.mean(self.vega_weights_t * (iv_pred - self.iv_mkt_t) ** 2)

        # ── Feller 惩罚项（PINN 约束） ──
        # 确保校准出的 (kappa, lambda, sigma) 满足 2κθ > σ²
        f_penalty = self.feller_penalty(theta_t)

        return iv_mse + lambda_feller * f_penalty

    # ── 单次 Adam 优化 ────────────────────────────────────────────────

    def _run_single_adam(
        self,
        x0_np: np.ndarray,
        lr: float = 5e-3,
        max_steps: int = 500,
        patience: int = 40,
        min_lr: float = 1e-6,
        lambda_feller: float = 10.0,
        verbose_prefix: str = "",
    ) -> tuple[np.ndarray, float]:
        """
        从 x0_np 出发运行一次 Adam 优化。
        返回 (best_theta_np, best_loss)。
        """
        lb = torch.tensor(self.PARAM_BOUNDS[:, 0], dtype=torch.float32, device=self.device)
        ub = torch.tensor(self.PARAM_BOUNDS[:, 1], dtype=torch.float32, device=self.device)

        theta = torch.tensor(x0_np, dtype=torch.float32,
                             requires_grad=True, device=self.device)
        opt = optim.Adam([theta], lr=lr)
        sch = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=patience, min_lr=min_lr
        )

        best_loss  = float("inf")
        best_theta = x0_np.copy()
        no_improve = 0

        for step in range(1, max_steps + 1):
            opt.zero_grad()
            loss = self._forward_loss(theta, lambda_feller=lambda_feller)
            loss.backward()
            opt.step()

            # Box projection：将参数强制约束在可行域内
            with torch.no_grad():
                theta.clamp_(lb, ub)

            loss_val = loss.item()
            sch.step(loss_val)

            if loss_val < best_loss:
                best_loss  = loss_val
                best_theta = theta.detach().cpu().numpy().copy()
                no_improve = 0
            else:
                no_improve += 1

            # 连续 patience 步无改善 → 提前停止
            if no_improve > patience:
                break
            # 学习率已降至最小 → 提前停止
            if opt.param_groups[0]["lr"] <= min_lr * 1.01:
                break

            if verbose_prefix and step % 100 == 0:
                # 打印 Feller 条件状态
                with torch.no_grad():
                    fp = self.feller_penalty(theta).item()
                    feller_ok = "✓" if fp < 1e-6 else "✗"
                print(f"  {verbose_prefix} step={step:4d} | "
                      f"loss={loss_val:.4e} | "
                      f"Feller={feller_ok}({fp:.2e}) | "
                      f"lr={opt.param_groups[0]['lr']:.2e}")

        return best_theta, best_loss

    # ── 主入口 ────────────────────────────────────────────────────────

    def calibrate(
        self,
        market_data: dict,
        n_starts: int       = 10,
        lr: float           = 5e-3,
        max_steps: int      = 500,
        patience: int       = 40,
        lambda_feller: float = 10.0,
        verbose: bool       = True,
    ) -> tuple[np.ndarray, float, list]:
        """
        Multi-start Adam 校准（IV 空间 + Feller 约束）。

        参数:
          market_data    : dict，key = 'r','tau','S0','K' + ('iv_market','vega' 或 'P_mkt')
          n_starts       : 随机初始点数量（默认 10）
          lr             : Adam 初始学习率（默认 5e-3）
          max_steps      : 每个起始点的最大步数（默认 500）
          patience       : 无改善早停阈值（默认 40）
          lambda_feller  : Feller 惩罚权重（默认 10.0）
          verbose        : 是否打印进度

        返回:
          best_theta  : np.ndarray (5,)，最优 Heston 参数
          best_loss   : float，最终 Vega 加权 IV MSE + Feller 惩罚
          all_results : list of (theta, loss)，所有起始点结果
        """
        self._prepare_market_tensors(market_data)

        lo, hi = self.PARAM_BOUNDS[:, 0], self.PARAM_BOUNDS[:, 1]
        starts = self.rng.uniform(lo, hi, size=(n_starts, 5)).astype(np.float64)

        if verbose:
            iv_mkt = self.iv_mkt_t.cpu().numpy()
            print(f"  市场期权数   : {self.M}")
            print(f"  初始点数     : {n_starts}")
            print(f"  Adam lr      : {lr}，max_steps={max_steps}，patience={patience}")
            print(f"  Feller λ     : {lambda_feller}")
            print(f"  market IV    : [{iv_mkt.min():.4f}, {iv_mkt.max():.4f}]")
            print()

        all_results: list = []
        best_theta: np.ndarray | None = None
        best_loss = float("inf")

        for i, x0 in enumerate(starts):
            prefix = f"[start {i+1:2d}/{n_starts}]" if verbose else ""
            theta_i, loss_i = self._run_single_adam(
                x0, lr=lr, max_steps=max_steps,
                patience=patience, lambda_feller=lambda_feller,
                verbose_prefix=prefix,
            )
            all_results.append((theta_i, loss_i))

            if verbose:
                sym  = ["κ", "λ", "σ", "ρ", "v₀"]
                pstr = ", ".join(f"{s}={v:.4f}" for s, v in zip(sym, theta_i))
                star = " ★" if loss_i < best_loss else ""
                # 检查最优结果是否满足 Feller 条件
                fp = self.feller_penalty(
                    torch.tensor(theta_i, dtype=torch.float32, device=self.device)
                ).item()
                feller_status = "Feller✓" if fp < 1e-6 else f"Feller✗({fp:.2e})"
                print(f"  [start {i+1:2d}/{n_starts}] loss={loss_i:.4e}  "
                      f"[{pstr}] {feller_status}{star}")

            if loss_i < best_loss:
                best_loss  = loss_i
                best_theta = theta_i

        if verbose:
            print()
            print(f"  ✅ 最优 loss = {best_loss:.6e}")
            for n, v in zip(["kappa","lambda","sigma","rho","v0"], best_theta):
                print(f"    {n:>8s} = {v:.6f}")
            # 最终 Feller 条件检查
            fp_final = self.feller_penalty(
                torch.tensor(best_theta, dtype=torch.float32, device=self.device)
            ).item()
            k, l, s = best_theta[0], best_theta[1], best_theta[2]
            print(f"    Feller: 2κθ={2*k*l:.4f} vs σ²={s**2:.4f} → "
                  f"{'满足 ✓' if fp_final < 1e-6 else '不满足 ✗'}")

        return best_theta, best_loss, all_results
