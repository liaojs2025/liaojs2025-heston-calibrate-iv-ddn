"""
calibration.py
HestonCalibrator — Multi-start Adam 校准器。
利用训练好的 HestonDDN 从市场期权价格反推 Heston 参数（论文 Eq 3.8）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim


class HestonCalibrator:
    """
    Multi-start Adam 校准器。

    策略：
      1. 在参数可行域内随机生成 n_starts 个初始点
      2. 每个初始点独立运行 Adam + ReduceLROnPlateau + Box Projection
      3. 取所有起始点中最终 loss 最小的结果输出

    Normalized 接口（与训练完全一致）：
      - 第9个输入特征: log(K/S0)
      - DDN 输出 / 损失目标: price_norm = Price / S0
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

    # ── 市场数据预处理 ────────────────────────────────────────────────

    def _prepare_market_tensors(self, market_data: dict) -> None:
        """
        将市场数据转为 normalized Tensor，缓存在 self 上。

        market_data keys: 'r', 'tau', 'S0', 'K', 'P_mkt'（均为绝对值数组）
        内部自动完成：
          K      → log(K/S0)
          P_mkt  → P_mkt / S0
        """
        S0_arr  = np.asarray(market_data["S0"],    dtype=np.float32)
        K_arr   = np.asarray(market_data["K"],     dtype=np.float32)
        P_arr   = np.asarray(market_data["P_mkt"], dtype=np.float32)

        def _t(arr):
            return torch.tensor(arr, dtype=torch.float32, device=self.device)

        self.r_t          = _t(np.asarray(market_data["r"],   dtype=np.float32))
        self.tau_t        = _t(np.asarray(market_data["tau"], dtype=np.float32))
        self.S0_t         = _t(S0_arr)
        self.log_k_s0_t   = _t(np.log(K_arr / S0_arr))
        self.P_mkt_norm_t = _t(P_arr / S0_arr).view(-1, 1)
        self.M            = len(P_arr)

    # ── 前向损失 ──────────────────────────────────────────────────────

    def _forward_loss(self, theta_t: torch.Tensor) -> torch.Tensor:
        """给定 theta Tensor (5,)，返回 normalized MSE loss。"""
        theta_exp = theta_t.unsqueeze(0).expand(self.M, -1)          # (M, 5)
        mkt_feat  = torch.stack(
            [self.r_t, self.tau_t, self.S0_t, self.log_k_s0_t], dim=1
        )                                                             # (M, 4)
        X_input   = torch.cat([theta_exp, mkt_feat], dim=1)          # (M, 9)
        P_pred    = self.model(X_input)                               # (M, 1)
        return torch.mean((P_pred - self.P_mkt_norm_t) ** 2)

    # ── 单次 Adam 优化 ────────────────────────────────────────────────

    def _run_single_adam(
        self,
        x0_np: np.ndarray,
        lr: float = 5e-3,
        max_steps: int = 500,
        patience: int = 40,
        min_lr: float = 1e-6,
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
            loss = self._forward_loss(theta)
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
                print(f"  {verbose_prefix} step={step:4d} | "
                      f"loss(norm)={loss_val:.4e} | "
                      f"lr={opt.param_groups[0]['lr']:.2e}")

        return best_theta, best_loss

    # ── 主入口 ────────────────────────────────────────────────────────

    def calibrate(
        self,
        market_data: dict,
        n_starts: int  = 10,
        lr: float      = 5e-3,
        max_steps: int = 500,
        patience: int  = 40,
        verbose: bool  = True,
    ) -> tuple[np.ndarray, float, list]:
        """
        Multi-start Adam 校准。

        参数:
          market_data : dict，key = 'r','tau','S0','K','P_mkt'（绝对值数组）
          n_starts    : 随机初始点数量（默认 10）
          lr          : Adam 初始学习率（默认 5e-3）
          max_steps   : 每个起始点的最大步数（默认 500）
          patience    : 无改善早停阈值（默认 40）
          verbose     : 是否打印进度

        返回:
          best_theta  : np.ndarray (5,)，最优 Heston 参数
          best_loss   : float，最终 normalized MSE
          all_results : list of (theta, loss)，所有起始点结果
        """
        self._prepare_market_tensors(market_data)

        lo, hi = self.PARAM_BOUNDS[:, 0], self.PARAM_BOUNDS[:, 1]
        starts = self.rng.uniform(lo, hi, size=(n_starts, 5)).astype(np.float64)
        print(f"initial guess: {starts}\n")

        if verbose:
            P_norm = self.P_mkt_norm_t.cpu().numpy()
            print(f"  市场期权数   : {self.M}")
            print(f"  初始点数     : {n_starts}")
            print(f"  Adam lr      : {lr}，max_steps={max_steps}，patience={patience}")
            print(f"  price_norm   : [{P_norm.min():.4f}, {P_norm.max():.4f}]")
            print()

        all_results: list = []
        best_theta: np.ndarray | None = None
        best_loss = float("inf")

        for i, x0 in enumerate(starts):
            prefix = f"[start {i+1:2d}/{n_starts}]" if verbose else ""
            theta_i, loss_i = self._run_single_adam(
                x0, lr=lr, max_steps=max_steps,
                patience=patience, verbose_prefix=prefix,
            )
            all_results.append((theta_i, loss_i))

            if verbose:
                sym   = ["κ", "λ", "σ", "ρ", "v₀"]
                pstr  = ", ".join(f"{s}={v:.4f}" for s, v in zip(sym, theta_i))
                star  = " ★" if loss_i < best_loss else ""
                print(f"  [start {i+1:2d}/{n_starts}] loss={loss_i:.4e}  [{pstr}]{star}")

            if loss_i < best_loss:
                best_loss  = loss_i
                best_theta = theta_i

        if verbose:
            print()
            print(f"  ✅ 最优 loss(norm) = {best_loss:.6e}")
            for n, v in zip(["kappa","lambda","sigma","rho","v0"], best_theta):
                print(f"    {n:>8s} = {v:.6f}")

        return best_theta, best_loss, all_results
