"""
model.py
HestonDDN — Deep Differential Network for Heston model option pricing.
论文: Zhang et al. 2025, "Calibrating the Heston model with deep differential networks"
"""

import torch
import torch.nn as nn


class HestonDDN(nn.Module):
    """
    架构：6 层隐藏层 × 150 个神经元，Softplus 激活，Xavier 初始化。

    关键设计：
      1. 内置 Min-Max 归一化（register_buffer 随模型存档、随 .to(device) 移动）
      2. 训练目标为 price_norm = Price / S0（一阶齐次化，消除 S0 量纲差异）
      3. 第9个输入特征为 log(K/S0)（log-moneyness，金融意义更明确）
      4. 无 Dropout：Dropout 使每次 forward 的导数不一致，导致微分损失项失效
    """

    def __init__(
        self,
        input_dim: int = 9,
        heston_param_dim: int = 5,
        hidden_layers: int = 6,
        neurons_per_layer: int = 150,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.heston_dim = heston_param_dim  # 前5维: kappa, lambda, sigma, rho, v0

        # ── 归一化 Buffer（不参与梯度更新） ──────────────────────────
        self.register_buffer("X_min", torch.zeros(input_dim))
        self.register_buffer("X_max", torch.ones(input_dim))
        self.register_buffer("y_min", torch.zeros(1))
        self.register_buffer("y_max", torch.ones(1))
        self.is_fitted: bool = False

        # ── 网络主体：Linear → Softplus → … × 6 → Linear(1) ────────
        layers = []
        layers += [nn.Linear(input_dim, neurons_per_layer), nn.Softplus()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(neurons_per_layer, neurons_per_layer), nn.Softplus()]
        layers.append(nn.Linear(neurons_per_layer, 1))
        self.network = nn.Sequential(*layers)

        # Xavier 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    # ── 归一化工具 ───────────────────────────────────────────────────

    def fit_scalers(self, X_raw: torch.Tensor, y_raw: torch.Tensor) -> None:
        """训练前调用一次，基于训练集计算并冻结归一化极值。"""
        self.X_min = X_raw.min(dim=0)[0].detach()
        self.X_max = X_raw.max(dim=0)[0].detach()
        self.y_min = y_raw.min(dim=0)[0].detach()
        self.y_max = y_raw.max(dim=0)[0].detach()
        self.is_fitted = True

    def _normalize_X(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.X_min) / (self.X_max - self.X_min + 1e-8)

    def _normalize_y(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_min) / (self.y_max - self.y_min + 1e-8)

    def _denormalize_y(self, y_scaled: torch.Tensor) -> torch.Tensor:
        return y_scaled * (self.y_max - self.y_min) + self.y_min

    def _normalize_grads(self, grads_raw: torch.Tensor) -> torch.Tensor:
        """
        论文 Eq (4.2)：
          d(P_norm)/d(theta_norm) = (delta_theta / delta_P) * d(P_raw)/d(theta_raw)
        grads_raw: (batch, heston_dim=5)，已除以 S0 的梯度标签
        """
        delta_X = self.X_max[: self.heston_dim] - self.X_min[: self.heston_dim] + 1e-8
        delta_y = self.y_max - self.y_min + 1e-8
        return grads_raw * (delta_X / delta_y)

    # ── 前向传播（推理） ──────────────────────────────────────────────

    def forward(self, X_raw: torch.Tensor) -> torch.Tensor:
        """
        输入原始特征，输出 price_norm = Price/S0。
        如需绝对价格，在外部乘以 S0。
        """
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit_scalers(X_train, y_train)")
        X_scaled = self._normalize_X(X_raw)
        return self._denormalize_y(self.network(X_scaled))

    # ── 训练步骤 ─────────────────────────────────────────────────────

    def compute_loss(
        self,
        X_raw: torch.Tensor,
        y_raw: torch.Tensor,
        grads_raw: torch.Tensor,
    ):
        """
        联合损失 = 价格 MSE + 梯度 MSE（论文 Eq 3.7）

        参数:
          X_raw     : (batch, 9)  原始输入特征，第9维为 log(K/S0)
          y_raw     : (batch, 1)  price_norm = Price/S0
          grads_raw : (batch, 5)  d_xxx / S0
        返回:
          total_loss, loss_price, loss_grad
        """
        if not self.is_fitted:
            raise RuntimeError("训练前必须先调用 fit_scalers!")

        y_true_scaled     = self._normalize_y(y_raw)
        grads_true_scaled = self._normalize_grads(grads_raw)

        X_scaled = self._normalize_X(X_raw)
        X_scaled.requires_grad_(True)

        y_pred_scaled = self.network(X_scaled)

        pred_grads_full = torch.autograd.grad(
            outputs=y_pred_scaled,
            inputs=X_scaled,
            grad_outputs=torch.ones_like(y_pred_scaled),
            create_graph=True,
            retain_graph=True,
        )[0]

        pred_heston_grads = pred_grads_full[:, : self.heston_dim]

        loss_price = nn.functional.mse_loss(y_pred_scaled, y_true_scaled)
        loss_grad  = nn.functional.mse_loss(pred_heston_grads, grads_true_scaled)
        total_loss = loss_price + loss_grad

        return total_loss, loss_price, loss_grad
