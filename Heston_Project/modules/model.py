"""
model.py
HestonDDN — Deep Differential Network for Heston model IV prediction.

升级说明（相对原版）:
  1. 预测目标从 price_norm = Price/S0 改为 BS 隐含波动率 (IV)
  2. 最后一层输出加 Softplus 激活，保证 IV 预测非负
  3. 【乘法 Trick】梯度 Loss 在价格空间计算，避免除以 Vega 导致的数值爆炸:
     pred_grad_price ≡ (∂IV_hat/∂θ) * Vega，与 true_grad_price 比较
  4. 梯度 Loss 采用 Smooth L1 (Huber Loss) 抗离群点
  5. 保留 Min-Max 归一化、Xavier 初始化、无 Dropout 等原有设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HestonDDN(nn.Module):
    """
    IV-目标深度微分网络。

    架构：6 层隐藏层 × 150 个神经元，Softplus 激活，Xavier 初始化。

    关键设计：
      1. 内置 Min-Max 归一化（register_buffer 随模型存档、随 .to(device) 移动）
      2. 训练目标为 BS 隐含波动率 (IV)，取代原来的 price_norm
      3. 最后一层加 Softplus 保证输出非负（IV ∈ [0, +∞)）
      4. 利用期权定价齐次性，仅使用 log(K/S0) 代替独立的 S0、K
      5. 8维输入: kappa, lambda, sigma, rho, v0, r, tau, log_K_S0
      6. 无 Dropout：Dropout 使每次 forward 的导数不一致，导致微分损失项失效
    """

    def __init__(
        self,
        input_dim: int = 8,
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

        # ── 网络主体：Linear → Softplus → … × 6 → Linear(1) → Softplus ──
        layers = []
        layers += [nn.Linear(input_dim, neurons_per_layer), nn.Softplus()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(neurons_per_layer, neurons_per_layer), nn.Softplus()]
        layers.append(nn.Linear(neurons_per_layer, 1))
        # ★ 关键：最后加 Softplus 保证 IV 预测值非负
        layers.append(nn.Softplus())
        self.network = nn.Sequential(*layers)

        # Xavier 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    # ── 归一化工具 ───────────────────────────────────────────────────

    def fit_scalers(self, X_raw: torch.Tensor, y_raw: torch.Tensor) -> None:
        """
        训练前调用一次，基于训练集计算并冻结归一化极值。
        y_raw 现在是 IV（隐含波动率），不再是 price_norm。
        """
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

    def _normalize_grads_price(self, grads_price_raw: torch.Tensor) -> torch.Tensor:
        """
        归一化 **价格空间** 的梯度标签（乘法 Trick 版本）:

        真实标签 d_xxx 在 CSV 中的定义是 (∂Price/∂θ_i) / S0，
        这里我们需要的是归一化空间的等效量。

        核心公式:
          d(y_norm)/d(X_norm) = (ΔX / Δy) · d(y_raw)/d(X_raw)

        但在乘法 Trick 中，我们是在归一化空间中用 autograd 算 ∂IV_norm/∂θ_norm，
        然后乘以 vega 映射到价格空间。为了让 true target 也在同一尺度下，
        我们对 true_grad_price 做相同的归一化:
          grad_price_norm_i = grad_price_raw_i * (ΔX_i / Δy_price)

        但由于我们没有 y_price 的 scale，改用 X_scale 只对 theta 维度归一化:
          grad_price_norm_i = grad_price_raw_i * ΔX_i

        等价地：网络在归一化空间输出 ∂IV_norm/∂θ_norm，乘以 vega 后
        需要与 true_grad_price 比较。两边都不做额外缩放，直接在原始物理量空间对比。
        这样最简洁、最稳定。

        因此：此函数不再需要，我们直接在原始空间做 Huber Loss。
        保留此方法仅为向后兼容（返回原始值不变）。

        grads_price_raw: (batch, heston_dim=5)，价格空间梯度 (∂Price/∂θ)/S0
        """
        return grads_price_raw

    # ── 前向传播（推理） ──────────────────────────────────────────────

    def forward(self, X_raw: torch.Tensor) -> torch.Tensor:
        """
        输入原始特征，输出预测的 IV (隐含波动率)。
        注意：最后一层 Softplus 保证输出非负。
        """
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit_scalers(X_train, y_train)")
        X_scaled = self._normalize_X(X_raw)
        # network 最后一层带 Softplus，输出非负的归一化 IV
        return self._denormalize_y(self.network(X_scaled))

    # ── 训练步骤 ─────────────────────────────────────────────────────

    def compute_loss(
        self,
        X_raw: torch.Tensor,
        y_iv_raw: torch.Tensor,
        true_vega: torch.Tensor,
        true_grad_price: torch.Tensor,
        lambda_deriv: float = 0.001,
    ):
        """
        联合损失 = IV MSE  +  lambda_deriv × Smooth_L1(价格空间梯度)

        ===================================================================
        【核心数学：乘法 Trick — 在价格空间而非 IV 空间计算梯度误差】
        ===================================================================

        原始做法（有除以零灾难）:
          true_grad_iv = true_grad_price / vega     ← 当 Vega→0 时爆炸！
          L_grad = MSE(pred_grad_iv, true_grad_iv)

        乘法 Trick（本实现）:
          pred_grad_iv = autograd(IV_hat, theta)    ← 网络自动微分
          pred_grad_price_equiv = pred_grad_iv * true_vega   ← 乘法映射回价格空间
          L_grad = Smooth_L1(pred_grad_price_equiv, true_grad_price)

        数学等价性:
          ∂Price/∂θ = (∂IV/∂θ) × Vega    (链式法则)
          所以比较 pred_grad_iv * vega vs true_grad_price 等价于
          比较 ∂IV/∂θ，但避免了除以 Vega 的数值不稳定性。

        Smooth L1 (Huber Loss) 的选择:
          价格梯度在深度 OTM/ITM 期权处可能有大离群值，
          Huber Loss 在残差 > 1 时降级为 L1，对离群点具有鲁棒性。

        参数:
          X_raw            : (batch, 8)  原始输入特征
          y_iv_raw         : (batch, 1)  真实 IV（隐含波动率）
          true_vega        : (batch, 1)  真实 BS Vega（∂Price/∂σ_BS）
          true_grad_price  : (batch, 5)  价格空间梯度 (∂Price/∂θ_i) / S0
          lambda_deriv     : 梯度损失权重（延迟课程学习，前 N epoch 设为 0）
        返回:
          total_loss, loss_iv, loss_grad
        """
        if not self.is_fitted:
            raise RuntimeError("训练前必须先调用 fit_scalers!")

        # ── 1. IV 拟合 Loss（MSE，在归一化空间计算） ──────────────────
        y_true_scaled = self._normalize_y(y_iv_raw)
        X_scaled = self._normalize_X(X_raw)
        X_scaled.requires_grad_(True)

        y_pred_scaled = self.network(X_scaled)       # 归一化空间的 IV 预测
        loss_iv = F.mse_loss(y_pred_scaled, y_true_scaled)

        # ── 2. 梯度 Loss（乘法 Trick + Smooth L1） ──────────────────
        if lambda_deriv > 0:
            # ── 2a. 自动微分: 求网络预测 IV 对归一化输入的梯度 ──
            # ∂(IV_pred_scaled) / ∂(X_scaled)  →  shape (batch, 8)
            pred_grads_full = torch.autograd.grad(
                outputs=y_pred_scaled,
                inputs=X_scaled,
                grad_outputs=torch.ones_like(y_pred_scaled),
                create_graph=True,
                retain_graph=True,
            )[0]

            # 只取前 heston_dim=5 维的梯度（kappa, lambda, sigma, rho, v0）
            pred_heston_grads_norm = pred_grads_full[:, :self.heston_dim]  # (batch, 5)

            # ── 2b. 反归一化: 将归一化空间的梯度转换回原始物理量空间 ──
            # 链式法则: ∂IV_raw/∂θ_raw = ∂IV_norm/∂θ_norm × (Δy / Δθ)
            delta_theta = (self.X_max[:self.heston_dim]
                           - self.X_min[:self.heston_dim] + 1e-8)  # (5,)
            delta_iv = self.y_max - self.y_min + 1e-8              # scalar
            pred_grad_iv_raw = pred_heston_grads_norm * (delta_iv / delta_theta)
            # pred_grad_iv_raw: (batch, 5)，物理量空间的 ∂IV/∂θ_i

            # ── 2c. 【核心乘法 Trick】将 IV 梯度乘以 Vega 映射回价格空间 ──
            # 数学关系: ∂Price/∂θ = (∂IV/∂θ) × Vega
            # true_vega: (batch, 1)，广播到 (batch, 5)
            pred_grad_price_equiv = pred_grad_iv_raw * true_vega   # (batch, 5)

            # ── 2d. 真实价格梯度标签 ──
            # true_grad_price 已经是 (∂Price/∂θ)/S0 形式
            # pred 也需要除以 S0 使两边量纲一致
            # 注意: 训练数据中 S0 固定=100，但为通用性，
            # 我们用 X_raw 中不含 S0，所以 true_grad_price 本身就是 d_xxx = (∂P/∂θ)/S0
            # pred 端: pred_grad_price_equiv 是 (∂IV/∂θ) × Vega = ∂P/∂θ（绝对价格梯度）
            # 需要也除以 S0 对齐: 但训练数据 S0=100（固定），所以 /100
            S0_FIXED = 100.0
            pred_grad_price_norm = pred_grad_price_equiv / S0_FIXED  # (batch, 5)

            # ── 2e. Smooth L1 Loss (Huber Loss): 对离群点鲁棒 ──
            loss_grad = F.smooth_l1_loss(pred_grad_price_norm, true_grad_price)
        else:
            # Warmup 阶段: 不计算梯度 Loss，节省计算量
            loss_grad = torch.tensor(0.0, device=X_raw.device)

        total_loss = loss_iv + lambda_deriv * loss_grad
        return total_loss, loss_iv, loss_grad
