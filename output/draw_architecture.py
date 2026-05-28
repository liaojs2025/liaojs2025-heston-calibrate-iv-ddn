"""
生成 Heston IV-DDN 网络架构图（含乘法 Trick 训练流程）
输出: output/architecture_ddn.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os, sys
import torch
import torch.nn as nn

# Auto-detect network architecture from project model.py (fallback to defaults)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
try:
    from Heston_Project.modules.model import HestonDDN
    _model = HestonDDN()
    DETECT = {
        'input_dim': getattr(_model, 'input_dim', 8),
        'heston_dim': getattr(_model, 'heston_dim', 5),
    }
    linear_layers = [l for l in _model.network if isinstance(l, nn.Linear)]
    if len(linear_layers) >= 1:
        DETECT['neurons'] = linear_layers[0].out_features
        # hidden count = number of hidden Linear layers = total linear layers - 1 (output projection)
        DETECT['hidden_layers'] = max(1, len(linear_layers) - 1)
    else:
        DETECT['neurons'] = 150
        DETECT['hidden_layers'] = 6
except Exception:
    DETECT = {'input_dim':8,'heston_dim':5,'neurons':150,'hidden_layers':6}

# ── 全局设置 ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 13,
})

fig, ax = plt.subplots(1, 1, figsize=(18, 11))
ax.set_xlim(-1, 19)
ax.set_ylim(-1.5, 11)
ax.set_aspect('equal')
ax.axis('off')

# ── 颜色方案 ──
C_INPUT    = '#4A90D9'   # 蓝色 - 输入
C_HIDDEN   = '#5B8C5A'   # 绿色 - 隐藏层
C_OUTPUT   = '#E8734A'   # 橙色 - 输出
C_LOSS     = '#D94F4F'   # 红色 - 损失
C_GRAD     = '#9B59B6'   # 紫色 - 梯度/微分
C_DATA     = '#F4D03F'   # 黄色 - 数据标签
C_NORM     = '#85C1E9'   # 浅蓝 - 归一化
C_BG       = '#F8F9FA'   # 背景
C_ARROW    = '#2C3E50'

fig.patch.set_facecolor(C_BG)

# ── 辅助函数 ──
def draw_box(x, y, w, h, text, color, fontsize=9, textcolor='white', alpha=0.9, bold=False):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='#333333', linewidth=1.2, alpha=alpha)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=textcolor, weight=weight, wrap=True)
    return (x + w/2, y + h/2)

def draw_arrow(x1, y1, x2, y2, color=C_ARROW, style='->', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               connectionstyle='arc3,rad=0'))

def draw_curved_arrow(x1, y1, x2, y2, color=C_ARROW, rad=0.3, lw=1.5, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               connectionstyle=f'arc3,rad={rad}'))

# ════════════════════════════════════════════════════════════════════
# 标题
# ════════════════════════════════════════════════════════════════════
ax.text(9, 10.5, 'Heston IV-DDN: Deep Differential Network Architecture',
        ha='center', va='center', fontsize=19, fontweight='bold', color='#2C3E50')
ax.text(9, 10.0, 'IV-Target with Multiplication Trick Gradient Regularization',
        ha='center', va='center', fontsize=13, color='#666666', style='italic')

# ════════════════════════════════════════════════════════════════════
# Part 1: 输入层 (左侧)
# ════════════════════════════════════════════════════════════════════
input_labels = ['κ', 'λ', 'σ', 'ρ', 'v0', 'r', 'τ', 'log(K/S0)']
input_colors = [C_INPUT]*5 + ['#6EAED6']*3  # 前5维 Heston 参数用深蓝，后3维市场变量用浅蓝

bx, by = 0.0, 2.5
bw, bh = 1.5, 0.52
gap = 0.08

# 输入框标题
ax.text(bx + bw/2, by + (len(input_labels))*(bh+gap) + 0.3,
        'Input (8-dim)', ha='center', va='center',
        fontsize=13, fontweight='bold', color=C_INPUT)

# 大括号标注 Heston 参数
ax.annotate('', xy=(-0.5, by + 2.5*(bh+gap)),
            xytext=(-0.5, by + 7.5*(bh+gap)),
            arrowprops=dict(arrowstyle='-', color='#999', lw=1))
ax.text(-0.75, by + 5*(bh+gap), 'Θ', ha='center', va='center',
        fontsize=14, fontweight='bold', color=C_INPUT, style='italic')

for i, (label, c) in enumerate(zip(reversed(input_labels), reversed(input_colors))):
    y_pos = by + i * (bh + gap)
    draw_box(bx, y_pos, bw, bh, label, c, fontsize=13, bold=True)

# ════════════════════════════════════════════════════════════════════
# Part 2: Min-Max 归一化
# ════════════════════════════════════════════════════════════════════
norm_x, norm_y = 2.5, 5.2
draw_box(norm_x, norm_y, 2.0, 0.8, 'Min-Max\nNormalize', C_NORM,
         fontsize=11, textcolor='#2C3E50', bold=True)

# 输入 → 归一化 箭头
draw_arrow(bx + bw + 0.05, by + 4*(bh+gap), norm_x, norm_y + 0.4, color=C_INPUT)

# ════════════════════════════════════════════════════════════════════
# Part 3: 隐藏层 (中间)  ← 使用检测到的网络参数动态绘制
# ════════════════════════════════════════════════════════════════════
hidden_x_start = 5.2
hbox_w, hbox_h = 2.0, 0.75
hgap = 0.35
layer_positions = []

# 归一化 → 第一层
draw_arrow(norm_x + 2.0, norm_y + 0.4, hidden_x_start, norm_y + 0.4, color=C_NORM)

num_hidden = DETECT['hidden_layers']
neurons = DETECT['neurons']
input_dim_detect = DETECT['input_dim']

for i in range(num_hidden):
    hx = hidden_x_start
    hy = 7.4 - i * (hbox_h + hgap)
    if i == 0:
        label = f'Linear({input_dim_detect}→{neurons})\n+ Softplus'
    else:
        label = f'Linear({neurons}→{neurons})\n+ Softplus'
    pos = draw_box(hx, hy, hbox_w, hbox_h, label, C_HIDDEN, fontsize=10)
    layer_positions.append((hx, hy))

# 层间箭头
for i in range(max(0, num_hidden - 1)):
    x1, y1 = layer_positions[i]
    x2, y2 = layer_positions[i+1]
    draw_arrow(x1 + hbox_w/2, y1, x2 + hbox_w/2, y2 + hbox_h, color=C_HIDDEN, lw=1.2)

# 隐藏层标题
ax.text(hidden_x_start + hbox_w/2, 8.5,
        f'Hidden Layers ({num_hidden} × {neurons} neurons)', ha='center', va='center',
        fontsize=12, fontweight='bold', color=C_HIDDEN)

# ════════════════════════════════════════════════════════════════════
# Part 4: 输出头
# ════════════════════════════════════════════════════════════════════
out_x = 8.0
out_y = layer_positions[-1][1] - 1.3

# Output projection
draw_box(out_x, out_y, 2.0, 0.7, 'Linear(150→1)', '#777',
         fontsize=11, textcolor='white', bold=True)
draw_arrow(hidden_x_start + hbox_w/2, layer_positions[-1][1],
           out_x + 1.0, out_y + 0.7, color='#777', lw=1.2)

# Softplus
sp_y = out_y - 1.0
draw_box(out_x, sp_y, 2.0, 0.6, 'Softplus\n(IV ≥ 0)', C_OUTPUT,
         fontsize=11, bold=True)
draw_arrow(out_x + 1.0, out_y, out_x + 1.0, sp_y + 0.6, color='#777')

# Denormalize
dn_y = sp_y - 0.9
draw_box(out_x, dn_y, 2.0, 0.6, 'Denormalize\n→ IV_pred', C_NORM,
         fontsize=11, textcolor='#2C3E50', bold=True)
draw_arrow(out_x + 1.0, sp_y, out_x + 1.0, dn_y + 0.6, color=C_NORM)

# ════════════════════════════════════════════════════════════════════
# Part 5: 损失函数 (右侧)
# ════════════════════════════════════════════════════════════════════
loss_x = 11.5

# ── L_IV (MSE Loss)
liv_y = 7.0
draw_box(loss_x, liv_y, 2.8, 0.9,
         'L_IV = MSE(IV_hat, IV_true)\n(normalized space)', C_LOSS,
         fontsize=10.5, bold=True)

# IV_true 数据标签
draw_box(loss_x + 3.3, liv_y + 0.1, 1.6, 0.7, 'IV_true\n(label)', C_DATA,
         fontsize=10.5, textcolor='#333', bold=True)
draw_arrow(loss_x + 3.3, liv_y + 0.45, loss_x + 2.8, liv_y + 0.45, color=C_DATA, lw=1.2)

# 网络预测 → L_IV
draw_arrow(hidden_x_start + hbox_w, layer_positions[1][1] + hbox_h/2,
           loss_x, liv_y + 0.45, color=C_LOSS, lw=1.2)

# ── Autograd (∂IV̂/∂Θ)
ag_y = 5.2
draw_box(loss_x, ag_y, 2.8, 0.9,
         'dIV_hat/dTheta  (autograd)\n-> pred_grad_iv', C_GRAD,
         fontsize=10.5, bold=True)
draw_curved_arrow(hidden_x_start + hbox_w, layer_positions[2][1] + hbox_h/2,
                  loss_x, ag_y + 0.7, color=C_GRAD, rad=-0.15, lw=1.2)

# ── × Vega (乘法 Trick)
mul_y = 3.6
draw_box(loss_x, mul_y, 2.8, 0.9,
         'x Vega  (Mult. Trick)\n= dP_hat/dTheta', C_GRAD,
         fontsize=10, bold=True, alpha=0.85)
draw_arrow(loss_x + 1.4, ag_y, loss_x + 1.4, mul_y + 0.9, color=C_GRAD, lw=1.2)

# Vega label
draw_box(loss_x + 3.3, mul_y + 0.1, 1.6, 0.7, 'Vega\n(BS)', C_DATA,
         fontsize=10.5, textcolor='#333', bold=True)
draw_arrow(loss_x + 3.3, mul_y + 0.45, loss_x + 2.8, mul_y + 0.45, color=C_DATA, lw=1.2)

# ── L_Grad (Smooth L1)
lg_y = 1.8
draw_box(loss_x, lg_y, 2.8, 0.9,
         'L_grad = SmoothL1\n(pred_grad_price, dP/dTheta)', C_LOSS,
         fontsize=10, bold=True)
draw_arrow(loss_x + 1.4, mul_y, loss_x + 1.4, lg_y + 0.9, color=C_GRAD, lw=1.2)

# True gradient label (∂P/∂Θ)
draw_box(loss_x + 3.3, lg_y + 0.1, 1.6, 0.7, 'dP/dTheta\n(QuantLib)', C_DATA,
         fontsize=10.5, textcolor='#333', bold=True)
draw_arrow(loss_x + 3.3, lg_y + 0.45, loss_x + 2.8, lg_y + 0.45, color=C_DATA, lw=1.2)

# ── Total Loss
total_y = 0.2
draw_box(loss_x - 0.3, total_y, 3.5, 0.9,
         'L_total = L_IV + lambda_d * L_grad', C_LOSS,
         fontsize=11.5, bold=True, alpha=0.95)
draw_arrow(loss_x + 1.4, lg_y, loss_x + 1.4, total_y + 0.9, color=C_LOSS, lw=1.5)
draw_curved_arrow(loss_x + 1.4, liv_y, loss_x + 0.5, total_y + 0.9,
                  color=C_LOSS, rad=0.4, lw=1.5)

# ════════════════════════════════════════════════════════════════════
# Part 7: 图例
# ════════════════════════════════════════════════════════════════════
legend_elements = [
    mpatches.Patch(facecolor=C_INPUT, edgecolor='#333', label='Input (Theta: Heston Params)'),
    mpatches.Patch(facecolor='#6EAED6', edgecolor='#333', label='Input (Market Variables)'),
    mpatches.Patch(facecolor=C_HIDDEN, edgecolor='#333', label='Hidden Layers'),
    mpatches.Patch(facecolor=C_OUTPUT, edgecolor='#333', label='Output (Softplus -> IV >= 0)'),
    mpatches.Patch(facecolor=C_GRAD, edgecolor='#333', label='Gradient Path (Autograd + Mult. Trick)'),
    mpatches.Patch(facecolor=C_LOSS, edgecolor='#333', label='Loss Components'),
    mpatches.Patch(facecolor=C_DATA, edgecolor='#333', label='Training Labels'),
]
ax.legend(handles=legend_elements, loc='lower left',
          bbox_to_anchor=(-0.05, -0.12), ncol=4, fontsize=10,
          frameon=True, fancybox=True, shadow=False)

# ════════════════════════════════════════════════════════════════════
# 保存
# ════════════════════════════════════════════════════════════════════
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'architecture_ddn.png')
plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=C_BG)
plt.close()
print(f"✅ 架构图已保存: {output_path}")
