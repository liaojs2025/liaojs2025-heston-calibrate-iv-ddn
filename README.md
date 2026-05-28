# Heston Model Calibration via IV-DDN

**Deep Differential Network for Heston Model Calibration in Implied Volatility Space**

This repository implements a novel approach to calibrating the Heston stochastic volatility model by training a deep neural network to predict **Black-Scholes implied volatility (IV)** rather than option prices directly. The method incorporates physics-informed constraints (Feller condition) and employs a multiplicative gradient trick to stabilize derivative-based regularization.

---

## 📋 Overview

The Heston model is a widely used stochastic volatility model for option pricing, characterized by five parameters: $\kappa$ (mean-reversion speed), $\theta$ (long-run variance), $\sigma$ (volatility of variance), $\rho$ (correlation), and $v_0$ (initial variance). Traditional calibration methods suffer from:
- **Non-identifiability**: Multiple parameter combinations yield similar option price surfaces
- **Numerical instability**: Price-space calibration is sensitive to deep OTM/ITM options
- **Physical inconsistency**: Lack of constraints on the Feller condition ($2\kappa\theta > \sigma^2$)

This project addresses these challenges by:
1. **IV-space calibration**: Training a Deep Differential Network (DDN) to predict implied volatility instead of prices
2. **Vega-weighted loss**: Automatically down-weighting low-Vega (unreliable) contracts
3. **Feller regularization**: Enforcing the variance positivity constraint via soft penalty (PINN approach)
4. **Multiplicative gradient trick**: Avoiding division-by-Vega in derivative-based loss terms

---

## 🏗️ Repository Structure

```
.
├── Heston_Project/              # Main implementation (proposed IV-DDN method)
│   ├── modules/
│   │   ├── model.py             # HestonDDN neural network architecture
│   │   ├── calibration.py       # Multi-start Adam calibrator with Feller penalty
│   │   ├── pricing.py           # QuantLib pricing, IV/Vega computation, data generation
│   │   └── __init__.py
│   ├── data/
│   │   ├── heston_dataset_200k.csv        # Pre-generated training data (200k samples)
│   │   ├── spy_2022_09_01.csv             # SPY call options (Sept 1, 2022)
│   │   └── spy_2022_09_02.csv             # SPY call options (Sept 2, 2022)
│   ├── models/
│   │   └── heston_ddn_weights.pth         # Pre-trained model weights
│   ├── 01_train.ipynb           # Training notebook: DDN training pipeline
│   ├── 02_calibrate.ipynb       # Calibration notebook: synthetic & real-market experiments
│   └── 03_calibrate_visualize.ipynb  # Visualization of calibration results
│
├── Zhang_realize/               # Baseline implementation (Zhang et al. 2025, price-space DDN)
│   └── [similar structure]
│
├── experiments result/          # LaTeX tables and text for thesis/paper
│   ├── table_training_details.tex         # Training hyperparameters & convergence
│   ├── table_test_performance.tex         # Test set IV prediction accuracy
│   ├── table_synthetic_calibration.tex    # Synthetic data calibration comparison
│   ├── table_spy_data.tex                 # SPY market data description
│   ├── table_spy_calibration.tex          # Real-market calibration performance
│   ├── text_synthetic_calibration.tex     # Narrative for synthetic experiments
│   └── text_spy_calibration.tex           # Narrative for real-market experiments
│
├── details/                     # Dataset statistics and parameter range tables
├── output/                      # Architecture diagrams and figures
├── par-yield-curve-rates-2020-2023.csv   # Federal Reserve par yield data
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.10+ required
pip install torch numpy pandas scipy QuantLib py_vollib_vectorized
```

### 1. Train the IV-DDN Model
```bash
cd Heston_Project
jupyter notebook 01_train.ipynb
```
**Key steps:**
- Loads/generates 200k synthetic Heston option prices (satisfying Feller condition)
- Computes BS implied volatilities and Vegas for all samples
- Trains HestonDDN with **multiplicative trick gradient loss**:
  $$\mathcal{L} = \text{MSE}(\hat{\sigma}_{\text{IV}}, \sigma_{\text{IV}}) + \lambda \cdot \text{SmoothL1}\left(\frac{\partial\hat{\sigma}_{\text{IV}}}{\partial\boldsymbol{\theta}} \times \mathcal{V},\; \frac{\partial P}{\partial\boldsymbol{\theta}}\right)$$
- Achieves **0.49% mean relative error** on 25.8k test samples (91% < 1% error)
- Saves weights to `models/heston_ddn_weights.pth`

### 2. Calibrate on Synthetic/Real Data
```bash
jupyter notebook 02_calibrate.ipynb
```
**Experiments:**
- **Part 1 (Synthetic)**: Calibrates on a known Heston parameter set (99 options), validates parameter recovery and re-pricing accuracy
- **Part 2 (Real SPY)**: Calibrates on SPY call options from Sept 2022, performs same-day and cross-day validation

### 3. Visualize Results
```bash
jupyter notebook 03_calibrate_visualize.ipynb
```

---

## 🔬 Key Technical Innovations

### 1. **IV-Space Calibration**
Unlike traditional methods that minimize option price errors, we calibrate by minimizing IV errors:
$$\mathcal{L}_{\text{IV}} = \frac{1}{M} \sum_{i=1}^{M} \mathcal{V}_i \cdot \left(\hat{\sigma}^{\text{IV}}_i(\boldsymbol{\theta}) - \sigma^{\text{IV}, \text{mkt}}_i\right)^2$$
- **Vega weighting** ($\mathcal{V}_i$) naturally down-weights unreliable deep OTM/ITM contracts
- More stable than price-space optimization for low-priced options

### 2. **Multiplicative Gradient Trick**
Standard derivative-based DDN loss requires $\partial\sigma_{\text{IV}}/\partial\boldsymbol{\theta} = (\partial P/\partial\boldsymbol{\theta}) \times (S_0/\mathcal{V})$, which **explodes when Vega→0**. Our solution:
$$\mathcal{L}_{\text{grad}} = \text{SmoothL1}\left(\frac{\partial\hat{\sigma}_{\text{IV}}}{\partial\boldsymbol{\theta}} \times \mathcal{V},\; \frac{\partial P}{\partial\boldsymbol{\theta}}\right)$$
- Multiply predicted IV gradient by Vega (rather than dividing true gradient by Vega)
- Uses Smooth L1 (Huber) loss for robustness against outliers
- Curriculum learning: gradient loss disabled for first 20 epochs

### 3. **Feller Condition Enforcement (PINN)**
The Feller condition $2\kappa\theta > \sigma^2$ ensures variance remains positive. We add a soft penalty:
$$\mathcal{L}_{\text{Feller}} = \lambda_F \cdot \text{ReLU}(\sigma^2 - 2\kappa\theta + \epsilon)$$
- Prevents unphysical parameter estimates
- $\lambda_F = 10.0$ balances fitting accuracy and physical consistency

### 4. **Multi-Start Adam Optimization**
- 10 random restarts from parameter bounds $[\kappa, \theta, \sigma, \rho, v_0] \in [0.01,5]\times[0.01,1]\times[0.1,1]\times[-0.9,-0.05]\times[0.01,1]$
- Learning rate: $5 \times 10^{-3}$ with early stopping (patience=40)
- Box projection to enforce hard constraints after each step

---

## 📊 Performance Summary

### Training Performance (200k Synthetic Data)
| Metric | Value |
|--------|-------|
| Test set size | 25,832 |
| Mean IV relative error | **0.49%** |
| Median IV relative error | 0.23% |
| Error < 1% | 91.0% |
| Error < 5% | 98.9% |
| IV MAE | 0.002705 |
| Training time (200 epochs) | 507 s (MPS/M1) |

### Synthetic Data Calibration (99 Options)
| Method | Proposed IV-DDN | Zhang (2025) Price-DDN |
|--------|----------------|----------------------|
| IV MRE (mean) | **3.08%** | — |
| Price MRE (QuantLib) | **12.78%** | 184.80% |
| Feller satisfied | ✅ Yes | ❌ No |
| Calibration time | 11.76 s | 11.30 s |

### Real SPY Data Calibration (10 Options per Day)
| Scenario | Proposed IV-DDN | Zhang (2025) Price-DDN |
|----------|----------------|----------------------|
| Same-day (9/2→9/2) IV MRE | **0.16%** | 0.20% |
| Cross-day (9/1→9/2) IV MRE | **0.96%** | 1.59% |
| MRE degradation (same→cross) | +0.80 pp | +1.39 pp |

**Conclusion**: IV-DDN achieves superior parameter stability across trading days while maintaining competitive fitting accuracy.

---

## 📁 Core Modules

### `modules/model.py` — HestonDDN Architecture
- **Input**: 8 features $[\kappa, \theta, \sigma, \rho, v_0, r, \tau, \ln(K/S_0)]$
- **Architecture**: 6 hidden layers × 150 neurons, Softplus activation
- **Output**: Predicted IV (Softplus ensures non-negativity)
- **Normalization**: Min-Max scaling with frozen bounds (stored in `register_buffer`)
- **Initialization**: Xavier uniform (stable gradient flow)

### `modules/calibration.py` — HestonCalibrator
- Multi-start Adam with Feller penalty
- Vega-weighted IV MSE objective
- Box projection for parameter bounds
- Early stopping via patience counter

### `modules/pricing.py` — QuantLib Interface
- `calculate_heston_price()`: QuantLib Heston analytic pricing
- `compute_iv_vega_batch()`: Vectorized BS IV/Vega calculation via `py_vollib_vectorized`
- `generate_training_data()`: LHS sampling with Feller filtering
- `check_feller()`: Validates $2\kappa\theta > \sigma^2$

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{liao2026heston,
  title={Calibrating the Heston model with implied-volatility Deep Differential Networks},
  author={Liao, Jiansong},
  year={2026},
  school={[Tsinghua University]},
  note={GitHub: https://github.com/liaojs2025/heston_calibrate_iv_ddn}
}
```

**Baseline comparison**: Zhang, Y., et al. (2025). "Calibrating the Heston Model with Deep Differential Networks." *Quantitative Finance*, 25(1), 1-15.

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **QuantLib**: For Heston analytic pricing
- **py_vollib_vectorized**: For efficient BS IV/Vega computation
- **Federal Reserve**: For par yield curve data (2020-2023)

---

## 📧 Contact

For questions or collaboration inquiries, please open an issue or contact [liaojs2025@github.com].

---

**Last Updated**: May 2026
