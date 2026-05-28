"""
generate_dataset_description.py

生成训练/验证/测试数据集的描述统计表格，输出为 LaTeX 表格和 CSV 文件。
保存到 details/ 文件夹中。
"""

import os
import sys
import pandas as pd
import numpy as np

# ── 路径设置 ──────────────────────────────────────────────────────
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.abspath(os.path.join(THIS_DIR, '..'))
sys.path.insert(0, ROOT_DIR)

CSV_PATH  = os.path.join(ROOT_DIR, 'Heston_Project', 'data', 'heston_dataset_200k.csv')
OUT_DIR   = THIS_DIR  # 输出在 details/

# ── 重现与 01_train.ipynb 完全一致的数据处理逻辑 ──────────────────
print("📂 加载数据集...")
df_raw = pd.read_csv(CSV_PATH)
df_raw.columns = df_raw.columns.str.strip()
print(f"   原始行数: {len(df_raw):,}")

# 若已有 iv 列则直接使用，否则用 py_vollib 补算（与 01_train.ipynb 一致）
if 'iv' not in df_raw.columns or 'vega' not in df_raw.columns:
    print("   补算 IV 和 Vega...")
    try:
        import py_vollib_vectorized as pvv
        import warnings
        price  = df_raw['Price'].values.astype(np.float64)
        S0     = df_raw['S0'].values.astype(np.float64)
        K      = df_raw['K'].values.astype(np.float64)
        tau    = df_raw['tau'].values.astype(np.float64)
        r      = df_raw['r'].values.astype(np.float64)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iv_s   = pvv.vectorized_implied_volatility(
                pd.Series(price), pd.Series(S0), pd.Series(K),
                pd.Series(tau),   pd.Series(r),
                pd.Series(['c'] * len(price)),
                q=0.0, model='black_scholes_merton', return_as='series')
            vega_s = pvv.vectorized_vega(
                pd.Series(['c'] * len(price)),
                pd.Series(S0), pd.Series(K), pd.Series(tau),
                pd.Series(r),  iv_s,
                q=0.0, model='black_scholes_merton', return_as='series')

        df_raw['iv']   = iv_s.values.astype(np.float64)
        df_raw['vega'] = vega_s.values.astype(np.float64)
    except ImportError:
        print("   ⚠️  py_vollib_vectorized 未安装，跳过 IV 补算")

# ── 过滤（与 01_train.ipynb 一致）────────────────────────────────
df = df_raw.copy()
# 1. 过滤 IV/Vega 无效
if 'iv' in df.columns and 'vega' in df.columns:
    before = len(df)
    df = df.dropna(subset=['iv','vega'])
    df = df[df['iv'] > 0]
    print(f"   IV/Vega 清洗: {before:,} → {len(df):,}")

# 2. Feller 条件: 2κλ > σ²
before = len(df)
df = df[2 * df['kappa'] * df['lambda'] > df['sigma'] ** 2].reset_index(drop=True)
print(f"   Feller 过滤: {before:,} → {len(df):,}")

# 3. 梯度列清洗
grad_cols = ['d_kappa','d_lambda','d_sigma','d_rho','d_v0']
before = len(df)
df = df.dropna(subset=grad_cols).reset_index(drop=True)
print(f"   梯度清洗后: {len(df):,} 条")

total = len(df)
print(f"\n✅ 最终可用数据: {total:,} 条\n")

# ── 数据集划分（与 01_train.ipynb 完全相同的比例）────────────────
from sklearn.model_selection import train_test_split

# 70 / 15 / 15 分割
idx = np.arange(total)
idx_train, idx_temp = train_test_split(idx, test_size=0.30, random_state=42)
idx_val,   idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42)

df_train = df.iloc[idx_train].reset_index(drop=True)
df_val   = df.iloc[idx_val].reset_index(drop=True)
df_test  = df.iloc[idx_test].reset_index(drop=True)

print(f"Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")

# ── 描述统计表 ────────────────────────────────────────────────────
# 关注的列：输入特征 + IV + Vega
FEAT_COLS = ['kappa', 'lambda', 'sigma', 'rho', 'v0', 'r', 'tau', 'log_K_S0']
TARGET_COLS = ['iv', 'vega'] if 'iv' in df.columns else []
GRAD_COLS = ['d_kappa', 'd_lambda', 'd_sigma', 'd_rho', 'd_v0']

ALL_COLS = FEAT_COLS + TARGET_COLS + GRAD_COLS

# 完整数据集的描述统计
desc_all = df[ALL_COLS].describe().T[['mean','std','min','25%','50%','75%','max']]
desc_all.columns = ['Mean','Std','Min','25%','50%','75%','Max']
desc_all.index.name = 'Variable'

# 各分割集的基本统计（行数、IV 均值、IV 标准差）
split_summary = pd.DataFrame({
    'Split'    : ['Full (after filtering)', 'Training', 'Validation', 'Test'],
    'N'        : [total, len(df_train), len(df_val), len(df_test)],
    'Share (%)':[100.0, 100*len(df_train)/total, 100*len(df_val)/total,
                 100*len(df_test)/total],
})
if 'iv' in df.columns:
    split_summary['IV Mean']   = [df['iv'].mean(), df_train['iv'].mean(),
                                   df_val['iv'].mean(), df_test['iv'].mean()]
    split_summary['IV Std']    = [df['iv'].std(),  df_train['iv'].std(),
                                   df_val['iv'].std(),  df_test['iv'].std()]
    split_summary['IV Min']    = [df['iv'].min(),  df_train['iv'].min(),
                                   df_val['iv'].min(),  df_test['iv'].min()]
    split_summary['IV Max']    = [df['iv'].max(),  df_train['iv'].max(),
                                   df_val['iv'].max(),  df_test['iv'].max()]

# ── 保存 CSV ─────────────────────────────────────────────────────
desc_all.to_csv(os.path.join(OUT_DIR, 'dataset_full_stats.csv'))
split_summary.to_csv(os.path.join(OUT_DIR, 'dataset_split_summary.csv'), index=False)
print("✅ CSV 已保存: dataset_full_stats.csv, dataset_split_summary.csv")

# ── 生成 LaTeX 表格 ───────────────────────────────────────────────
PARAM_NAMES = {
    'kappa'   : r'$\kappa$',
    'lambda'  : r'$\theta$',
    'sigma'   : r'$\sigma$',
    'rho'     : r'$\rho$',
    'v0'      : r'$v_0$',
    'r'       : r'$r$',
    'tau'     : r'$\tau$',
    'log_K_S0': r'$\ln(K/S_0)$',
    'iv'      : r'IV',
    'vega'    : r'Vega',
    'd_kappa' : r'$\partial P/\partial\kappa$',
    'd_lambda': r'$\partial P/\partial\theta$',
    'd_sigma' : r'$\partial P/\partial\sigma$',
    'd_rho'   : r'$\partial P/\partial\rho$',
    'd_v0'    : r'$\partial P/\partial v_0$',
}

PARAM_DESC = {
    'kappa'   : 'Mean-reversion speed',
    'lambda'  : 'Long-run variance',
    'sigma'   : 'Vol of variance',
    'rho'     : 'Correlation',
    'v0'      : 'Initial variance',
    'r'       : 'Risk-free rate',
    'tau'     : 'Time to maturity (yr)',
    'log_K_S0': 'Log-moneyness',
    'iv'      : 'BS Implied Volatility',
    'vega'    : 'BS Vega',
    'd_kappa' : r'$(\partial P/\partial\kappa)/S_0$',
    'd_lambda': r'$(\partial P/\partial\theta)/S_0$',
    'd_sigma' : r'$(\partial P/\partial\sigma)/S_0$',
    'd_rho'   : r'$(\partial P/\partial\rho)/S_0$',
    'd_v0'    : r'$(\partial P/\partial v_0)/S_0$',
}

def df_to_latex_table1(desc_df):
    """完整描述统计 LaTeX 表格"""
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Descriptive Statistics of the Heston DDN Training Dataset}')
    lines.append(r'\label{tab:dataset_stats}')
    lines.append(r'\resizebox{\textwidth}{!}{%')
    lines.append(r'\begin{tabular}{llrrrrrrr}')
    lines.append(r'\toprule')
    lines.append(r'Variable & Description & Mean & Std & Min & 25\% & Median & 75\% & Max \\')
    lines.append(r'\midrule')

    sections = [
        ('\\multicolumn{9}{l}{\\textit{Input Features (Heston Parameters)}} \\\\', FEAT_COLS[:5]),
        ('\\multicolumn{9}{l}{\\textit{Input Features (Market Variables)}} \\\\',  FEAT_COLS[5:]),
        ('\\multicolumn{9}{l}{\\textit{Target Variable}} \\\\',                    TARGET_COLS),
        ('\\multicolumn{9}{l}{\\textit{Gradient Labels (Price Space)}} \\\\',      GRAD_COLS),
    ]

    for header, cols in sections:
        lines.append(header)
        for col in cols:
            if col not in desc_df.index:
                continue
            row = desc_df.loc[col]
            name = PARAM_NAMES.get(col, col)
            desc = PARAM_DESC.get(col, col)
            lines.append(
                f'{name} & {desc} & '
                f'{row["Mean"]:.4f} & {row["Std"]:.4f} & '
                f'{row["Min"]:.4f} & {row["25%"]:.4f} & '
                f'{row["50%"]:.4f} & {row["75%"]:.4f} & '
                f'{row["Max"]:.4f} \\\\'
            )
        lines.append(r'\midrule')

    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def df_to_latex_table2(split_df):
    """数据集分割汇总 LaTeX 表格"""
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Dataset Split Summary for Heston DDN Training}')
    lines.append(r'\label{tab:dataset_split}')
    if 'IV Mean' in split_df.columns:
        lines.append(r'\begin{tabular}{lrrrrrr}')
        lines.append(r'\toprule')
        lines.append(r'Split & $N$ & Share (\%) & IV Mean & IV Std & IV Min & IV Max \\')
        lines.append(r'\midrule')
        for _, row in split_df.iterrows():
            lines.append(
                f'{row["Split"]} & {int(row["N"]):,} & {row["Share (%)"]:.1f} & '
                f'{row["IV Mean"]:.4f} & {row["IV Std"]:.4f} & '
                f'{row["IV Min"]:.4f} & {row["IV Max"]:.4f} \\\\'
            )
    else:
        lines.append(r'\begin{tabular}{lrr}')
        lines.append(r'\toprule')
        lines.append(r'Split & $N$ & Share (\%) \\')
        lines.append(r'\midrule')
        for _, row in split_df.iterrows():
            lines.append(f'{row["Split"]} & {int(row["N"]):,} & {row["Share (%)"]:.1f} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


latex1 = df_to_latex_table1(desc_all)
latex2 = df_to_latex_table2(split_summary)

with open(os.path.join(OUT_DIR, 'table_dataset_stats.tex'), 'w') as f:
    f.write(latex1)
with open(os.path.join(OUT_DIR, 'table_dataset_split.tex'), 'w') as f:
    f.write(latex2)

print("✅ LaTeX 表格已保存: table_dataset_stats.tex, table_dataset_split.tex")

# ── 同时打印摘要到控制台 ─────────────────────────────────────────
print("\n" + "="*70)
print("DATASET SPLIT SUMMARY")
print("="*70)
print(split_summary.to_string(index=False, float_format='%.4f'))
print("\n" + "="*70)
print("FULL DESCRIPTIVE STATISTICS")
print("="*70)
print(desc_all.to_string(float_format='%.4f'))
