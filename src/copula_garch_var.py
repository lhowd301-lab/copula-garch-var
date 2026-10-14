"""
Copula-GARCH VaR 模型 — 完整实现
====================================
中证500 + 中证1000 等权组合
AR(1)-GJR-GARCH(1,1)-Student-t 边缘分布
Clayton Copula (主) + Gaussian Copula (基准)
滚动窗口 (500天) → VaR 预测 → Kupiec POF 回测
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 读取数据
# ============================================================
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(PROJECT_ROOT / 'data/returns.csv', parse_dates=['date'])
df = df.set_index('date').dropna()
r500  = df['r_500'].values   # 日对数收益率 × 100
r1000 = df['r_1000'].values
dates = df.index
n_all = len(r500)

# ============================================================
# 1. 边缘分布拟合 — AR(1)-GJR-GARCH(1,1)-Student-t
# ============================================================
def fit_garch(y):
    """
    拟合 AR(1)-GJR-GARCH(1,1) with Student-t 分布
    返回: (mu_forecast, sigma_forecast, df_t, resid_std)
    """
    try:
        mod = arch_model(
            y, mean='AR', lags=1, vol='GARCH', p=1, o=1, q=1, dist='t'
        )
        res = mod.fit(disp='off', show_warning=False)
        # 一步向前预测
        fc = res.forecast(horizon=1, reindex=False)
        mu_fc  = fc.mean.values[-1, 0]
        sigma_fc = np.sqrt(fc.variance.values[-1, 0])
        df_t = res.params['nu']  # Student-t 自由度
        resid_std = res.resid / res.conditional_volatility
        return mu_fc, sigma_fc, df_t, resid_std
    except Exception:
        return np.nan, np.nan, np.nan, None


# ============================================================
# 2. Copula 模拟
# ============================================================

def sim_clayton(n_sim, theta):
    """
    Clayton Copula 模拟
    theta > 0, 此处 theta = 7.8878 (from 5.2)
    返回 (u, v) ∈ [0,1]²
    """
    u = np.random.uniform(size=n_sim)
    p = np.random.uniform(size=n_sim)
    v = (u**(-theta) * (p**(-theta / (1 + theta)) - 1) + 1) ** (-1 / theta)
    return u, v


def sim_gaussian(n_sim, rho):
    """
    Gaussian Copula 模拟
    rho = 0.9749 (from 5.2)
    返回 (u, v) ∈ [0,1]²
    """
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    x, y = np.random.multivariate_normal(mean, cov, n_sim).T
    u = stats.norm.cdf(x)
    v = stats.norm.cdf(y)
    return u, v


# ============================================================
# 3. 滚动窗口 VaR 预测
# ============================================================
WINDOW = 500
N_SIM  = 50000
np.random.seed(42)

# 测试期: 从第 WINDOW 天开始 (2020年初起有足够窗口)
# 论文设定: 初始训练 2018-2023, 预测 2024-2025
# 我们用全部历史做滚动, 但仅取 2024 年后的结果用于回测
results = []

for t in range(WINDOW, n_all):
    # 滚动窗口数据
    y500  = r500[t - WINDOW : t]
    y1000 = r1000[t - WINDOW : t]

    # 拟合两边 GARCH
    mu500, sig500, nu500, z500 = fit_garch(y500)
    mu1000, sig1000, nu1000, z1000 = fit_garch(y1000)

    if np.isnan(mu500) or np.isnan(mu1000):
        continue

    # --- 用预校准 Copula 参数 (来自 5.2 节) ---
    # Clayton θ = 7.8878
    u_clay, v_clay = sim_clayton(N_SIM, 7.8878)
    # Gaussian ρ = 0.9749
    u_gauss, v_gauss = sim_gaussian(N_SIM, 0.9749)

    # 逆 Student-t PIT → 标准化残差
    z500_clay = stats.t.ppf(u_clay, df=nu500)
    z1000_clay = stats.t.ppf(v_clay, df=nu1000)
    z500_gauss = stats.t.ppf(u_gauss, df=nu500)
    z1000_gauss = stats.t.ppf(v_gauss, df=nu1000)

    # 模拟收益率 (式 7)
    r500_sim_clay = mu500 + sig500 * z500_clay
    r1000_sim_clay = mu1000 + sig1000 * z1000_clay
    r500_sim_gauss = mu500 + sig500 * z500_gauss
    r1000_sim_gauss = mu1000 + sig1000 * z1000_gauss

    # 等权组合收益率
    rp_sim_clay  = 0.5 * r500_sim_clay  + 0.5 * r1000_sim_clay
    rp_sim_gauss = 0.5 * r500_sim_gauss + 0.5 * r1000_sim_gauss

    # VaR (式 8): 取 α 分位数的负值
    var_clay_99  = -np.quantile(rp_sim_clay,  0.01)
    var_clay_95  = -np.quantile(rp_sim_clay,  0.05)
    var_gauss_99 = -np.quantile(rp_sim_gauss, 0.01)
    var_gauss_95 = -np.quantile(rp_sim_gauss, 0.05)

    # 实际组合收益率
    rp_actual = 0.5 * r500[t] + 0.5 * r1000[t]

    results.append({
        'date': dates[t],
        'rp_actual': rp_actual,
        'var_clay_99':  var_clay_99,
        'var_clay_95':  var_clay_95,
        'var_gauss_99': var_gauss_99,
        'var_gauss_95': var_gauss_95,
    })

    if (t - WINDOW + 1) % 100 == 0:
        print(f'  Progress: {t - WINDOW + 1} / {n_all - WINDOW}  '
              f'({dates[t].strftime("%Y-%m-%d")})')

res_df = pd.DataFrame(results)
res_df = res_df.set_index('date')

# 只保留 2024-01-01 ~ 2025-12-31 的预测 (论文测试期)
res_df = res_df.loc['2024-01-01':'2025-12-31']
print(f'\nTest period: {res_df.index[0].strftime("%Y-%m-%d")} ~ '
      f'{res_df.index[-1].strftime("%Y-%m-%d")}, {len(res_df)} days')

# ============================================================
# 4. Kupiec POF 回测
# ============================================================
def kupiec_test(actual_loss, var_series, alpha):
    """
    Kupiec POF 检验
    actual_loss: -rp_actual (损失, 正数表示亏损)
    var_series:  VaR 预测值 (正数)
    alpha:       显著性水平 (0.01 或 0.05)
    """
    violations = (actual_loss > var_series).sum()
    T = len(var_series)
    p_hat = violations / T
    p = alpha

    # LR 统计量 (式 9)
    if violations == 0:
        lr = 2 * T * np.log(1 - p) - 2 * T * np.log(1 - p_hat)  # 近似
        lr = -2 * (T * np.log(1 - p) - (T * np.log(1 - p_hat)))
    elif violations == T:
        lr = 2 * T * np.log(p) - 2 * T * np.log(p_hat)
    else:
        lr = -2 * (
            (T - violations) * np.log(1 - p) + violations * np.log(p) -
            (T - violations) * np.log(1 - p_hat) - violations * np.log(p_hat)
        )
    lr = max(lr, 0)
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return violations, T, p_hat, lr, p_value


# 实际损失 = -组合收益率
loss = -res_df['rp_actual'].values

print('\n' + '='*70)
print('一、VaR 描述性统计 (测试期 2024.01 ~ 2025.12)')
print('='*70)

var_stats = []
for name, col in [('Clayton Copula 99%', 'var_clay_99'),
                   ('Clayton Copula 95%', 'var_clay_95'),
                   ('Gaussian Copula 99%', 'var_gauss_99'),
                   ('Gaussian Copula 95%', 'var_gauss_95')]:
    v = res_df[col].values
    s = {'模型': name, '均值': np.mean(v), '最小值': np.min(v),
         '最大值': np.max(v), '标准差': np.std(v)}
    var_stats.append(s)
    print(f'{name:20s}  Mean={np.mean(v):.4f}  Min={np.min(v):.4f}  '
          f'Max={np.max(v):.4f}  Std={np.std(v):.4f}')

var_stats_df = pd.DataFrame(var_stats)

print('\n' + '='*70)
print('二、Kupiec POF 回测检验')
print('='*70)

backtest_results = []
for name, col, alpha in [('Clayton Copula', 'var_clay_99', 0.01),
                          ('Clayton Copula', 'var_clay_95', 0.05),
                          ('Gaussian Copula', 'var_gauss_99', 0.01),
                          ('Gaussian Copula', 'var_gauss_95', 0.05)]:
    n_v, T, p_hat, lr, p_val = kupiec_test(loss, res_df[col].values, alpha)
    result = '通过' if p_val > 0.05 else '未通过'
    backtest_results.append({
        'Copula 模型': name.split()[0],
        '置信水平': f'{int((1-alpha)*100)}%',
        'α': alpha,
        '检验天数 T': T,
        '期望穿透': int(T * alpha),
        '实际穿透 N': n_v,
        '穿透率': f'{p_hat:.4f}',
        'LR 统计量': f'{lr:.4f}',
        'P-value': f'{p_val:.4f}',
        '结论': result
    })
    print(f'{name:20s}  alpha={alpha:.2f}  T={T}  expect={int(T*alpha):3d}  '
          f'actual={n_v:3d}  p_hat={p_hat:.4f}  LR={lr:.4f}  P={p_val:.4f}  [{result}]')

backtest_df = pd.DataFrame(backtest_results)

# ============================================================
# 5. 保存结果 (先保存再打印, 防止编码问题导致数据丢失)
# ============================================================
out_dir = PROJECT_ROOT / 'results'
out_dir.mkdir(exist_ok=True)
var_stats_df.to_csv(f'{out_dir}/table_var_stats.csv', index=False, encoding='utf-8-sig')
backtest_df.to_csv(f'{out_dir}/table_kupiec.csv', index=False, encoding='utf-8-sig')
res_df.to_csv(f'{out_dir}/var_daily_series.csv', encoding='utf-8-sig')

import sys
sys.stdout.reconfigure(encoding='utf-8')
print(f'\nDone. Files saved to: {out_dir}/')
print('  - table_var_stats.csv')
print('  - table_kupiec.csv')
print('  - var_daily_series.csv')
