# Copula–GARCH 联合尾部风险与 VaR

中证500与中证1000组合的条件波动、联合尾部风险和 Kupiec 回测。

## 内容与运行

`src/copula_garch_var.py` 保留课程研究的滚动模型，公开版改用仓库相对路径与 CSV 输入；`notebooks/` 保留去除输出的探索笔记；`historical_results/` 为原研究结果。

```bash
python -m pip install -r requirements.txt
python src/copula_garch_var.py
```

Python 3.10+。输入 `data/returns.csv` 为原工作簿中的指数市场序列，`r_500`、`r_1000` 是日对数收益率乘以100。运行包含滚动拟合与蒙特卡洛计算，耗时取决于样本长度。

## 方法与限制

代码实际均值模型为 AR(1)，波动部分为 GJR-GARCH(1,1)-Student-t，联合结构比较 Clayton 与 Gaussian Copula。原稿出现过 ARMA(1,1) 称呼，以代码实现为准。

Copula 参数是原研究中预设的固定值，尚未在本次整理中重新验证训练期估计及信息可得性；Student-t 残差的尺度归一化和 Kupiec 边界情形仍需研究复核。历史输出按原论文的 2024—2025 测试区间存档，不能据此声明完整样本外有效性。本仓库研究底层资产组合，不包含雪球产品完整收益结构。

## 项目说明

维护者：[lhowd301-lab](https://github.com/lhowd301-lab)。本仓库整理自既有课程、研究或工具项目，开发与文档整理使用了 AI 辅助。代码可运行性与研究结论有效性分别说明，不将历史模拟结果视为实盘业绩。
