# 附录绘图材料总索引

## 本轮整理

- [逐动作图、原始补采数组、特征、统计、时间报告与重绘脚本](https://github.com/Yutenji-Nyamu/rlinf_fastwam/releases/download/appendix-signals-20261111/appendix-action-figures-and-raw-20261112.tar.gz)
- [分析报告](REPORT.md)
- [时间报告](TIMING.md)

## 既有原始材料

- [appendix-reviewed-figures-20261112.tar.gz](https://github.com/Yutenji-Nyamu/rlinf_fastwam/releases/download/appendix-signals-20261111/appendix-reviewed-figures-20261112.tar.gz) — 4.8 MiB
- [appendix-simulation-arrays-and-figures-final.tar.gz](https://github.com/Yutenji-Nyamu/rlinf_fastwam/releases/download/appendix-signals-20261111/appendix-simulation-arrays-and-figures-final.tar.gz) — 1391.0 MiB
- [appendix-simulation-arrays-and-figures.tar.gz](https://github.com/Yutenji-Nyamu/rlinf_fastwam/releases/download/appendix-signals-20261111/appendix-simulation-arrays-and-figures.tar.gz) — 71.6 MiB
- [appendix-value-analysis-20261112.tar.gz](https://github.com/Yutenji-Nyamu/rlinf_fastwam/releases/download/appendix-signals-20261111/appendix-value-analysis-20261112.tar.gz) — 14.4 MiB
- [heun-dv-turn-switch-20261112.tar.gz](https://github.com/Yutenji-Nyamu/rlinf_fastwam/releases/download/appendix-signals-20261111/heun-dv-turn-switch-20261112.tar.gz) — 36.4 MiB

## 目录与复现

- `signals.json`：逐动作曲线、chunk边界与帧索引。
- `public/`：PNG、SVG、PDF、CSV及原帧。
- `raw/dojo-dense/`：全部保存观测上的K4补推理。
- `raw/grbc-controlled/`：受控外观干预RGB、物理状态、MCR特征。
- `raw/grbc-native-rand/`：官方静态随机化规则下的预设种子样本。
- `training-configs/`：新400轮配置与差异。
- `round-timing.csv` / `inference-cost.csv`：时间统计原表。

解压后运行 `python render_signals.py --data .` 重画仿真信号图；模型推理已完成，重画只需CPU。采集脚本依赖原RLinf/RoboTwin运行环境。

真机图与53个原始窗口归档在原私有数据仓库；本公开包只包含仿真材料。
