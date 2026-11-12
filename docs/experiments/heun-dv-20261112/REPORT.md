# Euler → Heun：单任务配对 DV 实测

- turn_switch，冻结同一 π0.5；8 条原轨迹、32 个观测，8 份配对噪声，共 256 对动作块。
- N=10、L=5、H=50、D=14；两边都用完整积分步起点的 x−t·v1，取最后 5 份估计的总体方差，再对 14 维求和。
- 逐 chunk 排序相关性平均 **0.9327**，中位数 **0.9574**；10%～90% 分位 **0.8728～0.9874**。
- 最高 20% DV 位置重合 **83.7%**；平均 DV 为 Euler 的 **1.340 倍**。
- 仅比较原轨迹实际执行的位置：排序相关性 **0.9348**，有效配对 248。
- 原生 Euler 与旧 B16 记录对齐；两种采样器首个 clean-action estimate 逐元素相同。每个动作块分别 10／20 次速度网络调用；批量推理耗时比 **1.311**（单轮诊断计时，含数据转换，不是严格速度榜单）。

这只能回答同输入换积分器后 DV 的数值与排序变化；没有执行 Heun 动作，不能据此报告成功率提高。噪声重复不等于新增独立轨迹。所有 32 个观测均可在网页切换查看，未按结果挑选。

原始文件：inputs.npz、euler.npz、heun.npz、queries.json；统计：summary.json、statistics.npz；图：overview.png／overview.svg。

实现参考 [EDM predictor/corrector](https://github.com/NVlabs/edm/blob/main/generate.py)；本项目采用无 sigma 除法的 flow velocity，在终点 t=0 也计算修正，因此是 20 次求值。DV 使用本项目原生 compute_endpoint_variance，Euler 路径不改。
