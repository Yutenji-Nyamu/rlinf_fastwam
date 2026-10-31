# N1/N5/N10联合信号：实现与执行

2026-10-31。当前阶段：源码完成、服务器CPU检查通过，真实GPU首批待验收。

## 增量与依据

复用`tools/pi05_u_video/run_batch.py`、`noise_samples.py`、`recording.py`及`signal_math.py`。新`tools/pi05_nstep_signal/observe.py`把三预算组合在一次观察内；主推理`record_signals=False`避免旧自动N5旁路重复计算。主N10输出复用，额外K3先生成并固定，N1/N5/N10共用每一份完整32维初噪声。所有旁路恢复CPU/CUDA RNG，不影响下一次主推理。

保存K4×N3的模型空间与环境空间最终动作，真实14维；k0每个N的x链、z=x−t·v链、实际时间网格；完整32维初噪声、三相机query观察、指令、种子、成败、完整头部视频、逐物理步日志和两臂动作/帧投影。N1 DV为NaN且valid=false，不能当0。DV5看全5次、DV10看后5次，时间窗口不同。相对分散度与绝对分散度同时保留，均不称概率熵；N10也不是真值。

机器人仍H50/C50、原Sidney多任务权重。50任务各32官方旧种子，继承每任务原动作上限。合并旧两批16为32改变同任务种子对应的噪声批排列，不能声称逐轨迹复现旧16并行；本次比较在同一B32观察和噪声内。nativeN10执行与开启旁路的动作/RNG验收要求完全一致。

## 配置与启动

- 计算和图形仅物理6/7；`CUDA_VISIBLE_DEVICES=DOJO_GPU_SCOPE=卡号`，复用已验收gpu-scope shim。
- 每卡32环境，三个预算依次算；额外K3实际合批比较B96与B48各两次（预热+计时），取合适速度，保留测量。环境/显存OOM仅把该32种子分两批16重试一次。
- 源码：`/home/chenyiteng/deployment-20261031/pi05-nstep-sz1-g67-v1/source`。
- 产物：`/home/chenyiteng/results/pi05-nstep-video/20261031-v1`；利用Home余量，首批统计真实输出量再估总存储。
- 命令：既有RoboTwin venv的Python运行`tools/pi05_nstep_signal/prepare.py D OUTPUT REV`，之后`launch.py D`；每条lane调用已跑通`pi05_u_video/run_batch.py config.json`。
- 单任务3600秒、900秒无进度为超时；原生仿真报错记缺测继续下一任务，正常未成功计失败；原始数组/视频验收失败停止本lane并交回RLT。输出盘不足2GiB时停止新增采集，已存在的RLT/无保存兜底接回。当前4/5训练不改。
- 交接前写实配、测试、lane；复用`priority_pair_handoff`精确停6/7的RLT driver并登记最近完整恢复点。新增可选逐卡reservation，6/7可独立完成并接回各自RLT，无需等另一卡。

## 无GPU检查

服务器同一venv、`CUDA_VISIBLE_DEVICES=''`运行`pytest -q tools/pi05_nstep_signal/test_observe.py`：2 passed，1.42秒；8个改动Python文件语法编译通过。

检查同噪声三个完整solver、合批顺序/拆分一致、主动作/RNG保留、原数组重算、N1 DV缺测、无效18维不影响评分。真实录像复用原验证：所有视频覆盖metadata和解码末帧、实际物理步映射，避免每次重复解码全视频。

## 发布边界

公开源码、配置/种子、进度、成败、CPU与真实smoke验收；视频、观察、原始大数组留服务器。其他用户存储和系统网络原始快照不入公开仓库。
