# 已有GEO、DV和多噪声素材

2026-10-31，本轮实际读取旧条目并查看两张GEO图。

## 两张已找到的GEO图

基目录：`docs/pi05_signals_20261003/offline-analysis/visualizations/`。

|任务|轨迹与seed|图|可以从图中描述的现象|
|---|---|---|---|
|adjust_bottle|adjust_bottle_b1_s01；100100034；成功|figures/adjust_bottle/geo.png（另有svg/geo_curve.png）|第三个动作块约100–150位置有明显宽峰；对应持瓶调整阶段|
|beat_block_hammer|beat_block_hammer_b0_s01；100150015；成功|figures/beat_block_hammer/geo.png（另有svg/geo_curve.png）|第一个块有较强变化；帧显示接近/抓握锤子，不能把峰直接说成敲击瞬间|

两张均为10/03旧采集，视频各9帧，属于块前后快照而非后来完整逐动作视频。横轴是提交的动作位置；橙色末块的真实执行前缀未知。不能把图中每个点都称为已执行动作，也不能把块前后帧精确对应到峰的瞬间。

已确认本地材料：`entries/<task>/geo.json`有轨迹、seed、峰位置、原服务器视频路径，PNG/SVG图存在。条目还记录`series/<task>/<episode>.npz`数据引用，但本轮在图目录下未找到对应NPZ，重算前需定位原离线包/服务器数组，不能把该相对引用当已下载数据。图为既有产物，本轮没有重新采集。

注意：同任务旧图册里的geo和dv通常**挑的是不同轨迹**。例如调瓶子geo为b1_s01，dv为b1_s14；敲方块geo为b0_s01，dv为b1_s14。因此不能用两张精选图的外观宣称GEO和DV相关。需要读取同一episode的两列再比较。

## 两种量为什么看起来有关系

- GEO：同一次生成里，相邻去噪步骤的速度方向/大小改变多少；除以速度规模。
- DV：同一次生成里，后几步各自预计的最终动作是否一致。
- 两者都看**同一条内部生成路径**，关注对象分别是速度与终点估计，因此相关但不等同。
- 一步只有一个速度/终点估计，无法估计跨步骤变化。记录NaN/valid=false，不记录0。
- 多噪声看不同生成起点的最终输出分散；它是另一个轴，一步仍可计算。

## 已验收代码可以复用什么

- `tools/pi05_signals/signal_math.py`：`geo_full`、`dv_score`、`noise_disagreement`。GEO保留H动作轴，只在14维上聚合。
- `tools/pi05_u_video/noise_samples.py`：同观察K4合批、样本顺序与额外噪声。
- `tools/pi05_nstep_signal/observe.py`：N1/N5/N10同噪声比较、相对/绝对分散度、DV缺测、旁算保留随机状态。
- `rlinf/algorithms/pi05_distillation/losses.py::endpoint_loss`：训练里已有的同噪声师生端点，新增D不必另跑一次正常老师。

GAD若选择新的扰动信号，只替换额外诊断前向部分；原样保留已跑通的模型/观察/动作/录像映射。训练内直接复用缓存前缀，避免把离线采集器整套搬到训练热路径。
