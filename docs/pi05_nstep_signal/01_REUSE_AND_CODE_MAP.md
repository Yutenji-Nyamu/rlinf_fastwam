# 复用与代码切口

2026-10-31，只读源码核对；不修改旧训练及旧采集定义。

## 已验收链

|来源|可直接继承|本次差异|
|---|---|---|
|10/17 π0.5 U完整视频|原始模型、50任务实配、逐动作连续录像、末帧、控制器插桩和帧对应|原M10与同噪声M5旁路，增加M1|
|10/19四噪声采集|主样本不动，另外三份噪声合批；保存原始动作而非只有分数|从仅N10扩展到每个N都四噪声|
|10/29 G族shadow|32环境、主B32、额外B96、fork_rng、同观察比较、原图像／状态存档|本次只需同一模型；不加载G族另一套权重|

主要源码：

- `tools/pi05_u_video/noise_samples.py::collect_noise_samples`：保存四份完整初噪声和模型／环境动作；额外三份按sample-major合批。当前额外推理固定`num_steps=10`。
- `tools/pi05_policy_eval/teacher_shadow.py::observe`：隔离旁路RNG、同噪声旁算、保存观察的已跑通模板。当前固定两模型N10；不要把它直接当三预算实现。
- `tools/pi05_policy_eval/run_batch.py`：成熟环境循环、32并行和完整视频开关。shadow分支有N10断言，需新增独立采集模式，不能直接修改旧G族分支含义。
- `tools/pi05_u_video/run_batch.py`：较旧runner固定B16、H50/C50、主N10，记录器仍可复用。
- `rlinf/models/embodiment/openpi/openpi_action_model.py::predict_action_batch`：底层已有正整数`num_steps`覆盖，支持N1；`return_dvac_telemetry=True`已返回各N的初始噪声、链、clean estimate及终点。
- 同模型`sample_actions`／`_sample_actions_with_prefix_cache`：`record_signals=True`被明确限制为N10，并自动增加一个N5旁路。这是旧多信号定义的保护，**本轮不粗暴删除限制**。
- `tools/pi05_signals/signal_math.py`：现成`ugrow_10_vs_5`、`noise_disagreement`、`dv_score`；统计采用真实14维。
- `tools/pi05_u_video/recording.py`、`validate.py`：复用录像、控制路径、末帧和映射；不重写模拟器。

## 最小新增方案

新增独立`observe_nsteps`和对应collector模式：实际执行分支按原batch、原noise走；其余观察分支在`fork_rng`内用同一观察和同一组四噪声，分别调用三个完整求解预算。对每个N调用`return_dvac_telemetry=True, record_signals=False`，即可拿到本轮DV所需的clean estimates，无需开启全层hidden/attention hooks。

额外样本先保留已有B96分块方式；主样本不能为凑B128与额外噪声混合，否则批量数值路径改变。初噪声取原推理经过dtype转换后的`initial_noise_full`，不能重新按seed猜一份。额外推理不改变下一次主决策的Torch/NumPy状态。

N5必须从同一个z0走完5步到t=0；不能截N10的第5个状态冒充N5。N1也完整走到t=0。相同噪声编号在三个预算间一一对应。

视觉语言前缀缓存可复用已有`_sample_actions_with_prefix_cache`，但以主动作/RNG保持一致为门槛。先用成熟公开predict接口完成小改；仅当计时显示重复前缀明显占时再接缓存，不为这项预优化阻塞采集。没有新增训练loss、模型、专家数据、环境动作或接管机制。
