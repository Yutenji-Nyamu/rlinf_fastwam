# 已跑通链路及最小切口

已按用户答复固定：Clean CP140弱 / Uτ3 CP70强，N10/H50/C20、pillbottle、200动作。配置以`tools/pi05_policy_eval/clean_c20_followups.py`和既有G族JSON为基线。直接复用`weak_weights`/`strong_weights`的strict全状态覆盖，不能当作只含动作专家的蒸馏导出加载。

权重：`/data/chenyiteng/models/protected-eval/pillbottle/clean_c17_cp140/full_weights.pt`；`/data/chenyiteng/models/protected-eval/pillbottle/grpo_u_tau3_cp70/full_weights.pt`。原始Sidney构造及归一化与旧接管保持。

## 两个旧实验各借什么

| 旧链路 | 已验收内容 | 本轮复用与改动 |
|---|---|---|
| 20261019四噪声U完整视频 | 16环境；主样本原样执行，额外3噪声合为48样本旁路；动作/RNG不变；完整视频、末帧、控制映射 | 复用采集、噪声批量、录像、原始数组、跳过和图册导出 |
| pillbottle高低U老师接管 | 当前观察算弱策略U；同观察、同一noise seed算强策略动作；当前块选择执行 | 借双模型加载和共噪声，不借门控；教师本轮每个有效query都旁路预测，学生继续执行 |

代码定位（本工作树已有）：

- `tools/pi05_u_video/noise_samples.py::collect_noise_samples`：保留执行样本，3个额外样本合批。当前N10与所选弱模型吻合，不改推理步数；保持样本主序及原dtype。
- `tools/pi05_u_video/run_batch.py`：复用obs保存、noise4、Recorder、query NPZ、episodes/done/validation；其B16/C50硬编码不直接作为C20执行入口。
- `tools/pi05_signals/signal_math.py::noise_disagreement`：既有逐动作四噪声公式直接复用。
- `tools/pi05_policy_eval/run_batch.py`：作为新独立采集入口的基础，已有B32、双模型strict加载、H50/C20分块执行和完整录像。增加显式teacher-shadow模式：每个有效query跑教师并保存，selected全部为false，学生执行；不能只把policy=weak就认为当前need_strong会加载老师。
- `rlinf/models/embodiment/openpi/openpi_action_model.py::predict_action_batch`：现成`noise=`输入及模型/环境坐标导出。显式传主样本完整noise，优于依赖不同推理代码恰好消耗同一随机序列。
- `tools/pi05_u_video/recording.py`、`lane.py`、`validate.py`：复用25Hz完整视频/末帧、有限超时、精确清场；不修改共享原生控制器。
- `docs/pi05_u_video/noise-gallery-20261020/operations`：已接受的全程信号、峰附近真实帧、完整视频和交互图册版式。

## 一次决策做什么

1. 从正在学生轨迹上的环境取当前obs，固定query/slot/seed。
2. 主噪声z0经学生预测，保留实际要执行的动作和完整noise。
3. 另外z1/z2/z3在同一obs下由学生旁路合批预测；不改变主样本或后续随机数。
4. 教师在**同一obs和z0**下预测，保存动作；不向环境提交教师动作。
5. 每个预测位置得到四噪声分散度、主学生与教师动作距离。提交原学生动作的前C步；记录实际执行部分、终止和视频映射。
6. 下一次观察重新开始；不是用上一块的信号来解释这一块。

初版教师1份与z0配对，学生共4份，教师不必4份。若要研究“两种策略分布差异”，才扩为4对同噪声；那是额外计算和不同问题。

## 一次短验收即可覆盖的事项

同种子开/关旁路，执行动作及下一次主噪声不变；教师噪声与学生z0逐项相同；不同槽位/噪声排列正确；只真实14维计算；离线从原数组重算两条信号一致；视频末帧与实际结束一致。先无卡数组测试，再在将来授权空卡做一小批，计入正式预算；不重写仿真、不让正式全批承担返工。

优先扩展已经支持B32/C20的policy_eval入口，把已验收noise_samples函数接过来；不重写仿真和控制映射。原U10/5仍可一起保存，为旧接管信号提供对照。
