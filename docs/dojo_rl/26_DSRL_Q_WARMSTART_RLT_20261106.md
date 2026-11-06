# 2026-11-06 · DSRL Q 预热与双环境 Clean RLT

## 当前状态

用户授权本轮自主实现、少量针对性测试、推送和正式运行：4/5 从旧 Clean
转为 DSRL Q 预热；6/7 从旧 +U 转为 Clean RLT。准备期间保留原任务，切换前
登记真实 RLT 回退及精确 UID/PID/start 身份，保留完整旧检查点，不触碰其他用户。

两环境 DSRL 均已通过训练和完整恢复 smoke；RT2 RLT 也已通过并进入正式运行。
Dojo RLT 已完成专属 token 与参考动作验收，短 rollout 回放边界问题修复后继续
smoke/恢复/计时。最新阶段以 `evidence/warmstart-rlt-20261106/status.json` 的时间为准。

| GPU | 任务 | 部署根下目录 | 当前工作 |
|---|---|---|---|
| 4 | RT2 turn_switch / DSRL Clean | rt2-dsrl-clean-g4-r2 | 正式高斯采集，目标 400 轮 |
| 5 | Dojo put_bottles_into_dustbin / DSRL Clean | dojo-dsrl-clean-g5-r2 | 已过训练/恢复，正式启动 |
| 6 | Dojo 同任务 / RLT Clean | dojo-rlt-clean-g6-r2 | 已过token/参考动作，修复短回放后继续验证 |
| 7 | RT2 同任务 / RLT Clean | rt2-rlt-clean-g7-r2 | 正式运行；加速组合不采用 |

部署根 `/data/chenyiteng/deployment-20261106/dsrl-rlt-warmstart`。
输出根 `/data/chenyiteng/results/dsrl-rlt-warmstart-20261106`。
权威路由仍为旧 `deployment-20261105/dsrl-fp32-accel/active-runs.json`，新部署根
同步保存副本。4/5 输出目录不带 `-r2`；后缀只表示评估脚本修复后的调度重试。
RLT/FQL 原队列继续保留，4/7 汇合等待已改为本轮两个新 owner。

## Q 怎样更新、更新多少

DSRL 的 Q 评价的是“观测 + 输入 VLA 的噪声”，不是直接评价机器人关节动作。
本轮先固定原 VLA 的高斯噪声策略：

1. 收集 2000 条**动作块转换**。期间不更新小策略，也不更新 Q。
2. 固定这批经验，只更新 Q 2000 次，每次 batch256。
3. 然后恢复原 UTD20：每新增一条转换安排20次 Q 更新，小策略每2次 Q 更新一次。

critic-only 阶段目标是 `r + continuation_discount * target_Q(next_obs, gaussian_noise)`。
成功/终止和折扣沿用已接受的 flat replay；目标噪声为新的标准高斯，不能用未训练
小策略的噪声代替。只更新在线 Q/其编码器和目标 Q 的 EMA，不更新 actor、α 或其
学习率调度器。之后共同学习才换回学习策略的下一噪声。

| 项目 | 本轮值 | 依据与解释 |
|---|---:|---|
| 收集阈值 | 两环境均2000 | RT2原值；Dojo原500提升到2000，使 Q 有更多不同转换 |
| 额外 Q 预热 | 2000更新 | ResFiT 明确实现 critic-only warmup；其默认1万不直接照搬，当前数据量及 UTD 已较高 |
| Q:actor | 2:1 | TD3 延迟策略更新的简单依据；仍保留现有 SAC target EMA 每步更新 |
| batch / UTD | 256 / 20 | 原正式配置，避免本轮同时改变整套数据重复利用强度 |
| Q ensemble / 聚合 | 10 / mean | 保留原实现，本轮没有额外叠加 min、CQL 或其他目标变更 |
| 精度 / 学习噪声范围 | 小网络和Adam FP32；VLA BF16；±2 | 上轮已修复并验收 |

2000 次并不等于“Q 已经正确”，也不是论文给出的 π0.5 最优值。它是有直接代码
依据的起点，训练后仍需看 Q 对当前噪声的评分、真实成功率和动作分布。预热计数、
bootstrap策略、噪声分组数都写入严格续训契约，恢复后不会重复已完成的预热。
2000×256次抽样，约相当于每条初始经验被抽到256次，因此先不额外增加预热强度。

## 分组噪声：当前结果与限定

每组是32维噪声，沿50个生成时间步展开。50/10/5组分别为1600/320/160维；
紧凑向量用于 actor、Q、log-prob 和 replay，只在进入 VLA 时重复展开。

RT2 已完成：原50组8/16成功，10组0/16、5组0/16，**保留50组**。
首版评估脚本有两处接口问题，均如实保留记录：

- OpenPI 配置是 frozen dataclass，直接赋值失败。未产生训练数据，槽位自动
  回 RLT；改成 `dataclasses.replace` 后重新接管。
- RoboTwin 的 reset 不使用传入的 `env_seeds`，导致降维组第一波仍使用第二波
  的种子。第二波8局能严格配对：原50组3/8，10/5组均0/8，足以支持本轮不启用
  降维。修复脚本为同时设置 `reset_state_ids`，不把16局写成完全配对比较。

Dojo 的 API 使用 `env_seeds`，请求与实际布局一致。50组10/16、10组2/16、5组0/16，
因此也保留50组。这个筛查只决定本轮是否使用降维，不声称统计等价。

## RLT 怎样移植

复用已运行的 RoboTwin RLT 源码 `3f35e89b9a9aa61d66aba2dfead41aebc2ee19cc`：
冻结 VLA、token-only 压缩、已修正的 causal shifted decoder、reference BC、
full_task 延迟接管、compact transition replay、完整优化器/回放/调度恢复。

必要改动只有 Dojo 环境注册、专属 token、Dojo 规范化动作解码、可选加速。
小策略输出规范化的14维动作块；Dojo 的12个臂关节由 OpenPI 输出变换恢复为
当前状态上的增量，两个夹爪保持绝对值，随后由原生适配器做关节排列转换。
decode context 保留当前观测的规范化状态和50×32模板，转换仅执行一次。

训练数据：RT2 原 canonical clean50 的4863帧；Dojo 官方 LeRobot v2.1 视频数据
中的投瓶任务，50条示范20160帧训练、另10条示范中的256帧验证重建。只下载了
240个所需文件，共686895079 bytes，清单固定 HF revision 和逐文件 SHA256。
示范数据不等同于原生仿真的训练/评估布局；后者仍沿用已接受的独立布局池。

Stage1 完全冻结 VLA，训练现有 token 网络2000步，batch32（4×8），FP32，
AdamW 2.5e-5、β=(.9,.95)、warmup100、cosine至0.1倍、clip1，沿用既有 RLT
Stage1 配置。只保存 token 权重及其独立优化器恢复状态，避免再复制整个 VLA。

| Stage2 项目 | 选择 | 原因 |
|---|---|---|
| 环境数 / horizon | 8；RT2 200，Dojo700 | 继承当前双环境配置 |
| chunk | RT2 20，Dojo50 | 与当前DSRL/原VLA控制频率一致；旧RT2 RLT为10，本轮明确改为20 |
| Q / actor batch | 256 | 同DSRL；旧RT2 RLT模板为512，本轮单卡256 |
| 收集 / 接管前更新 | 2000转换 / 5000更新 | 先由VLA采集、小策略学模仿，再接管；不是纯Q预热 |
| 在线更新 / 上限 | 每新增转换5次；每轮最多1600 | 复用RLT调度器，限制单轮积压但不丢弃待更新预算 |
| Q:actor | 2:1 | 沿用既有RLT |
| BC/Q 权重 | 7/.05 → 2.5/.45 | 沿用既有RLT比重；5000更新warmup，之后10000更新过渡 |
| γ | 每物理步 .999 | 原模板.99在Dojo700步只剩.00088；.999则约.496 |
| actor/rollout输出 | 都为identity | 明确复制同一模型配置，避免训练identity、推理默认为tanh |
| 预算 / 评估 | 400轮；每20轮评估RT2 12局、Dojo32局 | 继承当前评估数量和布局，Stage1另计 |

## 加速与验证范围

- DSRL 保留上轮已有效的指标集中、fused Adam、批量Q；RT2用Q编译，Dojo用
  foreach目标更新。本轮不重新试已失败的线程、NUMA、GPU replay、控制状态复用等。
- RLT Stage1 缓存的是现有 token-only `sft_forward` 中 `train=False` 的冻结 VLA
  prefix，按原BF16位模式保存，无额外降精度；首批回读逐位一致。token损失、
  causal mask、全部训练示范均保持。省掉的是2000次训练里重复的VLA前向。
- RLT Stage2只测一组低改动组合：指标集中 + fused Adam + foreach EMA。用真实
  smoke回放做 baseline / combined / baseline-repeat，各4次预热+64次更新；
  每案重置模型、目标模型、优化器、调度器和随机状态。参数最大差异阈值1e-5，
  replay/RNG一致且相对更快的基线至少快3%才保留。不能用该计时冒充整轮加速。

已完成：DSRL CPU预热计数/恢复余量/分组梯度检查；RT2训练与恢复smoke；RLT
两环境配置、模型维度、规范化、梯度、worker导入及foreach EMA CPU逐位检查。
RLT真实动作解码、训练恢复和计时由已登记流水线继续验证，失败自动回真实RLT。

## 实测与修复记录

- Stage1：RT2重建loss 2.6031→0.3138；Dojo 2.7512→0.4864，256帧验证loss
  0.5118。只表示重建训练正常，不表示小控制策略已经学好。
- 参考动作：RT2 6/8、Dojo7/8；两环境 `decode(extract_reference)` 与直接VLA
  推理的最大差异均为0。成功率来自小规模参考策略检查，不是训练后的RLT成绩。
- DSRL恢复：RT2 update_step 48→80，Dojo32→48，两者critic预热计数始终16，
  证明恢复未重复已完成的smoke预热。正式仍是2000，不沿用smoke参数或状态。
- RT2 RLT恢复：update_step 16→32，回放总转换16→32，完整恢复通过。
- RT2 RLT加速：baseline 17.992 ms、combined18.401 ms、baseline-repeat17.518 ms。
  最大在线/目标参数差异均1.1921e-7，随机与采样顺序一致；组合比更快基线慢约5%，
  正式关闭defer_metrics、fused Adam、foreach EMA。只是SAC更新计时。
- RT2 RLT首个smoke因复制actor配置漏了 `rollout.model.model_path=null` 退出。
  修复准备器及CPU断言，6卡预先同步；7卡真实RLT回退后用-r2继续，没有重训token。
- Dojo RLT首个smoke发现旧回放按actions的T+1长度迭代，而奖励和curr/next_obs仅T条；
  末尾是未执行的bootstrap query。改为奖励确定的真实T；CPU检查覆盖非终止短尾、
  提前终止丢弃后续块、纯truncation继续bootstrap。6卡回退后用-r2继续，复用已验收token。
  RT2完整200步回合因first-done截断不触发该边界；已运行的7卡不为此无效重启，
  当前完整回合和修复后行为一致，新源树及后续重启统一带修复。
- Ruff lint/format通过；源码补丁逐文件before/after SHA256及轻量结果随提交保留。

## 代码与来源

工具：`tools/dsrl_rlt_warmstart`。DSRL两个核心文件补丁和RLT八个核心文件补丁
均有before/after SHA256。通过精确源锁应用至独立导出运行树，避免把不同历史的
服务器核心文件直接覆盖公开checkout。默认开关保持旧行为。

- [ResFiT critic warmup代码](https://github.com/amazon-far/residual-offpolicy-rl/blob/main/resfit/rl_finetuning/scripts/train_residual_td3.py)
- [TD3 延迟策略更新](https://github.com/sfujim/TD3/blob/master/TD3.py)
- [DSRL官方代码](https://github.com/nakamotoo/dsrl_pi0) 与 [π0.5重复噪声讨论](https://github.com/nakamotoo/dsrl_pi0/issues/9)
- [RLT文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/rlt.html) 与 [论文](https://arxiv.org/abs/2604.23073)
- [RoboDojo官方数据](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo)
- [PyTorch Adam fused实现](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/optim/adam.py)

每条引用的用途和迁移限制见工具内 `sources.json`。Q预热2000、RLT接管5000和
本轮分组筛查是适配选择，不能表述为以上论文已经验证的最优参数。
