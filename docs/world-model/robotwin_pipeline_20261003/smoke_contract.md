# OpenDW / RoboTwin / Sidney GRPO 首次smoke合同

更新：2026-10-03。用户已选`adjust_bottle`与C32，允许接好直接短smoke及资源试配。本文记录实施素材和验收口径；是否已启动、卡位与结果须看本次owner现场回执，本文不冒充动态状态。

## 1. 直接继承什么

最近RoboTwin Clean工程基点为`2151a08ee1bd75df1bef0d8190e594bd5c7f7977`，纯Control历史来源为`1d015a2aa03ec8132d8207ba47a2be3dbe1d9591`。本轮已在本地`worktrees/pi05`逐固定Git对象核读二者，纠正旧文档“2151本地无对象”的过时记录；没有把当前dirty工作树当部署源。

| 项目 | 最近RoboTwin Clean实配 | 本次初始smoke候选 |
|---|---|---|
| 策略 | Sidney π0.5，三图＋14D命令state，14D绝对动作 | 保持 |
| 预测/执行/去噪 | H50/C50/M10 | H50/**C32**/M10 |
| 同时轨迹/组 | N32/G8，每环境rank整数个组 | **先N8/G8**，显存允许再N16/N32 |
| 串行采集 | R8，每条200动作 | **R1，每条32动作，一个chunk** |
| actor | 原两rank、GB512/MB32、U2 | 先物理4单rank、GB8/MB1、U2 |
| 算法 | GRPO、Flow-SDE noise0.5、LR5e-6 | 保持；Clean observe仅关闭额外DV记录 |
| runner/保存 | 历史续训至200，每10轮存 | **1轮、保存CP1**，独立新输出 |
| 物理评估 | 原任务固定32例 | 本次关闭；短smoke不测任务收益 |

N32是RoboTwin最近工程实配，Wan/LIBERO正式N64来自另一配方，二者不能混为“原并行”。用户当前允许按资源试配，因此先单卡N8；并行扩到N16/N32不改变每组G8。真实服务器是否容得下、有没有借到卡，须读本次测量与owner，不据模型磁盘体积猜显存。

历史已跑通两卡Sidney smoke：2026-09-03 `64 env × R1`、G8、400动作/C50=8chunk、GB512/MB32/U2；采集、两次optimizer、评估、完整CP保存均exit0，训练成功10/64。这证明同一策略/GRPO训练接口可复用，不证明新OpenDW推理已经通过。[历史验收](../../rlinf-shenzhen-multitask-pi05/evidence/CURRENT_RLINF_ADAPTER_IMPLEMENTATION_LEDGER_20260903.md)

## 2. 为什么一chunk需同步改actor batch

固定actor代码先将时间和环境展平，要求：

$$
B_{global}\ \bmod\ (B_{micro}\times W_{actor})=0,\qquad
S_{rank}\ \bmod\ (B_{global}/W_{actor})=0.
$$

其中本次样本槽数$S=N\times R\times(L/C)=N$。因此N8/R1/L32/C32只有8个chunk样本，不能仍填GB512或2048。初始单rank取global8/micro1，累积8；U2表示重复训练2遍，一runner轮有2次optimizer调度。“串行一chunk”“runner一轮”“optimizer一次”是三个不同数字。

若后续N32四actor rank，global32/micro8可用；若N8两actor rank，global8/micro4可用。为了先降低激活峰值，可继续micro1并累积。环境rank另须满足$N/(W_{env}\times stage)$能被G8整除；N8不能分给4个各2环境的EnvWorker。

代码证据：固定2151的`rlinf/workers/actor/embodied_fsdp_actor_worker.py`内初始化整除断言及`run_training`展平/切分；`rlinf/config.py`每EnvWorker组大小断言。这里是配置数学，不是新方法选择。

## 3. 哪些可直接实现，哪些留给smoke读数

可直接实现：三相机拆拼、14D raw action传入、各自stats归一化、C32的第32动作末帧、命令state更新、G8同reset、每槽独立history、reward/done到动作位置、WM卸载等待、资源监控。对应纯adapter与环境已经独立写入实施素材。

需用本次读数确定：OpenDW完整加载/一次推理峰值；同卡actor与WM阶段切换能否容纳；N8是否获得组内reward差异。它们不要求先做一串额外物理实验，可在直接smoke中留输入输出与峰值判断。

GRPO有效学习要求至少一个组的回报不同、保留mask非零、优势/梯度有限且有非零有效值。当前RM是连续score，**8条都未达到0.9成功阈值，不等于没有学习信号**：末分数不同且组均值处于[0.1,0.9]仍可留下优势。全组同分则相对优势为0；整组均值不在保留区间会被过滤。此时如实分开：

- `pipeline_completed`：策略→WM→奖励→轨迹→actor循环→保存完成。
- `learning_signal_verified`：另需真实非零优势和有效GRPO梯度。

不会为宣称成功而伪造reward、预塞成功样本或静默关闭过滤。零梯度也不能直接宣称参数不变，AdamW动量/衰减可能仍改变参数。

初始`prev_step_reward=0`沿WorldArena原实现，reset不先调用RM。因此一个C32块的差分累计回报为末图score（本配置系数1），不是末图score减真实reset图score。它是当前继承语义；同一G8起点的共同常量会在组内中心化中抵消，但原均值过滤阈值仍按该回报口径。不要将WM分数均值称作原生成败率。

## 4. 文件与继承入口

| 文件/入口 | 工作 |
|---|---|
| `examples/embodiment/train_embodied_agent.py` | 保持原训练入口 |
| `rlinf/runners/embodied_runner.py` | 只补`env_handle.wait()`，阻断WM卸载/actor装载重叠 |
| `rlinf/workers/env/env_worker.py` | 原chunk采集与offload接口复用 |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | 原Sidney采样、logprob、权重同步复用 |
| `rlinf/workers/actor/embodied_fsdp_actor_worker.py` | 原GRPO更新、FSDP、保存复用 |
| `rlinf/envs/world_model/opendw_adapter.py` | raw动作、三图、命令state、帧/动作奖励映射 |
| `rlinf/envs/world_model/opendw_robotwin_env.py` | reset/session/HTTP WM/reward、done与offload |
| `rlinf/envs/__init__.py` | 新环境类型与类注册 |
| 独立OpenDW+RM HTTP服务 | 同卡模型装卸与CPU返回；不合并三套仓库 |
| 独立smoke YAML及合同 | 任务、N/G/R/L/C、角色卡位、微批、唯一输出 |

现有`local_scripts/wan_goal_20261003/private_ray_driver_monitored.py`可复用“明确私有Ray地址、唯一namespace、placement核验、资源监控”的设计，但文件硬编码4–7四rank、LIBERO与`wan_goal_`命名，**不能不改直接启动本次GPU4单卡**。`resource_monitor.py`可复用只读GPU/RSS/PSS采样，管理进程清单须包含独立WM/RM服务；`resource_telemetry.py`的阶段hook设计可复用，旧`prepare_phase_barrier.py`锁的是Wan源SHA，不适用于Sidney，已另写2151固定源补丁生成器。

当前配置/runner实施素材：[说明](../../../local_patches/opendw_smoke_20261003/CONFIG_AND_RUNNER.md)、[配置生成器](../../../local_patches/opendw_smoke_20261003/build_smoke_config.py)、[runner补丁生成器](../../../local_patches/opendw_smoke_20261003/prepare_runner_barrier.py)。这些脚本不自行借卡或启动训练。

## 5. 唯一owner与停止/归还

本次若只借深圳3 GPU4：先实时核该卡当前RLT的UID/PID/start、当前owner与完整CP，写新cycle/newowner，不重放09-30/10-01旧Wan桥或CP125恢复链。5–7卡及共享Ray保持。父owner负责覆盖成功、OOM/异常和用户停止路径，按精确进程身份停止本次driver、私有Ray与WM服务，核计算与图形上下文释放后，沿刚冻结的原RLT合同续跑并验首轮。

旧可复用安全机制在`local_scripts/wan_goal_20260930/resource_switch/common.py`、`continue_pipeline.py`以及本轮`rlt_timeout_recovery_20261003/recovery_owner.py`：UID/PID-start/boot、pidfd、唯一回执、checkpoint完整性、release→restore→first-round顺序。**复用机制，不复用旧硬编码路径、旧owner身份或旧checkpoint。** 0–3不作候补卡；底层残余图形上下文必须如实记录。

停止条件：本smoke一runner轮正常完成；实际OOM/推理错误/NaN等故障；用户停止；唯一owner身份/资源不符。一次故障后调整只针对已证实的接口或容量问题，下一次用新attempt保留前次证据。完成不会自动拉长正式训练。
