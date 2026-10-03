# RoboTwin / Sidney π0.5 / WM：next-state 与接口审计

核验日期：2026-10-03。只读本地证据、官方源码、论文与 HF 元数据；本文研究子任务未 SSH、下载权重、安装环境或运行训练。主线程另提供了深圳2当日只读源码回执，已读入核验。

## 先给结论

**next-state 是闭环必须补齐的一环，但本项目未必需要先训练一个“真实关节状态预测器”。** 深圳2实时源、本地审计源码和当前官方 `RLinf_support` 均显示：策略接收的 14D `joint_action.vector`，是 **12 个关节的 drive target + 2 个夹爪缓存命令**，并非 14D 实测关节位置。此前将其概括成“必须预测实测 qpos”的说法应修正。正常完整执行 chunk 时，用末个已执行绝对动作构造下一状态是合理的低成本候选；仍须核对 TOPP、提前成功、失败、夹爪裁剪与时间边界，不能直接宣称完全等价。

**RLinf + OpenDW + WorldArena 目前是可组装的研究路线，不是三包拼接即可开训。** 除 state，优先解决的硬接口还有：Sidney C50 与 OpenDW 32 动作/8 个未来帧的时间边界、三路相机回拆与策略预处理、任务匹配的奖励，以及 reset/history/terminal/资源装卸。WorldArena 的零 state 不能原样继承。

**现成的真正 state 预测方案有 WEAVER。** 它公开了代码、三组权重、归一化文件，并确实联合预测图像和未来 proprioception；但发布的是 DROID/Franka 8D、两视角、5Hz 配方，仍需 RoboTwin 14D/三视角/控制时序适配与训练。若目标是尽快保留当前 Sidney 协议接通，OpenDW + 经核验的命令状态更新更近；若目标是研究完整的视觉—本体状态联合 WM，WEAVER 比“所有候选都不预测 state”更值得关注。

## 1. 当前 Sidney 的 state 到底是什么

### 策略契约

本地已验收 Sidney adapter 明确：`pi05=True, action_horizon=50, discrete_state_input=True`；`adapt_to_pi=False, extra_delta_transform=False, use_quantile_norm=False`。因此它使用状态 token，动作走原始 14D 绝对关节目标，不做通用 OpenPI 的关节 delta/absolute 变换。不要把 WorldArena 通用 π0.5 的 delta 配方套到 Sidney 上。

源码：[Sidney dataconfig](../../../references/sz_sidney_pi05_adapter_20260903/work/rlinf/models/embodiment/openpi/dataconfig/__init__.py) L454–466；协议历史路由：[OPENDW_PI05_GRPO_CONTEXT.md](../OPENDW_PI05_GRPO_CONTEXT.md)。本节只引用其模型输入协议，不沿用早期模板的 global batch / noise 等训练预算。

### 环境数据链

深圳2实际 RoboTwin HEAD 为 `0008ae6800df9f75fc8de7098bacb01735fd8fd2`，tracked clean。实时摘录确认 get_obs L512–519、robot getter L528–558、set_arm_joints L643–651、set_gripper L662–693 与下述语义相同。[当日实时源回执](E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/robotwin-pipeline-20261003/sz2/robotwin-state-source.out)。

已核当前官方 RoboTwin `RLinf_support@dca9ec682688821c42944eacd7f5dd9bfb15396f`：

1. `get_obs()` 调 `get_left/right_arm_jointState()` 拼成 `joint_action.vector`；[base_task L510–519](https://github.com/RoboTwin-Platform/RoboTwin/blob/dca9ec682688821c42944eacd7f5dd9bfb15396f/envs/_base_task.py#L510-L519)。
2. 这两个 getter 从 12 个关节读取 `get_drive_target()`；另有 `get_*_arm_real_jointState()` 才读取 `get_qpos()`，当前 observation 链没有调用后者；[robot L528–557](https://github.com/RoboTwin-Platform/RoboTwin/blob/dca9ec682688821c42944eacd7f5dd9bfb15396f/envs/robot/robot.py#L528-L557)。
3. 夹爪维返回 `left/right_gripper_val`。`set_gripper` 先把命令裁剪到 `[0,1]` 并存入该字段，之后才限制 drive target 的单次变化；函数名 `get_normal_real_gripper_val()` 也实际取 drive target，不能按名称当作实测夹爪宽度；[robot L560–570](https://github.com/RoboTwin-Platform/RoboTwin/blob/dca9ec682688821c42944eacd7f5dd9bfb15396f/envs/robot/robot.py#L560-L570)、[L653–692](https://github.com/RoboTwin-Platform/RoboTwin/blob/dca9ec682688821c42944eacd7f5dd9bfb15396f/envs/robot/robot.py#L653-L692)。
4. 深圳2相同固定源 [vector_env.py L56–62](https://github.com/RoboTwin-Platform/RoboTwin/blob/0008ae6800df9f75fc8de7098bacb01735fd8fd2/robotwin/envs/vector_env.py#L56-L62) 将 `joint_action.vector` 原样放入 `state`。本地 `git show 1d015a2aa03ec8132d8207ba47a2be3dbe1d9591` 核验的 Sidney archive：`robotwin_env.py` L189–202 堆成策略 `states`；`openpi_action_model.py` L802–819 传到 `observation/state`（state_indices为空时直通）；`aloha_policy.py` L196–224 在 `adapt_to_pi=False` 下原样保留。后续只有其既有 state 归一化/tokenization，没有换成物理实测状态。核验使用固定 git 对象，未把 dirty worktree 当作部署源。

所以，“state”在此是**策略约定的机器人观测字段**，不等于物理引擎全部状态，也不等于物体位姿。将其改成实测 qpos 即使看似更真实，也是在改既有策略输入分布。

### 为什么仍不能无条件取 `action[-1]`

RoboTwin 默认物理步长 `1/250`，但一个策略动作不是一个物理步。控制器先对目标轨迹做 TOPP，再逐物理子步设置位置/速度目标；失败时可能跳过该臂关节更新，成功时可能提前结束。因此要对齐的是**实际执行后返回 observation 的时刻**，而非固定“50 × 1/250 秒”。[物理步长](https://github.com/RoboTwin-Platform/RoboTwin/blob/dca9ec682688821c42944eacd7f5dd9bfb15396f/envs/_base_task.py#L220-L225)、[chunk 控制路径](https://github.com/RoboTwin-Platform/RoboTwin/blob/dca9ec682688821c42944eacd7f5dd9bfb15396f/envs/_base_task.py#L1803-L1986)。

深圳2固定源还有两个精确细节：`compress_path` 对全不变路径加标准差 `2e-4` 的微扰；downsample 用包含末端的 linspace，因此一般不会主动丢掉轨迹末点。`gen_sparse_reward_data` 在控制前一次性增加整个 chunk 的 action count，即使内部提前成功；不能把该计数当成实际跑完的物理子步数。成功时即时返回当下的 drive target，可能尚未抵达请求的末动作。[压缩与降采样 L444–460](https://github.com/RoboTwin-Platform/RoboTwin/blob/0008ae6800df9f75fc8de7098bacb01735fd8fd2/envs/_base_task.py#L444-L460)、[计数 L1747–1777](https://github.com/RoboTwin-Platform/RoboTwin/blob/0008ae6800df9f75fc8de7098bacb01735fd8fd2/envs/_base_task.py#L1747-L1777)、[提前成功 L1927–1945](https://github.com/RoboTwin-Platform/RoboTwin/blob/0008ae6800df9f75fc8de7098bacb01735fd8fd2/envs/_base_task.py#L1927-L1945)。这支持先做命令态误差审计，而不是承诺 `state_next == action[-1]` 恒成立。

候选更新应为：正常执行后，从末个实际执行/接受的绝对目标更新 12 关节，从经 `[0,1]` 裁剪的命令更新夹爪；终止、部分执行、TOPP 失败分别处理。还应保留真实 reset state。最后走 **Sidney 的 state normalizer**；不可直接把已经按 action stats 归一化的末动作塞进 state。

## 2. 其他公开方案如何闭合 state

| 方案 | 实际公开做法 | 对我们有何意义 |
|---|---|---|
| OpenDW | 接收图像、外部 action 与当前 proprio；返回视频/动作，无 next-state 预测头。交互 demo `prev=[base_state, actions[:-1]]`，推完设 `base_state=actions[-1]` | 末绝对命令近似有官方先例，但 demo 不证明其与我们的控制器、终止和 C50 完全一致 |
| RISE | 三视角生成；online 代码 `next_state=action_pred[:,-1]`，随后写回 policy observation。这里是策略模型空间动作末值，不是独立学习的动力学 state 输出 | 也是动作近似路线；不能复制归一化空间而忽略我们 state/action stats 是否不同 |
| A2World | rollout 的 `state` 是从 action history 计算的 22D 路径特征，LIBERO 公开特征只涉及第一臂；输出多视角视频，不输出 RoboTwin 14D proprio | `state` 这个参数名不代表已经解决 policy next-state；另需状态更新 |
| VLA-MBPO | LIBERO 分支 `discrete_state_input=False`，WM step 保持 `state=current_obs.state`；视频生成头+单腕，state 不参与 π0.5 状态 token | 它通过使用不读 state 的策略绕开问题。Sidney 读 state，不能原样照搬 |
| WorldArena-2.0 | WM env 输出全零 14D state，native RoboTwin 返回环境 state。官方仓库 issue #5 已指出；当前 open，0 comments | 公开代码不是 state 一致性的通过证据；本次没核到作者答复或修复 |
| WEAVER | 视觉 latent 与 proprio token 联合 flow matching；输出下一 proprio，反归一化后送回 π0.5 | 真正学习式 next-state 方案，但现成任务/机器人不是 RoboTwin |

一手来源：

- OpenDW 固定 `e33befa8005a1585e0140dbf464566e90bc79aa1`，当前 main 仍同一提交：[demo L1605–1629](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/playground/online_demos/robotwin_online_demo.py#L1605-L1629)、[外部动作接口 L542–604](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/policy/dw05_policy.py#L542-L604)。
- RISE 当前 main `5fac1e6ab9d50d4cc1ba4daeadf14023c8415955`：[action_pred 与 decoded 区别 L525–529](https://github.com/OpenDriveLab/RISE/blob/5fac1e6ab9d50d4cc1ba4daeadf14023c8415955/policy_and_value/policy_online/rlinf/models/embodiment/openpi_action_model.py#L525-L529)、[next_state L784–794](https://github.com/OpenDriveLab/RISE/blob/5fac1e6ab9d50d4cc1ba4daeadf14023c8415955/policy_and_value/policy_online/rlinf/models/embodiment/openpi_action_model.py#L784-L794)、[写回状态 L313–322](https://github.com/OpenDriveLab/RISE/blob/5fac1e6ab9d50d4cc1ba4daeadf14023c8415955/policy_and_value/policy_online/rlinf/envs/roborl/roborl_env_lerobot.py#L313-L322)。
- A2World：[action history L13–64](https://github.com/LogosRoboticsGroup/A2World/blob/077e10ad6cee07342b5e779f11fea78247584834/world_model/a2world/history.py#L13-L64)、[本轮详细审计](../robotwin_options_20261003/a2world.md)。
- VLA-MBPO：[env L183–225](https://github.com/LAMDA-RL/VLA-MBPO/blob/5e2b21044f2ef2189c22777643cf10e80a83bc13/src/openpi/training/libero_env.py#L183-L225)、[本轮发布审计](../robotwin_options_20261003/vla_mbpo.md)。
- [WorldArena issue #5](https://github.com/WorldArena2/WorldArena-2.0/issues/5)。该 issue 的 delta/absolute 警示针对其通用 π0.5 配置；Sidney 自身 `extra_delta_transform=False`。

## 3. 新补充：WEAVER 确实解决 proprio 预测，但还不是 RoboTwin 插件

论文 [WEAVER, Better, Faster, Longer](https://arxiv.org/html/2606.13672v1) §3.1 明确联合预测未来图像与 proprio；§4 是 Franka Panda + DROID π0.5，使用右外部相机与腕相机。q/action 为 8D（7 个关节 + 夹爪），WM 使用降采样后的 5Hz，发布任务是五个真实机器人操作任务，未发现 RoboTwin 配套实验/权重。

这不是仅有论文声明。当前源码 `arnavkj1995/WEAVER@3e41e90738fea9a7b142db6b2fd3d9ae72b7a82e` 有：

- state 输入投影与独立输出头 `[n_embed→n_states]`；[model.py L340–362](https://github.com/arnavkj1995/WEAVER/blob/3e41e90738fea9a7b142db6b2fd3d9ae72b7a82e/weaver/wm/model.py#L340-L362)。
- 将 state token 解码为 `output['states']`；[L444–467](https://github.com/arnavkj1995/WEAVER/blob/3e41e90738fea9a7b142db6b2fd3d9ae72b7a82e/weaver/wm/model.py#L444-L467)。
- state 与视觉一同从噪声生成，不是输入真实未来 state 作弊；future_states 参数用于形状/占位，其未来段被噪声替换；[L1205–1224](https://github.com/arnavkj1995/WEAVER/blob/3e41e90738fea9a7b142db6b2fd3d9ae72b7a82e/weaver/wm/model.py#L1205-L1224)、[L1389–1428](https://github.com/arnavkj1995/WEAVER/blob/3e41e90738fea9a7b142db6b2fd3d9ae72b7a82e/weaver/wm/model.py#L1389-L1428)。
- 将 state flow loss 纳入训练；[L2011–2015](https://github.com/arnavkj1995/WEAVER/blob/3e41e90738fea9a7b142db6b2fd3d9ae72b7a82e/weaver/wm/model.py#L2011-L2015)。
- 下一次策略查询使用预测末 state；[synth_data_gen.py L270–284](https://github.com/arnavkj1995/WEAVER/blob/3e41e90738fea9a7b142db6b2fd3d9ae72b7a82e/weaver/synth_data_gen.py#L270-L284)。

HF [arnavkj1995/WEAVER@2f8bb6a5cfe9afa8b84b9d42285ab7b08247854c](https://huggingface.co/arnavkj1995/WEAVER/tree/2f8bb6a5cfe9afa8b84b9d42285ab7b08247854c) 已公开 `WEAVER`、`WEAVER-FT`、`WEAVER-ReFlow`，每组含 `checkpoint.pt`（8,257,878,505 bytes）、`config.yaml`、`norm_stats_relabel.json`。FT 配置明确 DROID、两视角、192×320；不能根据权重文件字节数估算运行显存。

移到 RoboTwin，至少还要：14D state/action 输入输出适配；第三相机及 view identity；重新定义动作/图像时间采样；RoboTwin 同步轨迹转换与 norm；目标域视觉和 state 动力学微调；reward/value 按我们的任务语义重新监督。其“joint velocity→position”动作 adapter 是 Franka 专用，不能转给 Sidney 的绝对关节动作。[动作适配代码](https://github.com/arnavkj1995/WEAVER/blob/3e41e90738fea9a7b142db6b2fd3d9ae72b7a82e/weaver/robot/actions.py#L41-L139)。

WEAVER 原策略改进是 best-of-N + advantage 筛选合成轨迹，然后混合真/合成数据 SFT；不是当前 GRPO。可以只复用 WM 并另接 GRPO，但那是新的组合实验，不能直接转述其原论文收益。官方 launcher 用四张 H100；这只是官方配方，不是我们的最低卡数或已测资源结论。[官方 README](https://github.com/arnavkj1995/WEAVER/tree/3e41e90738fea9a7b142db6b2fd3d9ae72b7a82e)。

## 4. RLinf + OpenDW + WorldArena：关键切面完整清单

| 切面 | 当前证据与需要接的内容 | 未解决时会发生什么 |
|---|---|---|
| 模型/任务身份 | Sidney RoboTwin、OpenDW Robotwin bundle、WorldArena 对应任务 RM 必须分别锁源和资产；目前只核 WorldArena adjust_bottle/click_bell 任务包 | “都叫 RoboTwin”也可能任务、相机、机械臂或成功语义不匹配 |
| 动作语义与 norm | Sidney 14D absolute，保留其 actor/model_actions；给 WM 的是 policy 输出反归一化后的 raw actions，再按 OpenDW bundle 自己的 z-score 归一化/裁剪。不能混用两套 stats | 画面看似合理但动作控制方向、幅度错误 |
| 相机/尺寸 | OpenDW 是 H384×W320 拼图：上方主视角256×320，下方左右腕各128×160；回拆后按原 Sidney 三图预处理，保左右顺序/RGB/范围/resize | 直接塞整个拼图作 policy head 或缺腕图，会改变原政策输入协议 |
| 时间 C50 vs 32 | OpenDW 默认32动作、9帧含初始，4动作/未来帧；传50会32+18补齐成64动作效果，默认最后一帧不是t50。保持C50需要专门解决准确边界，不能自动改C32 | 图像、state、reward属于不同时间；action计步失真 |
| proprio/next-state | reset保存真实14D策略state；候选末已执行raw动作更新并用state stats处理。绝不零填或复用当前 state 到所有未来步 | 第二个chunk起 state/图片互相矛盾 |
| reward | OpenDW没有奖励。WorldArena RM须有对应任务、prompt、图像预处理与阈值/时序输入；先核真实数据false-positive与生成轨迹表现 | 把错误reward当学习信号，或其他任务复用失效 |
| reset/history | G8同组共享同一初始三图+state+任务；每个环境独立session、随机种子、图像/动作history；KIR抽样与done重置更新所有缓存 | 同组起点不同、串环境history、旧任务/旧图泄漏 |
| terminal/truncation | 400沿用原生动作预算计数，非视频帧/物理子步。原生成功、RM阈值与超时分清；处理chunk内done、last valid action、final_observation及auto-reset；保留原chunk计数语义 | 成功后继续计reward，越界动作进loss，重置图当final图 |
| GRPO训练信号 | 保原actor、old logprob、valid mask和G8；从WM收环境奖励。若改成短分支/增progress reward或value bootstrap，明确属于方法变化 | 很短的稀疏reward rollout可能全组同分、无有效优势 |
| backend/env接口 | 当前RLinf仅加OpenDW backend不够，env包装仍需三图+14state+reward/done。使用返回张量/逐环境session而不是纯演示视频列表 | 模型能出视频但策略—环境链没闭合 |
| batch/资源 | OpenDW示例偏单例，需明确B/G拆分、session结束、并发队列；onload/offload与actor屏障、峰值监控独立验收 | 多环境缓存爆涨、WM与actor显存重叠；权重可放下不代表可训练 |

OpenDW接口证据：[norm与absolute自动选择 L326–349](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/policy/dw05_policy.py#L326-L349)、[图像布局 L145–181](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/policy/dw05_policy.py#L145-L181)、[chunk/padding/offset L564–602](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/policy/dw05_policy.py#L564-L602)。RLinf 与 WorldArena 的发布/组合边界详见 [framework_and_extra_sources.md](../robotwin_options_20261003/framework_and_extra_sources.md)；已准备文档不代表上述接口已实现或跑通。

## 5. 建议路线：先忠实闭合现有协议，再决定是否学 state

1. **先核真实数据链与命令近似。** 在固定 native RoboTwin 基线的已授权数据中对比：实际返回的 `state_next`、末实际执行 raw action、夹爪裁剪、TOPP成功/失败、实际动作数和 observation 时间。分别统计12关节及2夹爪的误差/尾部/异常，别先设一个拍脑袋容忍阈值。再看两种state喂给冻结Sidney时，输出动作差异是否影响任务；这一步诊断不改正式策略训练。
2. **如果控制命令更新与原state契约吻合，先采用确定性更新。** 对正常执行可精确复刻控制目标，异常和terminal显式处理。保留三图、C50、原始state/action归一化，不把“加state”变成顺便改策略协议。当前C50时间对齐仍是独立前置项，不能因state解决就开训。
3. **只有剩余误差确实影响闭环，再加小型 residual state predictor。** 输入当前state、动作序列与控制时长/状态；拟合`state_next - 命令近似`。若误差来自接触/碰撞且目标字段确为实测状态，则还需视觉/历史条件。该预测器学习的是既有policy state契约，不能悄悄换成不同传感量。
4. **如果目标是长期、接触丰富且视觉—state强耦合的WM，考虑联合state模型。** WEAVER提供可复用的公开架构、训练loss与完整资产；仍需RoboTwin域适配。优点是共同生成；并不保证两者必然物理一致，需同步误差与闭环评测。

去掉 state 也能成为另一条路线，但 `discrete_state_input=False` 会改变 Sidney 原训练输入；需要有意训练/适配与重新建立基线，不是环境适配的小修。

物理 hybrid 也有两种：仅复刻 controller/运动学来更新命令态，是接口复现；若保留完整 RoboTwin 物理引擎执行每步并取真state/reward，则已是“真实仿真 + 生成视觉”的混合环境，算力节省与纯WM问题设定都变了，还可能出现图像和物理接触不一致。只模拟机器人而不模拟物体也不能保证碰撞后的实测qpos。因此 hybrid 可作为独立研究方案，不作为本轮默认接通办法。

## 仍未验证

- 深圳2当日 RoboTwin getter 已实时确认；其他服务器、未来部署及不同 RLinf adapter 版本仍应按实际源核对，不能将此结论无条件外推。
- 末动作近似在各任务/提前终止/控制失败情况下的数值误差；没有运行实验。
- OpenDW 任意长度 chunk、精确t50生成、任务分布外动作响应和实际显存/吞吐。
- WorldArena RM 在 OpenDW 生成图上的校准与完整任务覆盖。
- WEAVER 在 RoboTwin 14D三相机上的可迁移效果、所需数据量和资源；论文Franka提升不能外推。
