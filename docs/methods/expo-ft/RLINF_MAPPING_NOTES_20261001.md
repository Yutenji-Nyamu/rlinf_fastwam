# EXPO-FT → RLinf / RoboTwin / π0.5：接入审查

核查日期：2026-10-01。本文是静态源码映射及实施建议，没有运行项目测试、安装依赖或修改算法。现场证据由主审查通过固定host-key SSH只读取回；下列有日期的本地代码快照与此次现场证据分开使用。

## 1. 判断

可以在我们已经跑通的 RLinf + RoboTwin + π0.5 中独立实现 EXPO-FT；已有环境、模型转换、归一化、chunk 执行、日志评估、checkpoint 与 off-policy 调度能复用。**不能把 RLT 或 DSRL 改一个 loss / actor 输出维度就称为完整 EXPO-FT。** 最低闭环包含：多份 base chunk、action-space edit、rollout 和 TD backup 均候选选优、真实动作 chunk replay、base 在线 flow-matching 更新。

建议新建独立方法分支和配置，从已跑通的 π0.5 控制合同出发。先做同步原版 EXPO-FT；Real-Time 的旧观测候选、执行队列、prefix inpainting、延迟 backup / noise-Q 是另一套范围。此次用户授权是调研和上下文维护，尚未实施算法或派发实验。

## 2. 证据身份

| 源 | 身份 / 日期 | 本文使用范围 |
|---|---|---|
| EXPO-FT 官方发布源 | `local_scripts/expo_ft_20261001/official-repo`，HEAD `023cf9cfcb09dab962b6e806fea47d2954b2b9bb` | original `EXPOLearner`、base / edit / Q / replay 合同；与实时类分开 |
| 本地 π0.5 接口 worktree | `worktrees/pi05-stage`，HEAD `20672beeb415ff7f8d004a329009401b22f4967a` | `openpi_rlinf` PyTorch SFT / eval、prefix cache；这里旧 `RLTMLPPolicy` 仍写 `tanh`，不能据它覆盖当前 `identity` |
| 已登记当前 RLT 配置 | [N8 恢复记录](../../server-admin/RLT_N8_FULL_RESTORE_20260929.md)，2026-09-29 | π0.5、C10、200动作、student `identity`、五项完整 N8 调度；作为有日期的实配记录 |
| RLT transition / worker 快照 | `local_scripts/task_switch_20260923/source/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py`，2026-09-23 | 已接 RoboTwin truncation / transition / strict resume 的 worker；不是浮动上游 main |
| DSRL 私有移植快照 | `.tmp/dsrl_current_impl_20260823`，2026-08-23 | macro projection、32D latent-Q、compact ring / target shadow / sidecar；旧 π0 路线，非已验收 π0.5 DSRL |
| 深圳1 fresh source / model合同 | 主审查2026-10-01 09:59:41固定host-key只读确认；HEAD `55c1399a50826d61e8735a64daa2f1742f1b824f`，branch `codex/sz1-pi05-rlt-n8full-20260929`，clean | 现役actor=`rlt_mlp_policy` / identity，8环境、C10/14D、200动作；frozen `pi05_sidney_robotwin` feature base为H50/ODE10。四条Stage2 driver exact identity alive；Stage1已退出，其resolved没有algorithm/env属于SFT配方，不能当现役Stage2 overlay |

此次现场 worktree：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-rlt-n8full-sz1-20260929`。证据 完整只读源码与实配 JSON（未随本包发布的本地历史索引：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/server-review/sz1/expo-ft-source-files-20261001-1003.json`）。JSON实际时间为09:59:41，文件名label不充当采样时间。现役Clean实配SHA：支架`6382d29d8f34cbe149d0c2cdaf16a730228d9e442902b56dde6b9c30064b9525`，双瓶`74bfeedb9c92b111cf2f01aa778b58dad9645d5bc5fe80df773d174fa5bfa546`。

官方项目：[EXPO-FT](https://pd-perry.github.io/expo-ft/)。论文与方法边界见 [原论文笔记](RESEARCH_PAPER_NOTES_20261001.md)。8 base + 8 edits = 16 是论文配方；实现时 N / C / H / UTD 必须作为方法预算列出，不能把现有 RLT 的每轮 N8 等同于候选 N8。

## 3. 三种小策略的动作空间

| 路线 | Actor 真正输出 | Q 的 action 输入 | Base 是否直接参与在线梯度 |
|---|---|---|---|
| 当前 RLT Stage2 | 直接输出完整执行 chunk `C×14`，actor 输入 `ref_chunk + z_rl + proprio`；不是 `ref + delta` | executed chunk；critic 状态 `z_rl + proprio`，不把 ref 放进 critic 状态 | Stage2 冻结 VLA / token 模块；BC 正则训练小 actor |
| 旧已成功 DSRL port | 32D Gaussian latent noise，沿预测 H 重复；VLA denoise 后才得到真实动作 | 32D latent noise；不是 decoded / env action chunk | VLA 冻结；Q 梯度训练 noise actor，不穿 VLA |
| 原版 EXPO-FT | tanh Gaussian bounded delta `C×D_env`；条件包含视觉 / proprio / action chunk；执行 `base + scaled_delta` 或原 base 中的 Q 选优 | 真正执行的 C 步动作，原 / edited 候选处在同一 normalized action 空间 | base 另走监督 flow-matching loss；Q 梯度只穿 edit / Q 路径 |

因此 RLT 的直接动作 head 较接近 EXPO 的 chunk 维度，但算法不同。RLT token encoder 输出 `z_rl`，Stage2 actor 输出动作；不要说“RLT 小策略输出 RL token”。DSRL SAC plumbing 较接近 edit 的 entropy / target-Q，但其 latent 存储与 Bellman 目标不能直接挪用。

## 4. 按模块复用与必须改的接缝

| 接缝 | 已有文件 / 精确位置 | 可复用 | EXPO-FT 必须补 |
|---|---|---|---|
| π0.5 base 采样 | `worktrees/pi05-stage/rlinf/models/embodiment/openpi_rlinf/eval_action_model.py:289`、`:358`、`:406`；`pi0_model/pi0.py:389`、`:452` | 已有 Observation / transforms、Euler ODE、explicit noise、prefix KV cache | `B×N` 独立 noise / candidate 维度；结果保持 `[B,N,H,32]`，提取同一 C / 14D 空间；rollout 与 next-state backup 共用同一候选器 |
| Base FM 训练 | `worktrees/pi05-stage/rlinf/models/embodiment/openpi_rlinf/sft_action_model.py:68`；`pi0_model/pi0.py:318` | normalized + padded `[B,H,32]` 的原生 FM loss、已有模型权重格式 | success / demo online dataset、指定 trainable filter / optimizer / base target-state、base 权重同步；不能只调用 SFT 入口而不给真实执行序列 |
| 小 actor / Q | 现场`rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py:124`、`:142`、`:165`；历史DSRL `openpi/openpi_action_model.py:1499`、`:1574` | 小组件初始化、twin/ensemble-Q 工具、off-policy API 形式 | 新 `ExpoEditPolicy`，显式 `base + delta`、TanhNormal 和 scale logprob；独立可训练视觉 encoder 与 editor 共享、动作 chunk Q，不继承 RLT BC/Q 课程或 DSRL noise Q |
| Rollout / replay schema | `worktrees/pi05-stage/rlinf/data/schema/embodied_trajectory_builder.py:39`、`:73`、`:153`、`:180` | `actions` / `curr_obs` / `next_obs` / `forward_inputs` / reward / termination / truncation | 保留 π0.5 所需 RGB、state、prompt / token / image mask；实际执行动作和 full-H 的相邻执行序列，success-episode 标记及有效 mask；RLT compact feature-only 数据不足以训练 base |
| Chunk TD | 现场`rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py:125`、`:540`、`:915` | 物理步折扣 reward sum、auto-reset / truncation 分界经验 | target candidate argmax，选完再 target-Q backup；REDQ10 / random2-min 按官方合同；每行真实执行长度 / 有效性与 time-limit 规则明确 |
| Replay 容量 / 恢复 | `.tmp/dsrl_current_impl_20260823/rlinf/data/storage/replay/dsrl_transition.py:222`；`fsdp_sac_policy_worker.py:966` | CPU ring 容器、RNG / cursor / counter、critic FP32 EMA、strict sidecar 形式 | EXPO 新 schema / method fingerprint；base+edit+Q+temperature+optimizers+target params+success marks+candidate RNG+update cadence 状态，而非 DSRL phase 复用 |
| 环境 / 评估 / 管理 | 当前 RoboTwin worker / exact namespace / fixed evaluation 配置 | success 判定、200动作上限、canonical动作与一次 decode、固定 seeds、日志与归还机制 | 新方法入口登记及资源预算；这不需要改现役 RLT / Dojo 或共享 Ray |

这里的“可复用”表示已有行为和接口可参考，不表示把整个旧文件覆盖到新 source。

## 5. 候选生成、选择与数据字段

原版官方 `expo_ft/agents/alg/expo_ft.py:521–600` rollout：生成 N 个 base，对其中 M 个作 edit，将候选 Q 打分后 hard argmax；backup `:601–693` 也生成 next-state 多候选。`:775–805` 用选出的 next chunk 构造 TD。

最小数据合同建议：

- 环境动作：真正执行 / 干预覆盖后的 canonical `[C,14]`，不能把 edit 本身或未选择的 base 填入 TD 的 `actions`。
- 训练动作：对应 normalization / transforms 处理后的 `[C,14]`，以及成功轨迹中基于**后续真实执行动作**构造的 `[H,14]` FM target。未执行的 base H 尾部不能伪作已完成控制。
- 观测：RGB / proprio / prompt（或 tokens）、next 观测、termination / truncation / executed-step-valid、episode ID / success 标记、base model version。
- 轻量诊断：选择类型、index、候选数、Q统计、delta norm、base / Q version、seed / noise RNG；这是结果解释及恢复所需。整组未选候选可选存储，**不要求给每个未执行候选虚构 reward / next-state**。

官方 edit actor update `expo_ft.py:694–733` 条件动作是 replay 的 `batch["actions"]`，即已执行 chunk；再采样 delta 并用 `Q(obs,a+delta)`。它并不要求每次 update 重新调用 base，也不要求 replay 保存最初未编辑的 base / residual 成分。若为了追溯额外保存这些字段，应把它们与 TD 真正执行 action 分清。

官方 `N>1` 才进入 selection 分支；源码在 `M!=N` 时 rollout / backup 候选拼接需要特别核对，原配方 `M=N=8` 对齐。移植时先做明确 shape 合同，不自行声称所有 N/M 组合已得到发布源支持。

## 6. Base 更新为何不能直接用 RLT replay

官方 `expo_ft/data/batch_processor.py:81–87` 在完整 episode 结束后标 success；`:112–145` 从成功在线 episodes / 演示采样 base batch。`agents/alg/expo_ft.py:922–944` 每次 update call 在 UTD critic 完成后更新 base，随后 edit / temperature；base loss 来自 `agents/vla/pi05.py:162–173` 的 FM。成功标签覆盖整个成功episode，不能只筛选`reward=1`末步。

现役 RLT Stage2主要消费 frozen `z_rl + proprio + ref_chunk`，不更新 VLA；现场`rlinf/algorithms/rlt/transition.py:22–33`确认`core_rlt_obs`只保留这三项，其 compact feature replay 不能提供可训练视觉 / prompt / full-H action target。RLT Stage1 clean50 dataset 只能提供静态演示起点；无法替代 EXPO 在线成功数据、新 model version 或成功标记恢复。

采用现成 frozen RLT token 作 edit/Q 输入，可以省视觉网络，但会变为 **EXPO 的 RLT-feature variant**：作者原配方为独立可训练 ResNet-50 供 critic / edit 共享。这项方法差异要单列，不把它当忠实实现。若base FM允许改变prefix encoder，则已存的z会过时，需原图重算或严格锁住prefix权重，不能只因token模块frozen就认定feature恒定。base 全冻结同样只能作为消融，不能省去 FM 更新而称完整 EXPO-FT。

## 7. 三项最容易静默改错的控制合同

### 动作归一化

当前 π0.5 预测模型维32，RoboTwin双臂 canonical执行维14；官方真机通常7D velocity。EXPO 残差应加在 **base/Q/replay一致的模型归一化14D有效子空间**，只做一次 output-transform到canonical env-space；14→32零padding只在base FM接口。官方 `pi05.py:461–480` 明确 Normalize / Unnormalize 配对。

现场已有`openpi_rlinf/eval_action_model.py:383–403`接口中的ref_chunk经output_transform成为env-space；Stage2 head实配直接输出执行chunk、identity保留幅度。**不能直接把RLT ref_chunk作为EXPO normalized候选**。应在base sampler的model_actions处新增明确seam，再按同norm stats把真实executed action回映到训练/Q空间。本次五源码未包含model factory /实际rollout入口的完整调用链；`openpi_rlinf`是确认存在的可复用接口，生产feature wrapper的最终model_type分派仍以同HEAD实配与factory定位为准，不由“文件存在”反推。

`beta=.05` 是normalized action单位，不是“.05弧度”或“所有14维各加.05真实动作”。需沿当前 norm stats / quantile / delta-vs-absolute transform，并分别核对关节与gripper；canonical identity student不意味着 edit residual也要无界。base可能超出[-1,1]，不要无证据对完整combined动作增加整体tanh；edit的tanh仅限制delta。

官方pick另外启用`edit_action_xyzg=True`，把DROID Cartesian旋转edit的3/4/5维置零（official `expo_ft.py:468–477`），只改xyz/gripper。双臂14D关节空间需单列active edit维度；不能复制该mask下标，不应把这一任务专用限制当EXPO普适结构。

### Chunk 折扣及停止

官方 EXPO `expo_ft.py:804` 是 `sum_i gamma^i r_i + gamma^C * mask * next_Q`；`replay_buffer.py:515–532` 维护termination / timeout / fabricated-tail有效性，默认窗口到timeout会valid=0，terminal-tail规则有专用开关。

我们的旧DSRL macro projection `dsrl_transition.py:156–180` 则人工改成成功0、否则-1，加gamma^C和success continuation。**不能直接复用为EXPO奖励**。当前 RLT worker能提供物理步折扣和terminal/truncation经验，但要审查chunk内提前成功后是否padded、多环境auto-reset是否跨episode，不能因shape是C步就一律bootstrap `gamma^C`。

建议首版锁定实际C10与200动作control；若照论文C8，属于新的方法外预算/控制改动，先明确。terminal reward后continuation=0；time-limit按既有control合同明确保留真实final obs，严禁reset obs作为未终止下一状态。若选择variable executed length，用 `gamma^K` 与K步reward一致，并注明和官方fixed-C有效窗口做法的差异。

### 更新与通信

“每轮采样8”是我们8个并行环境/回合；“EXPO8候选”是每个观测生成8份base，计算规模相乘。候选数会扩大denoise / backup显存和训练吞吐成本，不能只按小MLP费用估计。

官方每update call **20 critic + 1 base FM + 1 edit + 1 temperature**；不是所有网络各20次。当前RLT U5 / critic:actor2:1 / BC-Q权重课程不能原样标成EXPO。完整base在线更新还需把π0.5权重同步到rollout，base checkpoint及维护的target-state与Q/editor同步版本记录；现役RLT主要同步小actor，成本不同。官方original source维护`target_actor_params` EMA，但`:629`的backup sampler传入的是当前`actor_train_state`，不能未经讨论把backup改成EMA base。

## 8. 后续最小实施 packet（尚未执行）

1. 锁定 live π0.5 source SHA、norm stats / action config / fixed seeds、H/C/200动作、model权重与资源；新方法与control采用同一身份。
2. 独立 `expo_chunk` schema / storage：canonical执行动作、normalized chunks、full-H成功执行序列、图像/指令、success/valid masks及strict resume。
3. 一个统一 candidate sampler供rollout和backup；N8+M8为候选配方，不擅改parallel env数；prefix cache只是优化，按shape/RNG核对等价。
4. 新edit/Q及learner，不改RLT/DSRL原入口；明确normalized delta scale、ensemble-Q、entropy scale/temperature、critic/base/editor更新次数。
5. 少量高信息检查：N/M/B shape及independentnoise、argmax候选回放一致、一次transform/14→32、terminal/truncation/episode边界、FM成功数据和padding、Q梯度不入VLA、base FM梯度入允许参数、checkpoint跨phase/计数/权重恢复。
6. 服务器原生short fresh+resume闭环验收后再给正式配置和估时；运行前列实际命令、输出、GPU/RAM/磁盘、停止条件。当前继续既有实验，本文不占GPU。

## 9. 现场源码指纹（2026-10-01）

| SZ1同HEAD相对路径 | SHA256 | 已确认边界 |
|---|---|---|
| `rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py` | `3839fc30fcf93752fc196069a103f01affa8c2ef8aa11ab5977439d31f9a252c` | 79–81支持identity；124/136 actor/critic输入；142–170直接chunk/critic action |
| `rlinf/models/embodiment/openpi_rlinf/eval_action_model.py` | `8e3847ca68f904dbc328ec53e89306b26c54adb556f7aab255efd317b61a4e2e` | 358–403 frozenfeature+env-space ref；406–439 prefix-cache采样 |
| `rlinf/models/embodiment/openpi_rlinf/sft_action_model.py` | `6004f1b186d688153856ba2930f500dad1c0a5d7c586a5214758217e05a88b6f` | 68–95 FM / RLT联合开关；142–157 normalized+pad输入合同 |
| `rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py` | `c432faabaae1aede6829c0583dac27fe918801505e3fe2660a6a963f38ab72bb` | 540–585 discountedchunk TD；611–703 BC/Q课程/无alpha；915–937truncation；1042继承SACplumbing |
| `rlinf/algorithms/rlt/transition.py` | `c85a419aed20b6cf7f13f514b0d02825f45d05ef3daf8dcf088164bf6821bf8f` | 22–33 core三字段；53–75forwardinputs读取 |

请求的旧名`rlinf/workers/rollout/embodied_worker.py`在该HEAD不存在；未把缺文件当作没有rollout能力。本轮不根据旧文件名做移植packet，实际rollout入口需在实施前从同HEAD定位。服务器源码/实配/HEAD只读；本文新增1件，算法/运行修改0。
