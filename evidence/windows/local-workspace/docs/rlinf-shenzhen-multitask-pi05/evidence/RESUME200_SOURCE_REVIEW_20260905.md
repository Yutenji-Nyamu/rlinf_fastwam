# Sidney pi0.5：formal100 接续到 global step 200 的恢复语义

日期：2026-09-05。范围：只读源码/配置讨论；未停止、启动、恢复加载或测试任何服务器任务。

## 1. 结论与当前证据边界

建议让当前 formal100 自然完成，在 Step100 的完整训练 checkpoint 保存结束后，再按用户批准的新合同接续至 global step 200。若当前仍健康，提前中断并回到 Step90 会丢弃 Step90 后未进入 checkpoint 的进度；它也不会比从 Step100 接续保留更多训练状态。

本次已用 **2026-09-05 11:23:45—11:23:49 CST服务器返回的Sidney部署源码及resolved** 完成核对。工作树为 `/data/chenyiteng/projects/rlinf-shenzhen/worktrees/sidney-pi05-current-rlinf`，现场HEAD `f50e235c5ab1f4390f0ba92bfb13390ed0a86810`、dirty为空。七个关键文件的原始源码文本SHA256均重算吻合；本地最初审阅副本与部署源内容也一致。**以下是部署源码确认的恢复语义；本次没有做真实恢复加载或训练测试。**

原始证据：[服务器只读快照](../../fastwam-robotwin-rlinf-grpo/evidence/discussion-audit-20260905/live_snapshot.json)、[实际resolved](../../fastwam-robotwin-rlinf-grpo/evidence/discussion-audit-20260905/sidney_resolved.yaml)。源码定位和SHA见第6节。

## 2. 从 checkpoint 恢复的内容

| 状态 | 源码行为 | 对本次接续的含义 |
|---|---|---|
| 模型参数 | local-shard 中 `model` 经 `model.load_state_dict` 载入 | 沿用 Step90/100 的训练后策略 |
| Adam 状态 | 保存/载入 `optimizers` state dict | 包括 Adam 的动量、二阶矩及内部 step；不重新用一个空 optimizer 起跑 |
| LR scheduler | 保存 `lr_schedulers` state dict，恢复时调用 `load_state_dict` | 恢复 scheduler 计数与状态；不是从第0轮重新预热 |
| actor RNG | 保存/载入 Python `random`、NumPy、Torch CPU 与当前加速卡 RNG | 保存 actor 进程的这些随机状态 |
| runner global step | 从 `runner.resume_dir` 的 `global_step_N` 目录名解析 N | `max_steps=200`、N=100 时运行101—200；N=90时运行91—200 |
| rollout / env 状态 | runner先重新初始化 rollout/env，再只调用 actor checkpoint load | 不恢复原环境场景、episode游标及 rollout worker 的运行中随机状态，不能声称逐轨迹、逐比特等价于一次不间断200步 |
| manager 的 `optimizer_steps` Python计数 | 构造时为0，未进入上述 checkpoint payload | 与Adam内部step不同；当前 `critic_warmup_steps=0`，不会由此重新触发critic warmup |
| grad scaler | 上述payload没有保存scaler状态 | 当前resolved为 `enabled=false`，本次没有启用的scaler动态状态需要延续 |

恢复链为 `EmbodiedRunner.init_workers` → `actor.load_checkpoint(<global_step_N>/actor)` → `FSDPModelManager.load_checkpoint` → `FSDPStrategy.load_checkpoint` → `Checkpoint.load_state_dict`。首个接续轮仍走正常 actor→rollout 权重同步，再重新采样。

恢复入口必须指向 **`.../checkpoints/global_step_100`**（或90），而不是 `full_weights.pt`。后者只有模型权重，不是完整训练状态入口。当前两rank local-shard需维持相同FSDP版本、模型和两rank切分；本提议不改变GPU数量或切分。

## 3. max_steps 100→200 与学习率

09-05 11:23 实际resolved再次确认：

- `runner.max_epochs=1000`、`runner.max_steps=100`；runner有效上限取两者较小者。
- `actor.optim.lr=5e-6`；没有显式 `lr_scheduler`、`lr_warmup_steps` 或 `total_training_steps` 叶。
- 当前builder默认 `lr_scheduler=constant`；warmup ratio默认0，因此warmup为0；`total_training_steps`默认0。
- constant scheduler过warmup后乘数恒为1；该分支不依赖 `runner.max_steps`，也不依赖所谓200步衰减总长。
- embodied actor在每个outer step训练结束后调用一次 `lr_scheduler.step()`；每轮两个optimizer call与scheduler一步不是同一计数。

因此，按已经核对的部署源码，并保持优化配置逐叶一致，**只把runner上限改为200不会把学习率曲线拉伸或重启；接续仍为5e-6**。不要为了“翻倍训练”额外添加cosine、warmup或新的scheduler总步数。

`max_steps`在runner构造时读取；当前代码没有热加载配置机制。只修改磁盘YAML不会让正在运行的formal100自动延长。自然结束后，另一个明确命名的resume run载入Step100，是这里讨论的接续方式。

## 4. “配置不变”的准确表述与预算

准确表述是：**模型、任务、环境、每轮采样、GRPO、优化、资源切分与评估/保存频率全部沿用原formal100；只延长总预算，填写恢复目录并隔离新run输出。** 完整配置不能字面称为每个叶子都不变。

接续合同应逐叶列出：

1. `runner.max_steps: 100 → 200`；`max_epochs=1000`无须改。
2. `runner.resume_dir: null → <原run>/.../checkpoints/global_step_100`。
3. 新run/experiment/log、video、train/eval data、disabled DVAC output等相关路径叶；不覆盖原formal100。
4. 其余resolved叶继承原run，包含64 train env、rollout4、每轮256 trajectories、G8、GB1024/MB32/update2、M10、noise0.5、200-action、GPU4/5两rank、fixed32/eval5/save10、local-shard。

| 接续起点和终点 | 新执行outer steps | 新train trajectories | 新optimizer calls | 新fixed eval episodes | 新checkpoint代数 |
|---|---:|---:|---:|---:|---:|
| Step100 → 200（建议） | 100 | 25,600 | 200 | 640 | 10 |
| Step90 → 200 | 110 | 28,160 | 220 | 704 | 11 |

Step100接续最多增加102,400个query records、5,120,000个指令action槽；这些不等于实际物理仿真插值步数。表格不包含重启初始化开销，也不把旧run Step90后被丢弃的计算重新算成有效接续进度。

## 5. 等100与中断回90

11:23:45—49现场快照为完整Step96、wrapper存活；Step90双rank各10,150,817,963 bytes，`full_weights.pt`为8,526,574,644 bytes；该次尚无Step100文件。这里只确认文件存在与大小，没有加载其内容；后续进度和预计结束时间以主agent本轮现场说明为准。

原formal100在100处正常触发fixed评估及保存。等待完成可以保留最新模型/optimizer状态与整条100步曲线；随后新增100步即得到清楚的总200步学习预算。中断回90相当于从较旧训练状态新走110步，当前90以后的结果只能保留为被分叉的旧证据，不能直接无标记拼接为一条严格连续的200步曲线。

若当前run出现实际失败，才按最后完整checkpoint重新决定起点。没有现场故障理由时，不建议为了延长训练主动中断一个接近自然终点的健康run。本轮只是讨论，不据此执行停止或续训；用户批准执行时仍需完整resolved、精确命令、新输出、预算、资源与停止条件。

## 6. 部署源码定位与本地审阅记录

以下源码路径均相对服务器工作树，也对应本地证据目录 `../../fastwam-robotwin-rlinf-grpo/evidence/discussion-audit-20260905/sources/sidney/`。

| 文件 | 关键行与作用 | 服务器源码SHA256 |
|---|---|---|
| [rlinf/runners/embodied_runner.py](../../fastwam-robotwin-rlinf-grpo/evidence/discussion-audit-20260905/sources/sidney/rlinf/runners/embodied_runner.py) | 163—185初始化并恢复actor/目录步数；308—327评估保存；482—545接续循环；655—660总步数上限 | `fd621df701923fed16f500ec4b4ce53ff95821d19a5ba7f2b70a0891fa5aef51` |
| [rlinf/workers/actor/embodied_fsdp_actor_worker.py](../../fastwam-robotwin-rlinf-grpo/evidence/discussion-audit-20260905/sources/sidney/rlinf/workers/actor/embodied_fsdp_actor_worker.py) | 748每个global batch的optimizer call；760每轮scheduler推进；926—991调用父级save/load及恢复模型version | `52df6fdac6c196462cefaa386ff1417cc5a94209b3df816902ee336a3dc1f3b1` |
| [rlinf/hybrid_engines/fsdp/fsdp_model_manager.py](../../fastwam-robotwin-rlinf-grpo/evidence/discussion-audit-20260905/sources/sidney/rlinf/hybrid_engines/fsdp/fsdp_model_manager.py) | 88 manager计数；350—400训练状态save/load；478—512 scheduler默认构造 | `9a9881f7f6160bfcc4624dec05986c6e7dc83b1081f9a0931ca5ac5e52896e5e` |
| [rlinf/hybrid_engines/fsdp/strategy/base.py](../../fastwam-robotwin-rlinf-grpo/evidence/discussion-audit-20260905/sources/sidney/rlinf/hybrid_engines/fsdp/strategy/base.py) | 184—267按rank保存、barrier与full权重；270—334按相同rank加载local shard | `13214654cb9da81d339211e4dd81b0e0cbf50e2d3906a1ee1ccb07915ef72098` |
| [rlinf/hybrid_engines/fsdp/strategy/checkpoint.py](../../fastwam-robotwin-rlinf-grpo/evidence/discussion-audit-20260905/sources/sidney/rlinf/hybrid_engines/fsdp/strategy/checkpoint.py) | 66—101 payload字段；104—132载入模型、optimizer、scheduler和RNG | `498052ef5761d55659c54198ebef99cbea7ece7b75085d309d2b3766bede770f` |
| [rlinf/hybrid_engines/fsdp/utils.py](../../fastwam-robotwin-rlinf-grpo/evidence/discussion-audit-20260905/sources/sidney/rlinf/hybrid_engines/fsdp/utils.py) | 537—545 constant/warmup LambdaLR，不依赖200步上限 | `ca38f9b0446a948e775fca26fe2f46bdc4f5b878de3a3209eff6da31af26a5ec` |
| [rlinf/utils/utils.py](../../fastwam-robotwin-rlinf-grpo/evidence/discussion-audit-20260905/sources/sidney/rlinf/utils/utils.py) | 665—701 Python、NumPy、Torch CPU/当前设备RNG保存与恢复 | `a7b17c215a48cb8a0608b1221b04ae2fa3ff7acf25f02b29006176037a38cc7b` |

- 完整读取 `00_INDEX_AND_EXECUTION.md`；读取当前专题formal100与学习审阅条目，锁定实际embodied actor而非generic actor。
- 从 `CURRENT_TRAINING_HEALTH_20260905.data.txt` 的首行JSON提取Sidney实际resolved；只作为带时间戳的配置/文件线索，不用其中训练状态代替本轮现场。
- 先审阅本地 `worktrees/pi05-stage` 对应七个文件，随后主agent统一只读取回Sidney实际源。对 `live_snapshot.json.sources` 中 `label=sidney` 的七个条目，用UTF-8原始文本重算SHA256，7/7与服务器记录一致；本地sources文件为CRLF，规范为LF后7/7与原始文本及SHA一致，排除换行格式差异。
- 候选副本与实际部署源码规范换行后逐字一致，所有恢复结论均已重新锁到本节部署源；读取11:23实际resolved确认LR、上限、local-shard、disabled scaler及warmup0相关叶没有变化。
- 只读提取本轮快照中的Step96及Step90三个文件清单；没有运行import、compile、测试或checkpoint load。仅新增、更新本篇独立证据文档，没有改根HANDOFF或任何服务器文件。

后续只在用户授权执行后验收实际恢复；本次源核对与文件清单不替代恢复加载成功证据。
