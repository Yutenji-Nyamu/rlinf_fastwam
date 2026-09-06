# π0.5 BC：种子、学习曲线、耗时与工程上下文复核

日期：2026-09-06。训练/资源读取10:17:57—10:18:14 CST，种子调用链追加核对至10:22。只读服务器；本轮只新增本地证据与数据图、更新入口。未改运行参数/代码、未补跑评估、未停任务、未删除、未push，也未改变共享Ray或其他用户进程。

## 1. 图应该补什么；现在学得怎样

用户提供的图是**训练后每5轮一次的固定初态评估**，不是训练前成功率。原图缺少每轮训练采集曲线；本次已加逐轮细线、最近10轮均值粗线。首轮采集确实发生在第一次更新前，但使用训练种子与采集协议，不能标成固定评估Step0。本轮没有补跑SFT固定32评估，也不借别的smoke首次评估冒充Step0。

[BC采集＋固定评估PNG](pi05-seeds-discussion-20260906/bc_comparison.png) · [三实验完整PNG](pi05-seeds-discussion-20260906/overview.png) · [离线交互图](pi05-seeds-discussion-20260906/index.html) · [全部标量CSV](pi05-seeds-discussion-20260906/metrics.csv)。原10:01快照和图片未覆盖。

| 10:18现场 | 完整轮次 | 首轮采集（首次更新前） | 最近10轮采集均值 | 最新固定评估 | 最新checkpoint |
|---|---:|---:|---:|---:|---:|
| GPU6 π0.5 BC | 45/100 | 16/32 | 68.44% | Step45：20/32 | 40 |
| GPU7 π0.5 BC＋DVAC | 43/100 | 9/32 | 58.44% | Step40：19/32 | 40 |
| GPU4/5 π0.5 GRPO | 156/200 | 83/256 | 66.95% | Step155：21/32 | 150 |

三项wrapper/worker均在，所查fatal/OOM等错误0，标量无非有限值；checkpoint必要文件现场存在，不等于本轮做过恢复测试。GRPO每轮256条、两项BC每轮32条，不能拿相同步数直接比较样本效率。不同完成进度的“最近10轮”窗口不同：同为Step31—40时，BC采集65.31%、DVAC57.81%，仍有差距。

两项BC采集均有上升：前10轮BC55.94%、DVAC46.25%。但“DVAC一直更差”不适用于固定评估：Step15/20打平，Step25为20对16、Step35为18对15，DVAC反而较高。当前不能证明DVAC稳定占优，也不能凭初始差距认定DVAC损失有害。

## 2. 每轮32条的种子，到底固定了什么

实际读取GPU6/7独立worktree、resolved config和相同训练种子文件，再用部署的分配函数在服务器CPU计算顺序（没有创建仿真环境）。

| 随机性层次 | 两项BC当前实际情况 | 能说明什么 |
|---|---|---|
| 环境基础seed | `env.train.seed=0` | 相同 |
| 环境rank/stage | 两项都rank0、stage0，offset0 | GPU6/7编号不参与这里的seed偏移 |
| 训练种子表 | 同一内容，1000个候选，整批截成992个 | 同源、同顺序，不是两份任意随机表 |
| 每轮取样 | 32个一批，游标顺序前移，31批后回绕 | 不同轮换初态；两个run同轮计划取同一批 |
| `use_fixed_reset_state_ids` | train=false | 允许前移，不等于不设seed |
| 固定评估 | eval seed0，同32个种子，8并行×4批 | 每次覆盖相同32个初态ID |
| 学习器seed | `actor.seed=1234` | 不会自动初始化另一个Ray rollout进程的随机状态 |
| rollout动作噪声 | 当前调用链未见显式设种子/独立Generator | **初态相同不保证生成相同动作** |

训练JSON SHA256两项均`55b25866308d4e7da486282982ead4d5857b53fd977f659b216045f1950b45e2`；固定评估JSON均`f0059e543cdcb23b6813d4532d0ecf05954c3c1fdf75a109b2de7b31dace816e`。第一批前8个ID均为`131330,131321,125102,137553,112661,118910,125202,143750`；完整列表和后续批次在原始证据中。

源码定位（相对于各自部署的RLinf worktree）：

- `rlinf/envs/robotwin/seed_utils.py:18`：局部CPU Generator按base_seed打乱一次；:37整批截断。
- `rlinf/envs/robotwin/robotwin_env.py:42`：基础seed与逻辑offset；:417读种子表；:463按游标更新。
- `rlinf/workers/env/env_worker.py:817`：`finish_rollout()`训练轮结束后前移初态；评估批次同样显式前移，4批后回到起点。

边界：这是对配置与部署代码的种子计划复核，不是注入运行中的EnvWorker记录每个实际场景哈希。RoboTwin遇`UnStableError`可能重试其他seed，不能把名义ID顺序等同于已经逐episode验证了像素/状态完全相同。

## 3. 更关键的缺口：动作随机数，以及它为何可能持续影响BC

已核对完整的rollout worker、父Worker、模型构建入口、OpenPI模型与原生噪声函数，并搜索workers/scheduler/hybrid_engines的设种子调用。

1. `rlinf/workers/rollout/hf/huggingface_worker.py:141`的`init_worker()`构建并加载模型，未设rollout RNG；父Worker和模型构建入口也未发现补设。
2. 同文件`:494`在online_bc模式使用原生eval/ODE采样；`:528`调用`predict_action_batch`，没有传固定noise。
3. BC `openpi_action_model.py:912` → `sample_actions()`，`:992`在noise缺省时调用`sample_noise()`。DVAC对应路径也是这样。
4. 安装的OpenPI `models_pytorch/pi0_pytorch.py:172`使用`torch.normal(...,device=device)`，无显式Generator，依赖**该进程的默认Torch随机状态**。
5. DVAC新增的末端估计、方差计算只是同次forward的张量计算，检查到的新增代码不额外抽随机数。两边ODE路径内部还会调用通用噪声函数，不能只控制最外面一处后就不检查随机流是否一致。

这说明先前只记`actor.seed=1234`不足以宣称全训练pipeline已配对控制。它是当前可明确指出的复现性缺口；但本轮没有强制相同噪声重放两模型，所以**还不能证明初始7条或后续全部差距唯一由它造成**。更不能把“GPU编号不一样”直接当成环境seed不一样。

首轮还没做DVAC加权更新，因此不能解释成“DVAC训练先把策略变差”。成功BC存在自然的反馈：初始动作不同 → 成功episode不同 → 累计成功池不同 → 后续监督目标不同。这是差距可能延续的机制推断，不是本轮新验证的因果结论。

还核对了RoboTwin的初始化：线程池reset有全局锁覆盖scene setup，不能随口归因于并行reset互相改seed；指令生成使用按seed建立的局部`random.Random`，当前每次只生成1个候选，也不能把对单元素列表的选择当成已证实的语言随机差异。原生/物理计算仍不承诺跨硬件、版本完全位级确定性。[PyTorch复现说明](https://docs.pytorch.org/docs/2.9/notes/randomness.html)

### 建议：固定随机协议，不要永久固定同32个训练初态

- **环境安排保留现在的方案**：两项同一可复现顺序，每轮换32个初态；不用把100轮都限制在同32个场景。
- 下一次干净配对实验，明确给rollout设seed，按**逻辑rank**而非物理GPU编号派生；训练与评估各用独立随机流，避免评估消耗改变下一轮训练噪声。
- 评估用固定初态ID和固定噪声安排。若要求严格逐episode配对，再按初态ID/查询序号生成噪声；不是每次查询都把全局seed重置成同一个值。
- 模型原生FM训练中的噪声与时间采样继续保留，只控制可复现序列，不删除随机学习目标。
- 不在这两条进行中的实验中途偷改。当前数据保留为探索结果；若授权修正，下一次从原SFT同步起步，再加真正的Step0固定评估。

## 4. BC每轮时间花在哪里

来自当前TB计时，BC取Step36—45、DVAC取34—43；评估列是本run已有评估事件的均值，不把没有评估的轮当成5分钟。普通轮列剔除5的倍数。

| 阶段 | BC | BC＋DVAC | 解释 |
|---|---:|---:|---|
| 采集/环境与推理等待 | 326.8秒 | 319.0秒 | 32条尝试，不是32条成功 |
| 监督更新 | 446.7秒 | 449.4秒 | micro32、global1024、U10 |
| 权重同步 | 5.0秒 | 4.5秒 | 独立于上述训练时间 |
| 普通轮总计 | 12.95分钟 | 12.86分钟 | 无评估/保存的近期均值 |
| 固定评估额外开销 | 324.8秒 | 318.0秒 | 每5轮一次，8×4 |
| 含评估/保存摊销的每轮均值 | 14.10分钟 | 13.99分钟 | 上述最近10个完整轮 |

单卡每次Adam要累积`1024/32=32`个micro-batch，U10即320个micro-batch、**10240次chunk抽样呈现/轮**，不是只过一遍最新32条的数据。当前大约57%普通轮时间在监督更新、42%在采集。DVAC新增开销没有在本轮均值中表现为明显变慢，但这不是严格配对benchmark。

每10轮保存，不是每轮：两项Step40整代分别20.76/20.53GiB。其中π0.5的完整权重＋本地训练分片固定约18.91GiB，另外有累计池等，池长大会增加保存量。保存轮`time/step`减采集/训练/同步/评估的差额约25—29秒；它包含保存和其他余项，不伪装成独立的精确save计时。

当前BC累计938条成功episode/3236个chunk记录，DVAC为796/2698（轮次不同）。即使池完全不变，FM每次重采噪声/t也不是重复一个固定标量loss；仍可能过拟合，不能仅由loss下降证明策略改善。

## 5. 本地上下文：材料多不等于全部被加载

本轮修改前按UTF-8文件字节实测：

| 当前要求完整读的文件 | 大小 |
|---|---:|
| AGENTS.md | 2.37KiB |
| PROJECT_CONTEXT.md | 7.32KiB |
| HANDOFF.md | 48.01KiB |
| 09-03窗口交接 | 6.86KiB |
| 本题π0.5 BC单一SSOT | 27.76KiB |
| 前四份合计／加本题SSOT | 64.55／92.31KiB |

本轮`rg`可见docs中439份Markdown（不含随后新增本文）；大块包括DVAC telemetry110、深圳PPO/RLT68、在线BC43、Fast-WAM41、RLT/DSRL迁移33，另有GRPO、π0.5和管理记录。代码草稿、worktrees、references、脚本、证据JSON/图也在磁盘，但不意味着每轮自动读进模型。

OpenAI的AGENTS说明是按全局/项目目录链加载指令文件；不是递归把整个项目库灌入上下文。其32KiB默认限制针对项目指令发现/合并，不是“我们手动读HANDOFF只能读32KiB”的限制。当前真正的额外负担，是AGENTS要求完整阅读的长交接和长SSOT，以及本聊天自身累积的历史/工具输出；92.31KiB是文件字节，不是已测token数。[官方AGENTS规则](https://learn.chatgpt.com/docs/agent-configuration/agents-md)

建议的裁剪是**归档旧入口内容，不删研究材料**：

1. 根HANDOFF缩回约3—6KiB，只留当前三项任务、授权边界、风险、下一步与专题路由；历史段落整体无损归档。
2. π0.5 BC SSOT保留模型/方法合同、参数来源、当前未决问题；逐次smoke/fatal/启动/刷新留证据文件，不在根目录重复。
3. 09-03窗口交接改成恢复旧窗口时按需读，而不是所有新任务必读；这属于阅读规则变化，应单独确认后执行。
4. 保留AGENTS安全规则；不为了短而去掉现场刷新、授权和他人边界。不要删除源码worktree或Git历史来“省上下文”。

本轮只讨论该结构，未执行大规模归档/裁剪。

## 6. 自己的/data存储与Git覆盖

详细数值和精确清理候选统一放在[管理复核§1—§3](../../server-admin/SZ_STORAGE_GIT_REVIEW_20260906.md)，不复制另一套清理计划。

要点：本人/data约1695.39GiB，results1527.05GiB占90.1%；最保守新增候选是π0 v8与π0.5 smoke的Step1四个大文件，共36.10GiB，分别保留Step2。DSRL正式较早三代的12个大文件90.35GiB是后续可讨论项；不能据本次目录检查声称已验证删除不影响任何恢复依赖。当前运行、最佳历史点、模型起点、共享Ray先不动。

代码方面：本人RLinf的22个codex分支逐项比对远端HEAD一致；26个RLinf/RoboTwin linked worktree已经读状态。旧GRPO/PPO/DVAC/Prism、Fast-WAM、DSRL/RLT有轻量证据归档，BC有smoke/正式启动记录，Sidney已有100步总结ZIP；不等于所有原始日志/当前尾段/本地文档均已备份。具体未提交修改、网络未核实和尚未封存项见管理复核§3。本轮没有push。

## 7. 整机现场与风险

GPU4/5当前67.07/48.93GiB，GPU6/7为64.32/72.89GiB；BC/DVAC从启动至今采样峰值均约73.73GiB（80GB卡实际79.65GiB）。GPU1/2/3空闲；GPU0其他用户模型服务约9.6GiB，本轮不深入查看或干预。

128逻辑CPU，load约6.5；RAM总2015.51GiB、available669.67GiB。Swap约5.97GiB基本用满，但短采样si/so与当前CPU/内存/IO压力均0，不能仅凭swap used说正在抖动。四个环境进程RSS仍很大，BC/DVAC环境RSS约246.43/193.19GiB；共享映射使RSS不能直接相加，且本次没有定位私有内存增长的对象。继续运行不等于容量风险消失。

/data可用630.06GiB、/home1246.15GiB，inode不紧。首要后续工程事项是**rollout随机协议缺口和环境内存增长**；二者不要混成一个解释，也不要未经定位再归因OIDN。

## 8. 原始证据与本轮执行记录

- 现场标量/日志/进程/资源/ckpt：[PI05_DISCUSSION_LIVE_20260906.json](PI05_DISCUSSION_LIVE_20260906.json)。
- resolved/wrapper/环境与BC代码：[PI05_SEEDS_SOURCE_20260906.json](PI05_SEEDS_SOURCE_20260906.json)。
- 实际HF worker/模型/原生noise、种子顺序：[PI05_SEED_CHAIN_VERIFIED_20260906.json](PI05_SEED_CHAIN_VERIFIED_20260906.json)。源码均记录路径、文本、SHA。
- 全量本人存储与Git只读结果：[SZ_STORAGE_GIT_READONLY_20260906.json](../../server-admin/SZ_STORAGE_GIT_READONLY_20260906.json)。
- 并行只读脚本：`local_scripts/pi05_discussion_readonly_20260906.py`；远端命令位于同目录`remote_commands/sz_pi05_seeds_readonly_20260906.sh`、`sz_own_storage_git_readonly_20260906.sh`、`sz_pi05_seed_chain_followup_20260906.sh`。
- 最初补充source读遭RoboTwin远端HTTP/1.1查询45秒超时；失败输出保留在`PI05_SEED_CHAIN_FOLLOWUP_20260906.json.stderr.txt`，空JSON不是成功证据。移除该网络读后，源代码/种子读正常完成于`VERIFIED`文件。没有切换认证路线、修改Git配置或无限重试。
- 本地数据图渲染命令：`python local_scripts/render_pi05_three_runs_20260906.py --input PI05_DISCUSSION_LIVE_20260906.json --output pi05-seeds-discussion-20260906`。仅解析已落地数据；PNG实看确认标签/对照关系，HTML保留悬停值和CSV入口。
