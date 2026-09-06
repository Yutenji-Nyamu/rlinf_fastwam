# 深圳 RLinf 扩展实验矩阵：π0.5 / PPO-DVAC / Fast-WAM / OGPO

PPO-DVAC 逐文件实现前审计见
[`01_PI0_PPO_DVAC_PRE_IMPLEMENTATION_AUDIT.md`](01_PI0_PPO_DVAC_PRE_IMPLEMENTATION_AUDIT.md)。
实施与 smoke 流水见
[`02_PI0_PPO_DVAC_IMPLEMENTATION_LEDGER_20260830.md`](02_PI0_PPO_DVAC_IMPLEMENTATION_LEDGER_20260830.md)。
π0.5 的独立当前计划见
[`../rlinf-shenzhen-pi05-robotwin/00_INDEX_AND_PLAN.md`](../rlinf-shenzhen-pi05-robotwin/00_INDEX_AND_PLAN.md)。
RoboTwin 多任务 π0/π0.5 checkpoint 候选、Fast-WAM env offload/OIDN 与跨模型 Flow-SDE 的
2026-09-02 联合只读调研见
[`02_ROBOTWIN_MULTITASK_PI_OFFLOAD_FLOWSDE_RESEARCH_20260902.md`](02_ROBOTWIN_MULTITASK_PI_OFFLOAD_FLOWSDE_RESEARCH_20260902.md)。

> 状态：2026-09-01 旧 π0 两卡 PPO pair 已收束：Control按授权停在完整Step60；PPO-DVAC在完整Step58后因Ray整机95%内存阈值退出，代码/配置审计未发现no-op或低级错误。轻量终态包为
> [`../../exports/shenzhen_pi0_ppo_control_dvac_w0p5to1p5_stopped_pair_light_evidence_20260901.zip`](../../exports/shenzhen_pi0_ppo_control_dvac_w0p5to1p5_stopped_pair_light_evidence_20260901.zip)，图与原始小型材料见
> [`evidence/ppo-control-dvac-stopped-20260901`](evidence/ppo-control-dvac-stopped-20260901)。
> 随后已在GPU4/5与6/7 fresh启动 π0.5 GRPO Control / GRPO-DVAC `[0.5,1.5]` formal-100；两边均进入Step1 rollout，严格配置与并发隔离见
> [`../rlinf-shenzhen-pi05-robotwin/evidence/FORMAL_PAIR_CUTOVER_AND_STARTUP_LEDGER_20260901.md`](../rlinf-shenzhen-pi05-robotwin/evidence/FORMAL_PAIR_CUTOVER_AND_STARTUP_LEDGER_20260901.md)。
> Fast-WAM current GRPO及Action-DVAC-Adv已完成实现与strict-resume smoke；formal尚未启动。OGPO尚未进入current实施。
> 本文是该扩展专题的唯一当前入口。已有各算法专题仍保存自己的实现细节与历史证据，本文只做跨专题索引、判断和实施顺序。

## 0. 先给结论

这不是一个应当一次铺开的笛卡尔积。最清楚的扩展方式是一次只扩一个轴：

1. **先扩算法，模型保持 π0**：两卡 PPO Control → 两卡 PPO + DVAC Action-Adv `[0.5,1.5]`。
2. **再扩模型，算法走官方主线**：两卡 π0.5 PPO。
3. **再接 Fast-WAM 到 current RLinf**：深圳官方 standalone 已经跑通，不必重新部署；首线做 critic-free GRPO，而不是先做 PPO。
4. **最后迁移 OGPO**：它是独立 actor/critic/replay/runner/checkpoint 栈，不是薄 YAML；工程量和风险均最大。

当前矩阵如下：

| 模型 / 算法 | PPO | GRPO | DVAC | OGPO |
|---|---|---|---|---|
| π0 | 深圳四卡已跑通；两卡 Control smoke 已闭环 | 深圳两卡已充分跑通 | 多个 GRPO-DVAC 变体已跑；PPO-DVAC `[0,2]` smoke 已闭环 | AutoDL 旧 RLinf 已实现；深圳 current 待迁移 |
| π0.5 | **RLinf 官方 RoboTwin 主线，下一条官方型模型基线** | RLinf 只在 LIBERO 有 π0.5 GRPO 先例，不是官方 RoboTwin recipe | 暂不先做 | 暂不做 |
| Fast-WAM | AutoDL 旧 RLinf 跑过但无改善证据 | AutoDL 旧 RLinf 跑过；深圳 current 待迁移 | standalone telemetry 已有；训练权重不是首线 | 暂不做 |

## 1. 参考材料索引及其作用

| 材料 | 里面有什么 | 对当前任务的作用 | 不能怎样使用 |
|---|---|---|---|
| [RLinf RoboTwin 官方训练文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html) | 官方支持的模型、算法、安装、模型下载、启动、发布结果 | 锁定 π0.5 在 RoboTwin 的正式入口是 PPO；锁定 OpenPI/RoboTwin 环境合同 | 不把官方 8–16 GPU 数值直接照搬为深圳两卡预算 |
| [RLinf π0.5 PPO 官方配置](https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml) | 8 卡 placement、256/128 env、GAE/value、update5、M5、LR | 两卡缩放的母配置 | 不用深圳 GRPO 的 G8/actor-only 字段覆盖它 |
| [深圳 π0 PPO/GRPO/RLT 主计划](../rlinf-shenzhen-pi0-ppo-rlt/00_INDEX_AND_IMPLEMENTATION_PLAN.md) | current source lock、四卡 PPO、两卡/四卡 GRPO、RoboTwin 资源与故障经验 | 复用每 rank simulator 数、FSDP、输出隔离、fixed eval 经验 | 不把 GRPO advantage/group 配置迁进 PPO |
| [深圳 Action-Adv 主计划](../rlinf-shenzhen-grpo-dvac-action-adv/00_INDEX_AND_PLAN.md) | DVAC `[B,H]`、action-level ratio/clip、修正后的 H 聚合与 sidecar | PPO-DVAC actor 挂点的 current 代码先例 | 不给 value target、return 或 critic loss加 DVAC 权重 |
| [Fast-WAM 官方仓库](https://github.com/yuantianyuan01/FastWAM) | 当前 standalone 训练/推理、RoboTwin release checkpoint、H32/C24/M10 | 当前 Fast-WAM oracle 与新接口的唯一上游依据 | 附件中的旧命令不能替代当前官方代码 |
| [Fast-WAM 深圳专题](../fastwam-robotwin-rlinf-grpo/00_INDEX.md) | standalone 49/64、四任务数据、旧 RLinf GRPO/PPO、迁移设计 | 证明深圳推理已通；解释旧实验协议和 current 接口差异 | 不把训练 rollout 上升称为固定评估增益 |
| 用户附件 `E:/0school/研二上/iclr27/fastwam.md` | AutoDL 部署流水、代理/依赖/CuRobo/SAPIEN 处理 | 遇到相同具体故障时作历史线索 | 它不是本轮授权，也不是深圳命令；其中账号信息不得复制或复用 |
| [OGPO 论文 v4](https://arxiv.org/html/2605.03065v4) / [官方代码](https://github.com/simchowitzlabpublic/OGPO_public) | OGPO+CA、10Q、EMA、whole-chain ratio、逐交互 UTD | 判断旧实现语义和“更新过多”假说 | 不用 RLinf GRPO/PPO 的 loss 名字替代 OGPO 数学 |
| [OGPO 专题](../rlinf-robotwin-pi0-ogpo/00_INDEX_AND_IMPLEMENTATION_PLAN.md) | 旧实现白名单、v1/v2 结果、replay/checkpoint 故障、调用链 | current port 的 source oracle | 旧 6.1k 行 patch 不能整块 cherry-pick 到 current |

## 2. π0.5：先跑官方 RoboTwin PPO

### 2.1 官方到底支持什么

截至 2026-08-30，RLinf current 的 RoboTwin × π0.5 正式入口是 `adjust_bottle + PPO`。官方一般性支持 π0.5 + GRPO，但公开例子在 LIBERO；所以以后做 RoboTwin π0.5 GRPO 可以称为派生实验，不能称为官方 RoboTwin 配方。

深圳锁定的 RLinf `7d07a421...` 与当前官方 main 在 π0.5 PPO 配置、π0.5 model preset、RoboTwin preset、eval YAML 和启动脚本上逐字节一致。因此**为了 π0.5 不需要先升级 RLinf**；继续 source-lock `7d07a421...` 更能保持与现有 π0 实验可比。

官方文档给出的 `adjust_bottle/demo_clean` oracle 是 π0.5 SFT `85.94%`、PPO `96.09%`。这说明任务仍有约 10 pp 的 PPO 提升空间，但不是深圳两卡结果的保证。

### 2.2 两卡应该怎么缩

| 字段 | 官方 8 卡 π0.5 PPO | 推荐深圳 2 卡 | 理由 |
|---|---:|---:|---|
| train env | 256 | 64 | 每 rank 都是32，保持 simulator 并发 |
| eval env | 128 | 32 | 每 rank 都是16，已避开两卡 fixed64 的 Vulkan 压力 |
| rollout epochs | 4 | 4 | 每 outer step 仍为256 trajectories |
| actor global / micro batch | 2048 / 32 | **512 / 32** | 保持官方每 rank effective batch、梯度累积和每轮 optimizer-step 数 |
| update epochs | **5** | **5** | π0.5 官方 recipe 的真实模型特定差异，不能静默改成 π0/GRPO 的2 |
| denoise steps | 5 | 5 | π0.5 官方模型合同 |
| actor/value LR | `5e-6 / 1e-4` | 不变 | 保持官方算法配方 |
| advantage/value | GAE + value head | 不变 | 这是 PPO，不是 GRPO |
| H/C/D | 50/50/14 | 不变 | checkpoint 与 RoboTwin action 合同 |
| eval/save cadence | 10/10 | **5/10** | fixed32/eval5与深圳两卡曲线同轴；只改变测量频率 |

这一缩放每步是 256 条 trajectory、最多 1,024 个 query records；`GB512/update5`保持官方每轮10次
optimizer step和约5,120次record presentation。它会比两卡 π0 GRPO 的 update2 更重，时间和显存不能只按模型参数或 π0 单步线性外推。此前草案中的`GB1024`会把每轮optimizer step减半，现已在π0.5专题中明确纠正。

### 2.3 环境能不能复用

可以复用的：

- current RLinf source lock、RoboTwin `RLinf_support`、assets、ALOHA、三相机、seed 文件；
- 现有 OpenPI venv原则上也能复用，因为官方 π0/π0.5 走同一 `--model openpi --env robotwin` 安装路径。

需要独立的：

- 新的 π0.5 工作树、run/output 路径；
- 完整 π0.5 SFT checkpoint（约 8.53 GB，包含三份权重、index、norm stats 和 assets），放 `/data/chenyiteng`；
- 第一次真实 checkpoint load + official fixed eval，用来确认既有 venv 对 π0.5 真实可用。

因此不是“新建一整套仿真环境”，也不应在正在训练的 π0 工作树或 venv 里安装包。首线先尝试只读复用现有 venv；只有真实 import/load 缺依赖时，才复制出 π0.5 专用 venv。

### 2.4 推荐闭环

`SFT fixed32 → one outer step → checkpoint`，成功后再讨论 formal 总步数。首线固定`eval5/save10`，官方 PPO checkpoint不先下载；需要作oracle时再取。

## 3. π0 两卡 PPO + DVAC Action-Adv `[0,2]`

### 3.1 先建立两卡 PPO Control

深圳四卡 PPO 已用 `128 train / 64 eval / 4 rollout epochs / B2048 / MB32 / update2` 跑到完整 Step47，Step10/20/30/40 fixed64 为 `58/62/58/62`，数值正常；终止点是 EnvWorker 主存增长，不是 PPO 数值失败。

两卡 Control 应按卡数只缩外层：

| 字段 | 四卡 PPO | 两卡 PPO Control |
|---|---:|---:|
| train/eval env | 128/64 | 64/32 |
| env per rank | 32/16 | 32/16 |
| rollout epochs | 4 | 4 |
| trajectories/step | 512 | 256 |
| max query records | 2048 | 1024 |
| global/micro batch | 2048/32 | 1024/32 |
| update epochs | 2 | 2 |
| GAE/value/LR/H/C | 原 PPO | 不变 |

这与两卡 GRPO 共用的是资源壳，不是算法字段。

### 3.2 PPO-DVAC 与 GRPO-DVAC 哪些相同、哪些不同

相同的只有 actor 侧机制：

$$A^{\mathrm{eff}}_{i,h}=A_i\,w^{\mathrm{DVAC}}_{i,h},\qquad w\in[0,2].$$

- 保留 action-level log-prob、ratio、clip；
- 使用已经修正的“先沿 H 求和，再在 query/batch 上平均”，保持与 Control 接近的更新尺度；
- DVAC history/sidecar、selected denoise window 和 telemetry 可复用 current Action-Adv 实现。

不同点是基础 advantage：

- GRPO-DVAC 的 $A_i$ 来自同 reset group 内的相对 reward；
- PPO-DVAC 的 $A_i$ 来自 GAE 和 learned value head。

因此 DVAC **只改 actor policy advantage/loss**；PPO 的 return、value target、value loss、GAE、critic optimizer 全部不改。也不引入 GRPO 的 G8、group filter 或 actor-only 配置。工程切口小，但必须经过一次 PPO shape 审计，确认 actor 用 `[B,H]`、critic 仍用 `[B,1]`。

### 3.3 实验顺序

先跑 clean 两卡 PPO Control，再从同一 resolved config 复制 PPO-DVAC `[0,2]`；两份配置除 method fields、placement、命名和输出路径外必须 leaf-by-leaf 相同。`[0,2]` 较强，但这是用户指定的第一格；如果后续结果不稳，再把 `[0.5,1.5]` 作为单独强度实验，不在首版里悄悄改。

## 4. Fast-WAM：不是重新部署，而是迁移到 current RLinf

### 4.1 深圳 standalone 已经到什么程度

深圳已锁官方 Fast-WAM `7faa711...` 和 HF release revision `8eaceeb...`，按官方 RoboTwin entry 跑通：

- 最小 `adjust_bottle`：`1/1` 成功；
- 四任务各16条：adjust/move/turn/pick=`16/11/10/12`，合计 `49/64`；
- 64条均自然退出，fatal=0，并保存视频与 telemetry。

所以“先去深圳跑通一个官方成功推理”已完成，不应重做。今后 current RLinf 集成验收只需拿这个 standalone 当 B=1 动作 oracle，而不是再安装一次官方环境。

### 4.2 AutoDL 旧实验到底表现怎样

| 实验 | 旧协议 / 结果 | 能说明什么 | 不能说明什么 |
|---|---|---|---|
| GRPO adjust_bottle | 训练成功率约97%，大量全成功 group | 任务过饱和、group 信息不足 | 不能据此判断 Fast-WAM RL 有效 |
| GRPO move_stapler | 到Step77；首10轮40.47%，末10轮51.25%，best67.19% | 训练 rollout 有上升信号 | 无固定种子 eval、未完成100步，不能称稳健提升 |
| PPO move_stapler | 到Step33；首10轮46.80%，末10轮45.86% | 没有观察到上升 | 不能用 EV NaN/value_clip0 断言 critic 坏；这些是日志统计缺陷 |

旧 GRPO 每轮只有 `64 trajectories / 512 unique transitions / 512 presentations`；旧成功 π0 GRPO 是 `256 / 1024 / 2048`。因此“Fast-WAM 每轮数据/呈现偏少”是 move_stapler GRPO 的合理协议弱点，但仍不是已证明唯一根因：

- adjust_bottle 的首要问题是饱和，而不是少数据；
- PPO 本来已有 `128 trajectories / 1024 transitions / 2048 presentations`，仍没改善，少数据不能单独解释 PPO。

### 4.3 为什么不能直接搬旧 adapter

旧集成基于 RLinf `6d0db56...` 和 Fast-WAM `45d8e1...`。当前深圳 official Fast-WAM `7faa711...` 已将旧 dict KV cache 接口改成 tensor cache；current RLinf 又有 typed `PolicyOutput → ChunkStepResult → Builder → Trajectory`、新的 actor/FSDP/checkpoint/registry。

因此可复用的是模型语义和少量 opt-in 接口，不是旧文件：

1. Fast-WAM policy adapter：H32、执行C24、M10、动作归一化和 current tensor cache；
2. current rollout typed path：把 action、old logprob、reward、done 放进现有 builder；
3. current actor/FSDP、weight sync、checkpoint/export；
4. 独立 joint worktree/venv；standalone oracle 环境保持只读。

### 4.4 推荐首条 current RLinf 配方

先做 **Fast-WAM + GRPO + move_stapler_pad**，不是 PPO：

- critic-free，迁移面比 value head 小；
- move_stapler 不饱和，较能看出学习；
- 每步 `128 trajectories`，C24 后为 `1024 unique actor transitions`；
- `global512 / micro待显存实测 / update2`，共 `2048 presentations`；
- group4、32 groups；固定种子独立 eval 从第一版就开启。

这套协议保持旧 Fast-WAM 的 group4 和模型合同，同时把 unique transitions、optimizer updates 和 sample presentations 对齐到成功 π0 经验。它是“π0-aligned candidate”，不是 Fast-WAM 官方 RL recipe。

GRPO 链路稳定后再做 PPO。PPO 需要新增 current tensor-cache value feature、修正 EV/value-clip 日志，并记录 value-head param delta；这就是为什么 PPO 不是第一条迁移线。

## 5. OGPO：旧结果的问题不是一句“更新太多”

### 5.1 旧实现有多大

AutoDL 旧实现锁定 commit `5d5c84e3...`，相对旧 RLinf 是 26 files、`+6121/-57`。它包含：primitive replay、10-head FP32 Q、target Q、EMA actor、whole denoise-chain sampler/scorer、OGPO+CA actor、专用 runner 与 checkpoint sidecar。它明显大于 PPO-DVAC 或 Fast-WAM adapter，不是配置搬运。

### 5.2 旧结果与失败边界

| run | 预算与终态 | fixed eval | 终止原因 |
|---|---|---|---|
| v1 | 35,000 rows、2,500 paired updates、exit0 | `5%→35%→5%` | 正常完成，最终性能回落 |
| v2 | 64,078/90,000 rows、2,703 updates | 最高40%，到60k仍30–40%波动 | checkpoint replay 整份 clone 导致 CPU RSS阶梯增长，Ray 95% kill；非GPU/数值崩溃 |

### 5.3 “每轮更新太多”应怎样精确理解

旧 v1/v2 的 UTD 是 `0.1/0.05`，都远低于 OGPO 官方所说普遍可工作的 UTD=1。因此证据**不支持总更新量过大**。

真正偏离官方节奏的是集中度：旧 runner 先并行收8个完整 episode，再连续执行约124–160（v1）或62–80（v2）次完整 π0 actor+critic update。官方伪代码是在每个 primitive `env.step` 后交错更新。于是可信假设是：

- actor burst 太长，online policy 与 EMA/reference 的距离在一轮内突然拉大；
- critic/CA 排序也可能在早期不稳。

但这仍是待测机制，不是定论；v2把 UTD 减半后最高 eval 到40%，既没有证明修好，也没有证明无效。更干净的后续变量是把 `utd_q` 与 `utd_pi` 分开并限制 actor burst，而不是继续把总 UTD盲目减小。

另一个必须先排的低级混杂是：旧 OGPO 从 native C50 改成C10，同时把 native sampler 换成OGPO tapered-SDE。step0 已经变化，因此先要在零更新下做 `native C50/C20/C10`，再固定C比较 native/OGPO sampler。

### 5.4 current port 要适配什么

数学可复用：OGPO+CA、10Q、EMA、same-chain ratio、B64/G8/K4/H50/C10。

必须重接：

1. current typed primitive transition；
2. current OpenPI 的 RTC、`model_actions/actions` 和 denoise hook；
3. `EmbodiedFSDPActor`、多 optimizer warmup、sync；
4. 独立 OGPO transition replay；
5. runner 的 row budget；
6. `local_shard` 模型 checkpoint + **流式/分块 replay sidecar**。`local_shard` 只能解决 FSDP 协调，不能解决旧 replay clone 的 CPU RSS。

首版迁移应保持 v2 的 UTD0.05 和其余算法语义，以分离“框架 port”与“算法改进”。port 通过一次 paired update + save/load 后，再单独讨论 actor interleave / `utd_q≠utd_pi`。

## 6. 推荐实施顺序与每条的完成定义

| 顺序 | 工作 | 为什么排这里 | 完成定义 |
|---:|---|---|---|
| 1 | π0 两卡 PPO Control | current 代码已有，风险最低；给 PPO-DVAC匹配对照 | fixed32 → one update → save/reload → fixed32 |
| 2 | π0 两卡 PPO-DVAC `[0,2]` | 复用现有 DVAC/current Action-Adv；只新增 actor切口 | 与Control leaf-level parity，one update/sidecar/reload闭环 |
| 3 | π0.5 两卡官方型 PPO | RLinf 官方 RoboTwin recipe；扩模型轴 | π0.5 checkpoint load、SFT fixed32、one update、reload fixed32 |
| 4 | Fast-WAM current GRPO | standalone oracle已通；模型适配中等 | B=1 parity、一次真实GRPO update、save/reload、fixed eval |
| 5 | OGPO current port | 自定义栈最多、checkpoint历史风险最大 | 零更新 sampler audit、一次paired update、streaming replay save/load |

如果论文优先级是“先证明跨模型”，可以把第3项提到第1项；但工程风险从小到大的顺序仍如上。

## 7. Git、环境和运行隔离

- 每条线从 source-lock `7d07a421...` 建独立 `codex/` branch/worktree；Fast-WAM 与 OGPO 分别保留自己的方法工作树。
- π0/π0.5 可以候选共享只读 OpenPI venv；Fast-WAM joint integration 使用独立 joint venv；official standalone venv永不安装RLinf。
- checkpoint、HF cache、RoboTwin资产和run均放 `/data/chenyiteng`；源码可放 `/data/chenyiteng/projects`，不写大产物到 `/home` 或根分区。
- 多个 RLinf job 继续使用已验证的 shared Ray + 独立 namespace/placement/`RLINF_CODE_WORKING_DIR`/绝对输出路径/local-shard；只清理 owned job。
- 当前 GPU4–7 训练不属于本规划授权范围；实施前必须刷新GPU、RAM、磁盘、网络和其他用户状态。

## 8. 当前停点

本轮建议默认锁定：

1. π0.5 首线用官方 RoboTwin PPO，保留 update5；
2. PPO-DVAC 只改 actor advantage，不改 critic；
3. Fast-WAM 不重做 standalone，current 集成先 GRPO/move_stapler；
4. OGPO 首版先忠实迁 v2 数学与 UTD0.05，算法节奏改动另立实验。

π0 PPO-DVAC分支为`codex/sz-ppo-dvac-action-adv-fix@74617ced...`，两卡formal正在GPU4--7运行。
π0.5分支为`codex/sz-pi05-robotwin-rl@256eeeb4...`；共同两卡合同为`64×4/GB512/MB32/update5/M5/local_shard`。三段smoke已全部闭环；DVAC Step2 ESS=`0.957`，GPU峰约61.9 GiB/卡，当前停在formal资源壳讨论前。Fast-WAM current与OGPO仍未启动。
