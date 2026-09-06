# 参考材料矩阵

## 1. 调查策略

参考材料不应等深阅读。只对与目标调用图相交的文件做逐行审计，其余材料用于交叉验证或事故索引。

- **P0：必须彻查**——固定 commit、文件调用图、tensor contract、train/eval、FSDP、同步和 checkpoint。
- **P1：定向深查**——只看与迁移模式或历史失败直接相关的文件与提交。
- **P2：按需查询**——保留关键词和章节索引，不重新阅读全部流水账。

## 2. 材料目录

| 材料 | 权威性 | 提供的信息 | 可复用内容 | 主要风险 | 深度 |
|---|---|---|---|---|---|
| 服务器 π0 + GRPO 基座 | 本项目运行真相 | 真实 RLinf、RoboTwin、GRPO、Ray、双卡、CPU patch transport、DCP | runner、环境、分组、reward、同步、checkpoint、监控 | 2026-07-17 20:05已现场确认pin；实施前仍做短status防漂移 | P0，现场状态优先 |
| `audits/20260717-084926-grpo-current` | 历史结构化证据 | 100-step 指标、resolved config、资源与 checkpoint | 资源预算、回归基线、审计流程 | 不是源码，也不是实时状态 | P0，提炼摘要 |
| Fast-WAM 官方仓库 `45d8e145` | 模型官方权威 | 模型、scheduler、checkpoint、RoboTwin adapter | 三相机、14D qpos、norm、horizon、denoise | `infer_action` 只有推理语义，没有 RL logprob/replay | P0，相关调用图逐行 |
| Fast-WAM 官方论文/权重/stats | 官方协议 | 架构、训练目标、评测协议、normalization | 设计依据和 baseline | 论文数字不是本机复现 | P0，接口与协议 |
| Fast-WAM standalone 安装指南 | 项目 runbook | AutoDL 安装与 smoke 顺序 | 用户后续操作索引 | 执行前仍需核对官方 HEAD/环境 | P1 |
| 社区 Fast-WAM×RLinf `fd780f0` | 已跑通的社区实验 | policy wrapper、Flow-SDE、critic-free GRPO、早期raw-video PPO critic、root FSDP2且禁用block/expert wrap、bucket sync、学习与泛化报告 | 第一版概率/FSDP/同步骨架、PPO feature反例和 bug 测试清单 | PPO正向学习证据不足；LIBERO两相机/7D；4×A100；RLinf仍会单独wrap非tied Embedding；proprio虽在allowlist但实际无RL梯度；单任务提升伴随全130任务遗忘；非上游 | P0，按提交链 |
| RLinf server pin `6d0db56` | 当前系统代码基线 | BasePolicy、OpenPI/π0.5 PPO critic、GAE/loss、worker、FSDP、sync、RoboTwin | 实际承载接口；value LR/head wrap/old-new-bootstrap/DCP | 现场可能已有未记录修改 | P0 |
| RLinf v0.3/current main | 官方对照 | 新扩展点、修复与参考实现 | 选择性 backport 和 StarVLA 模板 | 不应为版本号整体切换 | P0，相关子图 |
| RoboTwin `RLinf_support@0008ae68` | 环境官方 | VectorEnv、obs、qpos action、reward/reset | 14D qpos 与 group/reset 契约 | 与 Fast-WAM vendored RoboTwin 不是同一树 | P0，接口级 |
| Fast-WAM vendored RoboTwin | standalone 权威 | 官方推理实际依赖的 evaluator/env | standalone parity | 不整体覆盖 RLinf RoboTwin | P1，针对性 diff |
| π0 最新日志文档 | 项目历史 | 新 RLinf 迁移、GRPO 100-step、offload/RAM/DCP | 已验证参数与事故史 | 时间线有重复和过期判断 | P1，事实表 |
| 早期 OpenPI 文档 | 历史证据 | 初次 PPO/eval/value/OOM | 基线演化与主机 RAM 教训 | 旧仓库和旧配置 | P2 |
| Motus 官方与接入代码 | 官方模型 + 已跑通大部分工程链的项目实验 | 官方 wrapper、batch、tensor-only replay、model-side logprob、FSDP、同步、DCP、GRPO/PPO | 最接近本任务的 WAM→RLinf 工程模板 | 旧 RLinf；完整 video chain/RAM；chunk sum、critic 与学习质量未成功 | P0，目标调用图逐文件 |
| LaWAM 官方与接入代码 | 官方模型 + 局部跑通的项目实验 | flow expert、value dtype、EE adapter、batch、planner 边界 | dtype/注册/replay/环境失败门 | EE planner 与 Fast-WAM qpos 不同；g4 rollout 曾被 planner timeout 阻断 | P1 |
| `Exp_snd.md` | 原始流水账 | Motus standalone、TTS/OPD、环境问题 | 偶尔查询旧现象 | 约 1.3 万行，核心相关度低 | P2，只建索引 |

## 3. P0 彻查标准

对 P0 材料的“彻查”至少产生以下记录：

1. repo URL、branch、完整 commit、license 和本地/服务器路径。
2. 与目标调用图相交的文件和入口函数。
3. 输入/输出 shape、dtype、device、normalized/physical space。
4. train/eval 分支、`no_grad` 边界和 trainable parameter allowlist。
5. rollout replay 字段、old/new logprob、mask 和聚合粒度。
6. FSDP wrap、权重同步、state-dict key 和 checkpoint 格式。
7. 把历史 bug 转成可执行回归测试。

不需要逐行调查 RLinf 全仓库或把五份日志重写一遍。应维护三个干净 diff：

```text
服务器 RLinf 当前工作树 vs server pin/upstream
社区 Fast-WAM 分支 fd780f0 vs 其基线 5d75412a
未来 Fast-WAM 功能分支 vs 当前 π0 server pin
```

## 4. 历史文档入口

- `E:\0school\研一下\aaai\07170856\pi0 + ppo_grpo.md`
- `E:\0school\研一下\aaai\07150939\Openpi + PPO AutoDL A800.md`
- `E:\0school\研一下\aaai\07150939\Motus + RLinf.md`
- `E:\0school\研一下\aaai\07150939\lawam rlinf.md`
- `E:\0school\研一下\aaai\07150939\Exp_snd.md`

本轮已对五份文档完成整份结构/标题索引，并读取与 standalone、RoboTwin、rollout、GRPO/PPO、FSDP、同步、OOM 和失败点直接相关的章节；不是只看旧交接摘要。它们的职责如下：

| 文档 | 本任务采用的内容 |
|---|---|
| `pi0 + ppo_grpo.md` | 当前 π0→官方新 RLinf 的迁移、100-step GRPO、offload/RAM/DCP 与配置演化 |
| `Openpi + PPO AutoDL A800.md` | 早期 OpenPI PPO/eval/value/OOM 事故与为何转向 actor-only GRPO |
| `Motus + RLinf.md` | Motus standalone、RLinf 接入、20/30-step GRPO、PPO、FSDP、同步和 Ray/主机 OOM 时间线 |
| `lawam rlinf.md` | LaWAM model/actor/rollout/env 初始化、end-pose/value dtype、PPO Global Step 1 与 planner timeout |
| `Exp_snd.md` | Motus standalone、TTS/OPD、官方数据预处理与 batch 历史；只在对应问题出现时查询 |

这些文件保存历史证据；精确代码行为仍以锁定源码和实施前服务器只读审计为准。

## 5. 来源到实现的约束表

| 目标部分 | 第一依据 | 第二依据 | 明确不做 |
|---|---|---|---|
| 三相机/state/prompt/norm/denorm | 官方 Fast-WAM RoboTwin deploy | standalone 黄金 fixture | 不用社区 LIBERO 图像/夹爪适配 |
| RLinf rollout/replay 外层 | 当前 π0 基座 | Motus 已跑通 wrapper | 不复制 Motus video chain/debug/critic |
| Flow-SDE 与 actor 重算 | 社区 Fast-WAM 最终链 | π0 的同 transition replay | 不重新发明概率公式；不使用社区历史错误 denominator |
| FSDP | 社区 root FSDP2、禁用block/expert wrap | 官方 MoT 实际调用方式 + RLinf pin默认单独wrap非tied Embedding | 不自定义block/expert wrap；实施时审计实际FSDPModule inventory |
| 权重同步 | 社区 GPU-default bucket sync 的跑通结果 | π0/Motus selective-sync与CPU transport机制 | 不复制社区已OOM的sparse patch路径；CPU bucket若选用明确标为项目适配 |
| GRPO runner/reward/loss | 当前 π0 基座 | 社区 critic-free GRPO | 不在 Fast-WAM policy 内再实现一套 loss |
| PPO critic/GAE/loss | RLinf π0/π0.5当前pin | 社区Fast-WAM raw-video critic、Motus/LaWAM observation critic | 外层复用RLinf；Fast-WAM只适配last-video-cache-V feature，不复制社区浅层feature或另写loss |
| checkpoint | RLinf DCP + pin `no_dist=True` converter + 官方 Fast-WAM deploy schema | Motus 的 resume/export 教训 | resume与deploy分层；不写matching-world-size gather，不把别名重复的整棵state dict直接导出 |

因此“迁移”不是整仓 merge：每个目标文件都有明确来源，且任何与来源不同的改动都必须标成兼容性修复或后续实验变量。

## 6. 已确认的成功边界

| 参考链 | 已确认成功 | 仍未证明/已失败 |
|---|---|---|
| π0 基座 | RoboTwin + RLinf + critic-free GRPO 完成 100 steps；通用 trajectory、actor、同步、DCP 可用 | 这是 π0 系统基座，不证明 Fast-WAM adapter 正确 |
| 社区 Fast-WAM | unsaturated LIBERO-10 从 `0.820→0.926`，独立复现 `0.836→0.904`；root FSDP2（禁block/expert wrap）、bucket sync、resume/eval 跑通 | full LIBERO-130 `0.921→0.877`，存在专项化/遗忘；proprio 实际无 RL 梯度；实际wrap还含RLinf默认Embedding |
| Motus | official inference、batch rollout、chain replay、old/new logprob、FSDP、同步、DCP、GRPO/PPO backward 基本贯通 | 一组 eval `0.875→0.6875→0.3125`；critic、chunk ratio 和学习质量未成功；长 PPO 遇到 Ray/主机 OOM |
| LaWAM | model/actor/rollout/env 初始化，value dtype 修复，b1 PPO 完成 Global Step 1 | g4 rollout 被 RoboTwin planner timeout 阻断；EE planner 不能迁移到 Fast-WAM qpos |

这张表决定引用方式：社区代码证明 Fast-WAM RL 工程与学习可以成立；Motus 证明 WAM 模型怎样干净进入 RLinf；π0 证明本机系统外层；官方 Fast-WAM 决定最终数值语义。

## 7. 历史反例如何约束本次实现

原始日志只保留证据，不把流水账复制进计划。实施时按下面的“错误 → 不变量 → 集中验收”使用：

| 历史证据 | 已发生的问题 | 本次不可退化的不变量 | 集中验收 |
|---|---|---|---|
| `Motus + RLinf.md:924-948,1088-1126` | 初版只有 B=1 eval；B>1 逐样本循环且 `forward_inputs` 为空 | singleton 只作官方 oracle；生产 B>1 必须真实 batched，train 必须返回标准 replay | P1/P2 |
| `Motus + RLinf.md:1184-1210` | 缺一只 wrist 时复制另一只 | 三路相机缺失、次序或 shape 错立即失败，不复制图像兜底 | P1 |
| `Motus + RLinf.md:4577-4625,5925-5939` | 手写第二套 train denoise，且 T5 只 pad 到 batch max，导致 eval 正常而 rollout 全失败 | eval/train 共用 adapter、固定文本长度、conditioning、scheduler、velocity 和同一去噪循环 | P1/P2 |
| `Motus + RLinf.md:6751-6758` | 可变 token 长度进入 trajectory 后 `torch.stack` 失败 | 所有 `forward_inputs` 在决策步和 batch 上固定 shape，只包含 Tensor | P1/P2 |
| `Motus + RLinf.md:4015-4035,4184-4204` | 为过 smoke 关闭完整 init sync、冻结/忽略约 7.6M 有效模块 | 保留标准初始同步；不为 FSDP 报错静默缩小既定 trainable 集合 | P2 |
| `Motus + RLinf.md:6448-6471` | backward 跑通但独立 eval `0.875→0.6875→0.3125` | runner step 只证明工程接通；更新后必须做固定 seed 独立 eval | P3 |
| `Motus + RLinf.md:1968-2056`、`pi0 + ppo_grpo.md:221-311` | 在现役 π0 venv 增量安装 Motus 后，`transformers` 与 OpenPI 要求冲突；最终只能恢复 golden backup | π0清单只作只读参考；第三个联合venv在独立最终路径从空创建，不修改/复制/搬迁现役π0 venv | I0/P2 |
| `lawam rlinf.md:1525-1558,1707-1760` | 手写 adapter 偏离官方；EnvOutput 白名单丢字段，训推都错 | 复用官方预处理语义并逐层检查 env→worker→policy 字段；不在 policy 内补假数据 | P1/P2 |
| `lawam rlinf.md:2094-2141,2391-2444` | FSDP value dtype 错；B=1 step 成功仍被 B=4 EE planner 阻断 | FSDP 前审计 dtype/device；区分模型、RLinf 和环境执行根因；LaWAM EE/planner 不迁移到 Fast-WAM qpos | P2 |
| `lawam rlinf.md:1131-1182` | 增量安装把 OmegaConf 降级，RLinf Hydra 因缺 `omegaconf.vendor` 失效 | 联合环境对 Hydra/OmegaConf 做显式兼容锁与 compose 测试，不让模型安装器隐式改基础配置栈 | I0/P2 |
| 社区提交 `0f0fcf4→9654a5a` | 曾用 actor 当前 logprob 代替 behavior denominator 来制造 ratio≈1，随后撤销 | old logprob 永远来自 rollout 真实采样；参数不变时自然对齐，参数变化后 old 保持不变 | P1/P2 |

因此只迁移后期收敛实现：官方预处理、真实 batch、model-side 共核、标准 RLinf trajectory/loss/sync/DCP。早期“先凑通”的 singleton、双 denoise loop、可变 replay、伪 old-logprob 和关闭同步都只作为反例。

## 8. 模型特性不能跨项目照抄

Motus 和 LaWAM 提供的是接入模式与失败证据，不是 Fast-WAM 的数值常量。实施时使用下面的边界：

| 主题 | 可迁移模式 | Fast-WAM 必须使用自己的事实 |
|---|---|---|
| Batch | Motus 后期“逐样本官方预处理后 stack，再一次 model-side batch forward”；RLinf OpenPI 的 batch rollout/replay | Fast-WAM 官方 `encode_prompt(Sequence)`、proprio `[B,D]`、官方 VAE batch wrapper、MoT KV cache 和 action expert；禁止复用 singleton fallback |
| 文本 | 固定长度 Tensor 才能进入 trajectory | 长度来自 Fast-WAM resolved `tokenizer_max_len`，当前锁定 YAML 为 128；不能复制 Motus 的 256/512 或 batch-max padding |
| 观测/动作 | adapter 先与官方 wrapper 做 parity | Fast-WAM 三相机 384×320、14D qpos、官方 stats；不迁移 LaWAM endpose/EE/planner，也不迁移社区 LIBERO 两相机/7D |
| 去噪 | train/eval/model-side replay 共核 | Fast-WAM shifted continuous scheduler、raw timestep、signed delta、有效 `sigma_shift`；Flow-SDE 概率定义再对齐 RLinf 官方 OpenPI |
| 冻结/FSDP | 模型作为真实 `nn.Module` 子树暴露给 RLinf | 先 canonicalize Fast-WAM 的 `mot/video_expert/action_expert/dit` 别名；不能复制 Motus 为过 FSDP 而忽略或少训有效模块 |

这也是“适配有理由”的含义：参考实现决定结构；模型自己的官方代码决定 shape、时间、normalization 和 checkpoint。两者冲突时必须记录差异并用集中验收解决，不能默选更简单的一边。
