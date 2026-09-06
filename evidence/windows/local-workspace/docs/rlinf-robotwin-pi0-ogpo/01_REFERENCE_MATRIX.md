# OGPO 专题参考矩阵

本文件回答“每个参考有什么、什么时候有用、程度如何”。它是
[`00_INDEX_AND_IMPLEMENTATION_PLAN.md`](00_INDEX_AND_IMPLEMENTATION_PLAN.md) 的条件读取附录，不是第二份实施计划。

## 1. 权威等级

| 标记 | 含义 |
|---|---|
| P0-M | OGPO 方法语义真值；冲突时决定目标、损失与更新规则 |
| P0-S | 目标系统真值；决定 RLinf/π0/RoboTwin 的现有可执行接口 |
| P1-E | 已验证工程参考；只迁移明确合同，不迁移算法目标 |
| P2-H | 历史/运维参考；用于定位，不用于当前状态或方法选择 |

P0-M 与 P0-S 不是谁压过谁：前者回答“OGPO 应该是什么”，后者回答“在这个系统里接口实际
是什么”。二者接缝处出现的差异必须作为显式适配记录。

## 2. 来源职责总表

| 来源 | 等级 | 提供 | 不提供 | 读取触发 |
|---|---|---|---|---|
| [OGPO paper v4](https://arxiv.org/html/2605.03065v4) | P0-M | bi-level/denoising MDP、off-policy TD critic、group-Q baseline、inner PPO、SDE correction、chunk return、OGPO+/CA | RLinf/RoboTwin 接口、π0 参数所有权 | 任一方法/公式/超参争议 |
| [`OGPO_public@0b3be413`](https://github.com/simchowitzlabpublic/OGPO_public/commit/0b3be413cde766a41257c6b19c0c2b06393a557f) | P0-M | JAX/Flax 可执行 oracle；`OGPOAgent`、runner、config、scripts、critic/actor update 顺序 | 现成 π0/RoboTwin backend | 对齐公式、update 或 source symbol |
| [RLinf RoboTwin guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html) + [`RLinf@36baa750`](https://github.com/RLinf/RLinf/commit/36baa75031ff863449322b548be49bd5620698ef) | P0-S | π0 PPO RoboTwin config、runner/worker/policy/sync/DCP 主链 | OGPO critic/replay/inner objective | 目标文件与最小 diff 设计 |
| 服务器实施父基线 `/root/autodl-tmp/RLinf@6d0db56b`；OGPO child `5d5c84e3` | P0-S | 首版可执行父 commit、K=4/H=50、共享 `.venv`、SFT/RoboTwin 路径；child 是当前已推送实现 | 永久当前的资源/进程、最新上游语义、OGPO objective | 实现、运行或恢复前现场刷新；最新快照只写 HANDOFF/账本 |
| [`OpenPI@15a9616a`](https://github.com/Physical-Intelligence/openpi/commit/15a9616a00943ada6c20a0f158e3adb39df2ccac) | P0-S | π0/π0.5 flow、checkpoint、prefix、action、transform、norm stats | RLinf 分布式训练和 OGPO | 模型行为、参数/时间/动作争议 |
| [RoboTwin main@13c3c47f](https://github.com/RoboTwin-Platform/RoboTwin/commit/13c3c47ff4312dd62484bcd51be034af55c062d1) / [`RLinf_support@0008ae68`](https://github.com/RoboTwin-Platform/RoboTwin/tree/0008ae6800df9f75fc8de7098bacb01735fd8fd2) | P0-S | observation/action、task、success/reset、数据与仿真；后者是 RLinf 集成兼容路径 | OGPO、RLinf worker 与 critic | env transition、兼容性与评估争议 |
| 本机 `.research-rlinf@c5ca51c` | P1-E | 可离线搜索的官方 RLinf 快照、精确 path/symbol | 服务器实施基线或最新官方状态 | 做静态 source map 后再回服务器核对 |
| `docs/rlinf-robotwin-pi0-traditional-rl/` | P1-E | C=20 先例、reward/done、Q/target、FP32 target-shadow、resume、资源格式 | OGPO actor、latent DSRL 目标与超参 | target/resume 接缝问题 |
| `docs/rlinf-robotwin-pi0-rltoken/` | P1-E | `H=50,C=10,D=14`、canonical norm、transition、sync、resume fingerprint、eval | OGPO objective、RLT token/stage 参数 | execution prefix、action coordinate、sync/DCP |
| `docs/rlinf-robotwin-pi0-qam/` | P1-E | 四块 π0 prefix pooling、10Q FP32、raw-observation/replay/resume、tied PaliGemma head 的 FSDP ownership 窄修复 | AM/VJP 迁移依据、OGPO 效果或 C 的因果结论 | critic/replay 接口或精确同构 FSDP 错误时窄读 |
| `docs/fastwam-robotwin-rlinf-grpo/` | P2-H | 已遇到过的 Flow-SDE/logprob、FSDP/sync/Ray 现象 | π0/OGPO 数学与有效超参 | 新实现真的遇到同类具体问题时再查 |
| `audits/` | P2-H | 历史 resolved config、metrics、resources、logs、图 | 当前状态、严格 A/B、失败原因 | 复现某次具体 run 时读取 |
| `exports/` | P1-E + P2-H | DSRL/RLT/QAM 高信息量 runtime、关键源码/日志/manifest | 全部 checkpoint/模型/数据 | 定向核对某个历史实现证据 |
| `local_scripts/remote_*` | P2-H | 历史只读探针、launcher、验收、收尾命令 | 当前可直接重放的 runbook | 新建有界命令前参考，不盲目重放 |
| 根 `HANDOFF.md` | 当前操作真值 | 当前路由、授权边界、live 停点 | 算法定义 | 每次实施前确认可做与不可做的范围 |
| `docs/project-history/` | P2-H | 旧停点与历史归档 | 当前授权、算法定义或当前服务器事实 | 只在追溯时按索引读取 |

## 3. 每个实现组件主要像谁

| 组件 | 第一来源 | 第二来源 | π0/RoboTwin 最终做法 |
|---|---|---|---|
| OGPO+CA loss/update | OGPO paper + `OGPOAgent` | 无 | success-only flow-BC + per-head CA；无 vanilla runtime |
| runner/worker/FSDP | RLinf π0 PPO | DSRL/QAM worker | 保留 lifecycle，新增 `embodied_ogpo` owner，不走 GAE/value |
| π0 actor ownership | RLinf `train_expert_only` | QAM F1 | frozen VLM + full action expert/projections + EMA expert；无 value head |
| tied PaliGemma FSDP ownership | RLinf FSDP1 auto-wrap | QAM 同模型/同错误的已验证修复 | OGPO-only 重命名 tied `lm_head` wrap tag，使 head/embedding 同留 root；独立 expert head 不变 |
| real action trace | OGPO `action_queue` | RLinf `RoboTwinEnv.step` | H=50 预测，C=10 singleton execution，存 primitive transitions |
| C=10/canonical action | RLT RoboTwin Stage 2 | DSRL C=20 作为可行性旁证 | Q 使用 normalized canonical `[10,14]`；env 执行同一语义动作经 output transform 后的 physical `[10,14]` |
| sequence replay/return | OGPO dataset/q-helper | QAM compact ring | 连续 primitive sequence，`Σγ^i r_i + γ^h Q`，success 仅索引 view |
| critic 数学/网络 | OGPO PaliGemma image critic | QAM prefix-pool/FSDP 形态 | 10 个独立 FP32 5×512 Q；四块 frozen prefix + proprio + action |
| whole-chain likelihood | OGPO `pg_helper.py` | RLinf π0 prefix/velocity/logprob | 对完整 raw `[K+1,50,32]` chain 做 same-chain score；Q/env 另取 `[10,14]` 前缀；修正 pre/post-clip 评分点错配 |
| target/sync/resume | OGPO EMA/target 语义 | DSRL target-shadow + QAM/RLT manifest | online/EMA expert 随模型同步并按 mode 选用；DCP + rank sidecars + replay/success fingerprint |

## 4. 精确的窄读取点

### DSRL

- `00_INDEX_AND_IMPLEMENTATION_PLAN.md` §5：`H=50,N=20` 与每 episode 决策密度；
- 同文件 §7–§9：reward/end、flat replay、FP32 target-shadow/resume；
- `evidence/FORMAL_CLOSEOUT_REPORT_STEP198_20260729.md`：只用于预算/资源/评估口径。

### RLT

- `00_INDEX_AND_IMPLEMENTATION_PLAN.md` §2：`H=50,C=10,D=14`；
- §4.2：`Q(z_rl,proprio,action[C,D])` 接口；
- §7.1/§7.3/§7.4：调用链、compact transition、per-slot return；
- §8：trainer sidecar、fingerprint、completion manifest、首次动作前同步。

### QAM

- `00_INDEX_AND_IMPLEMENTATION_PLAN.md` §3.4：RoboTwin fixed-slot 旧适配及限制；
- §4.3：四块 frozen prefix + proprio + flattened action 的 10Q 接口；
- §6.1–§6.3：raw observation 去重、feature cache、ring 与 resume；
- `evidence/IMPLEMENTATION_LOG.md`“失败 1：tied embedding 跨 FSDP ownership boundary”与“通过结果”：
  `FULL_SHARD + use_orig_params=true` 下 tied `embed_tokens/lm_head` 的二维权重错误与两卡窄修复；
- `evidence/QAM_FORMAL_STOP247_CLOSEOUT_20260801.md`：TD fit 不等于 held-out action ranking。

OGPO 不使用 `dQ/da`，所以 QAM 的 VJP/adjoint 主体默认不读；只有讨论为什么不用 BPTT/
critic gradient，或核对 π0 action expert ownership 时才打开。

### Fast-WAM PPO/GRPO

- `00_INDEX.md` §3：哪些链路已验证、哪些结果不足；
- `01_REFERENCE_MATRIX.md`：官方、社区、Motus、LaWAM 的职责；
- `05_IMPLEMENTATION_PLAN.md` 中 old/new logprob、actor-rollout sync、value observability 的
  对应章节；
- `audits/20260720-1257-fastwam-ppo-current/` 等只在追溯具体 run 时读取。

## 5. 历史结果怎样表述

QAM、Fast-WAM PPO/GRPO 只记录当时协议下“没有建立预期收益”的事实；现有证据不能确定失败
原因，也不能据此改变 OGPO 的 C、损失或检查清单。只有新实现出现同类具体症状时，才按精确
日志入口回查。历史 run 必须带时间、路径、协议和样本数；服务器当前状态仍以现场刷新为准。

## 6. 保留、降温与淘汰规则

- P0 来源保持 active，并在实现前锁 commit/tag；
- P1 只保留精确章节/文件链接，不复制大段参数；
- P2 保持冷归档，不删除，但退出 OGPO 默认上下文；
- 发现来源冲突时记录“官方继承、必要适配、已知缺陷、新方法改动”四类之一；
- 不删除本机或服务器材料；任何清理仍需精确清单、大小、可恢复性和另行授权。
