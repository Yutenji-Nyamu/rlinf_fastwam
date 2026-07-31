# QAM × π0 × RoboTwin：上下文与来源地图

最后更新：2026-07-29。

本文是来源索引，不是第二份实施计划。规范性结论只在
[`00_INDEX_AND_IMPLEMENTATION_PLAN.md`](00_INDEX_AND_IMPLEMENTATION_PLAN.md) 维护。

目标是让后续 QAM 窗口不再加载 DSRL、RLT、Fast-WAM 和全部历史材料：默认只读主计划，
需要追溯某条结论时再按本页定位到精确来源。

## 1. 后续窗口的最小读取集

每个新的 QAM 任务按顺序读取：

0. `AGENTS.md` 由工作区自动生效，不另复制进 QAM 文档；
1. 根目录 `PROJECT_CONTEXT.md`；
2. 根目录 `HANDOFF.md`；
3. 本专题 `00_INDEX_AND_IMPLEMENTATION_PLAN.md`；
4. `evidence/IMPLEMENTATION_LOG.md` 的最新未完成批次；
5. 只有讨论术语/B/F/C/M 选择时才读
   `02_METHOD_AND_PORT_DECISION_GUIDE.md`；
6. 只有遇到来源或调用链争议时才读本文对应条目。

不要默认读取：

- DSRL 全计划和长流水；
- RLT 全计划和长流水；
- `01_FULL_REFERENCE_HISTORY_20260728.md` 全文；
- Fast-WAM 诊断历史；
- 旧聊天或 Memories 的完整 rollout。

## 2. 可信顺序

不同事实不能混用一条总排序：

- **动态服务器状态**：本轮 live 只读检查 > 根 `HANDOFF.md` 的最新时间快照 >
  专题计划/账本 > 历史调查；
- **QAM 方法与代码语义**：锁定官方 commit > 对应论文版本 > 本专题解释 > 历史聊天；
- **当前授权和停点**：`AGENTS.md` + 根 `HANDOFF.md` > 专题主计划 > 旧账本；
- **π0/RoboTwin 可执行行为**：实施时锁定的 server commit/resolved config/运行产物 >
  本地 grep 镜像。

动态事实必须标时间；旧快照只用于定位，不能覆盖更新后的交接事实。

本地 `.research-rlinf` 在本轮为
`main@c5ca51cc21c007a41d287159f9e1b14e0200000e`、working tree clean。服务器 pin
`6d0db56b...` 是其祖先；本任务涉及的 runner/env/rollout/OpenPI/SAC-Flow/replay/sync/DCP
文件在两个 commit 间未见差异。它仍只是 grep 镜像，未来实现以重新刷新的服务器 baseline
为准。

## 3. 上下文资产总表

| 上下文 | 对 QAM 的用途 | 读取粒度 | 不能据此推出 |
|---|---|---|---|
| `AGENTS.md` | 自动生效的工作区执行、授权、账本、分支与 smoke 规则 | 自动注入/每轮生效 | QAM 算法参数 |
| `PROJECT_CONTEXT.md` | 稳定工作区规则、服务器/本地边界、账本与 smoke 批准习惯 | 每轮全文 | QAM 算法参数 |
| `HANDOFF.md` | 当前专题路由、授权和服务器动态停点 | 每轮全文 | 长期算法真值 |
| QAM 主计划 | 当前方法与工程 SSOT | 每轮全文 | 未执行测试的结果 |
| QAM 方法/决策指南 | 术语、官方与计划调用链、B/F/C/M 选择解释 | 方法讨论时 | 当前授权或已冻结结论 |
| QAM 实施账本 | 每次操作、结果、问题、修复、复测 | 读最新批次 | 论文方法主张 |
| DSRL 主计划/账本 | branch/worktree、config opt-in、compact replay、resume、同步、fixed-input 验收范例 | 按链接小段读取 | latent、reward、H/N、UTD、capacity 等 QAM 数值 |
| RLT 主计划/账本 | source lock、action/norm/resume contract、四态词、首 rollout sync | 按链接小段读取 | token/C10/z_rl/两阶段语义 |
| 传统 RL 历史全文 | 2026-07-28 的 QAM/DSRL/RLT 调查线索 | 仅追溯 QAM 对应节 | 当前规范或 live 状态 |
| 本地 `.research-rlinf` | 便于 grep 的 RLinf 上游代码镜像 | 精确 path/symbol | 服务器当前 HEAD |
| 本地 `.tmp/qam-official-2726d767` | 官方 QAM 锁定 commit 的临时只读研究 clone | 精确 path/symbol | 项目要直接 vendoring JAX 代码 |
| AutoDL live | 模型、数据、GPU、进程、Git、worktree 与实际运行入口 | 每次按需只读刷新 | 可在未授权下写入/运行 |
| 官方 QAM 论文/仓库 | QAM 方法和可执行 oracle | 固定版本/commit | π0/VLA 可行性或真实机器人效果 |

## 4. 官方 QAM 精确索引

### 4.1 主来源

- [arXiv v4 HTML](https://arxiv.org/html/2601.14234v4)
  - Section 4：QAM 分布、memoryless SDE、lean adjoint、AM objective；
  - Algorithm 1：训练步骤；
  - Appendix F：实现和超参数。
- [官方项目页](https://colinqiyangli.github.io/qam/)
  - 论文、代码、结果的作者入口。
- [ICLR 2026 官方页面](https://iclr.cc/virtual/2026/poster/10006800)
  - 论文状态与作者；
- [官方仓库锁定 commit](https://github.com/ColinQiyangLi/qam/tree/2726d767c9a0a7a46d49693f0391f73dc2cf58ac)
  - `agents/qam.py`
    - `critic_loss`
    - `adj_matching`
    - `actor_loss`
    - `_update`
    - `sample_actions`
    - `compute_flow_actions`
    - `create`
    - `get_config`
  - `utils/networks.py`
    - `ActorVectorField`
    - `Value`
  - `utils/datasets.py`
    - `Dataset.sample_sequence`
    - `ReplayBuffer`
  - `experiments/reproduce.py`
    - plain QAM、QAM-FQL、QAM-EDIT 的任务配置与运行规模。
- [OGBench 官方项目页](https://seohong.me/projects/ogbench/)
  - 区分 state-based 与 `visual-*` 环境；
- [OGBench Cube state observation](https://github.com/seohongpark/ogbench/blob/master/ogbench/manipspace/envs/cube_env.py#L731-L769)
  - nonvisual state 拼 proprio 与 simulator-derived block pose/orientation。

2026-07-29 当前性核验：

- arXiv 仍是 v4（2026-05-18）；
- `git ls-remote` 的 `main` 仍为
  `2726d767c9a0a7a46d49693f0391f73dc2cf58ac`；
- GitHub 显示 15 commits、无 tag/release；
- 官方 requirement 是宽松下界，不是 lock；无 CI、tests 或预训练 QAM checkpoint；
- 论文方法只写抽象 state $s$，全文未规定视觉/privileged modality 或 encoder；
- 正式复现实验使用不带 `visual-` 前缀的 OGBench/MuJoCo 环境；actor/critic 读取同一份
  observation，不是 asymmetric critic。OGBench manipulation 的 nonvisual state 含
  simulator-derived block pose/orientation；
- 代码是 low-dimensional state + 4×512 behavior/fine MLP + 10 个 4×512 critic MLP；
  没有真机、π0、视觉/语言 encoder。

### 4.2 代码事实与论文事实的差异

| 事项 | 代码事实 | 移植动作 |
|---|---|---|
| plain actor objective | behavior-FM + fine-flow AM | P1 oracle 两项都做；frozen π0 省略 FM 时标为端口适配 |
| terminal Q gradient | 默认 target critic | parity 按代码；文档注明 |
| terminal ensemble | target-Q mean gradient，不含 `rho × std` | 与悲观 TD bootstrap 分开断言 |
| lean-adjoint base | 默认 target actor slow | parity 按代码 |
| singular time | 使用 `t+h` | 不简化成 `t` |
| 真实动作采样 | `sample_actions/compute_flow_actions` 用 fine ODE | SDE 只在 trainer 的 AM 辅助轨迹 |
| 最后 SDE step | behavior-flow ODE step | 单独断言 |
| AM 时间积分 | flow-step sum，无显式 `h` | parity 按代码 |
| grad clip | global norm 1 | 不按论文 prose 改 elementwise |
| target update | 每 update EMA，读取 update 前 online 参数 | 保存一拍滞后的更新顺序 |
| `target_actor_fast` | 构造但无有效用途 | 不移植 |
| `residual=True` | 有代码，无官方复现命令 | 视作实验性适配 |

### 4.3 不相关或延后的上游内容

- `agents/bam.py`：basic-adjoint/backprop ablation，不是首版；
- QAM-FQL/QAM-EDIT：延后；
- QSM May fix：只修 QSM baseline，不是 QAM core 修复；
- OGBench result pickle/plot notebook：不构成 π0/RoboTwin 验收。

### 4.4 公开近邻实现与论文

| 来源 | 一手入口 | 对本任务的价值 | 不能据此推出 |
|---|---|---|---|
| LWD | [论文](https://arxiv.org/html/2605.00416) / [项目页](https://finch.agibot.com/research/lwd) | $\pi_{0.5}$ 类 VLA、真机、QAM；在线冻结 VLM、更新 action expert；VLM state feature + action-chunk critic，支持 B1/F1/C1 高层方向 | 未找到可直接抄的完整官方代码；DIVL/double-Q 不是 Plain 10-Q |
| Q-VGM | [论文](https://arxiv.org/html/2606.08015) | frozen prefix RL token + proprio + action-sensitive Q；可校准 C1 表示和 action injection | Euler look-forward/Q-gradient search，不是 adjoint matching |
| RL2-VLA QAM fork | [rl2-train branch](https://github.com/rl2-vla/qam/tree/rl2-train) | BridgeV2 + 预计算 π0/VLA latent 的 dataset adapter | flow 仍是小 MLP；不端到端更新 π0 expert，不是 RoboTwin/online port |
| TRQAM | [官方仓库](https://github.com/yonghdong/trqam) / [论文](https://arxiv.org/abs/2605.27079) | 独立 JAX QAM baseline、OGBench/Robomimic low-dim 交叉检查；后续稳定性参考 | 不是视觉 VLA 或 RoboTwin 实现 |
| Microsoft Adjoint Matching | [PyTorch/diffusers 代码](https://github.com/microsoft/soc-fine-tuning-sd) | 仅参考 PyTorch adjoint/VJP/autograd 写法 | 没有 TD critic、机器人 replay 或 QAM update |
| 非官方 LWD reproduction | [仓库](https://github.com/HaoyunT/lwd_wall_repo) | 只作负面代码线索 | 未完整实现 reverse-adjoint VJP，不能作为主抄对象 |

截至 2026-07-29，没有找到可直接移植的公开 PyTorch π0/OpenPI/RoboTwin Plain-QAM
实现。算法 oracle 仍唯一锁定官方 QAM；LWD/Q-VGM 只帮助判断 VLA 适配是否合理。

## 5. RLinf 系统来源

### 5.1 RoboTwin × π0 PPO/GRPO 家族

主路径：

- `examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml`
- `examples/embodiment/config/env/robotwin_adjust_bottle.yaml`
- `examples/embodiment/config/model/pi0.yaml`
- `rlinf/models/embodiment/openpi/openpi_action_model.py`
- `rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py`
- `rlinf/models/embodiment/openpi/policies/aloha_policy.py`
- `rlinf/workers/rollout/hf/huggingface_worker.py`
- `rlinf/workers/env/env_worker.py`
- `rlinf/runners/embodied_runner.py`

可复用：

- `adjust_bottle` task/env/assets/seeds；
- π0 checkpoint、三相机、language、state transform；
- norm/unnorm 和 active 14D action；
- `[B,H,D]` action chunk 到 RoboTwin `chunk_step`；
- actor/rollout/env group、weight sync、evaluation 和 DCP 运行壳。

不可复用：

- PPO ratio clipping、GAE、value head、old/new logprobs；
- GRPO group-relative reward/advantage 和 group sampling；
- PPO/GRPO 的 update epoch、batch、N、reward filter；
- 任何旧 PPO/GRPO checkpoint 作为 QAM resume。

2026-07-28 的服务器 pin 中，tracked RoboTwin OpenPI config 只有 Dagger/PPO 同族；
本地存在的临时 A800 PPO/GRPO 配置未进入锁定的 exact common commit，不能作为规范来源。

### 5.2 OpenPI 显式 velocity

`rlinf/models/embodiment/openpi/openpi_action_model.py`：

| symbol | QAM 用途 |
|---|---|
| `OpenPi0ForRLActionPrediction.nft_forward` | 传入显式 `x_t/timesteps`，返回 `v_theta` |
| `get_velocity` | action expert 在给定状态/时间的原始 velocity |
| `_build_prefix_cache` | 三相机/语言 prefix KV cache |
| `get_suffix_out` | state、action、time suffix 的可微前向 |
| `_sample_actions_with_prefix_cache` | fixed-noise rollout parity 的参考，不作为 AM 反传 API |
| `_sft_forward_with_rlt_prefix` | 明确 OpenPI 的时间/速度约定 |
| `predict_action_batch` | 将 normalized model action 经 `output_transform` 变成 env action；`forward_inputs` 同时保存两者 |
| `output_transform` | `Unnormalize` + `AlohaOutputs` 14D/坐标编码；不是 QAM action-gradient 路径 |
| `freeze_vlm` | 冻结边界参考；QAM 不能复用 DSRL 条件分支名称 |

已确认的时间约定：

```text
OpenPI: x_t = t * noise + (1-t) * action
OpenPI: target velocity = noise - action
QAM:    x_u = (1-u) * noise + u * action
QAM:    target velocity = action - noise
mapping: u = 1-t, f_QAM = -v_OpenPI
```

动作坐标合同：

```text
model_action: normalized [H_model, 32]
Q/replay:     P_N(model_action) -> normalized [N, 14]
env action:   output_transform(model_action) -> unnormalized/Aloha [N, 14]
```

planned `0:N` 不按事后结果裁剪。critic/VJP 只用 normalized 版本；env 版本只作执行与
provenance 对齐，Q gradient 不穿过 `Unnormalize/AlohaOutputs`。

官方 QAM 在 rollout、TD next action 和 `clip_adj=True` terminal-Q 路径使用
`[-1,1]` clamp。π0 端尚未直接照抄：P2 先量真实 normalized active action 的越界率，
再保证 Q 所见 canonical action 与送入 `output_transform` 的实际执行 action 完全一致。

服务器 checkpoint header 的 F1 只读计数（2026-07-29）：

```text
allowlist:
  gemma_expert.model
  action_in_proj / action_out_proj / state_proj
  action_time_mlp*

173 tensors
314,713,120 parameters
checkpoint storage 635,998,336 bytes
excluded gemma_expert.lm_head: 263,323,648 parameters
```

`gemma_expert.lm_head` 不在 action-velocity 调用链，不能因名称相邻而复制。FP32 Adam
两个 moments 约 2.34 GiB 总量；真实 activation/VJP/FSDP peak 仍必须在服务器实测。

现有 prefix 接口/探针：

```text
_build_prefix_cache -> prefix output/mask/KV
full prefix:  [1, 816, 2048] BF16
image prefix: [1, 768, 2048]
mask true:    773
```

C1 critic view 缓存三个 camera position block mean + 一个 language position block mean
的 `[4,2048]` BF16 feature，不缓存 full tokens/KV；attention 后这些 block 不是纯 source
feature。AM 与 TD next-action 还需要 frozen OpenPI prefix conditioning，因此 replay
另以 `obs_id/next_obs_id` 引用 canonical 三相机 uint8/task/proprio observation store，
采样时重算 prefix KV。block valid count、image shape/bytes、round-trip、pooling/transform
fingerprint 与 recompute 吞吐在 P2 固定。

### 5.3 NFT worker

`rlinf/workers/actor/fsdp_nft_policy_worker.py` 可借：

- 显式 denoise-state batch 的搬运；
- OpenPI forward type 路由；
- FSDP 下真实模型 loss/backward/update 的接线；
- chunk-level 聚合和固定输入测试组织。

不可借：

- NFT 自己的 advantage、flow-state sampling、DPO/MSE objective；
- NFT optimizer/update 数值；
- 整个 worker 复制后堆 QAM `if`。

### 5.4 SAC-Flow

主路径：

- `examples/embodiment/config/dosw1_pick_sac_flow.yaml`
- `examples/embodiment/config/maniskill_sac_flow_state.yaml`（服务器 pin 可见）
- `rlinf/models/embodiment/flow_policy/flow_policy.py`
  - `FlowPolicy`
  - `FlowStatePolicy`
  - `FlowTActor`
  - `JaxFlowTActor`
- `rlinf/workers/actor/fsdp_sac_policy_worker.py`
- 对应 SAC runner/replay/sync。

可借：

- off-policy actor/critic 更新的系统边界；
- replay batch → critic → actor → target 的调度；
- FSDP actor worker、日志和运行编排。

不可借：

- entropy-SAC target；
- alpha/target entropy；
- tanh Gaussian logprob；
- FlowPolicy 的现有低维网络；
- SAC 的 optimizer/update 次序而不与官方 QAM 对照。

现有 SAC target/resume 也不是可直接继承的正确实现：历史审查已经确认 target 初始化时机、
target/update-step checkpoint 不完整等风险；应复用 DSRL 已修复的工程思路，再为 QAM
建立自己的 round-trip 证据。

## 6. DSRL 与 RLT 的精确复用边界

### 6.1 DSRL

入口：

- `docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md`
- `docs/rlinf-robotwin-pi0-traditional-rl/evidence/IMPLEMENTATION_LOG.md`
- `docs/rlinf-robotwin-pi0-traditional-rl/evidence/SMOKE_APPROVAL_20260728.md`
- `docs/rlinf-robotwin-pi0-traditional-rl/evidence/SMOKE_EXECUTION_LOG_20260728.md`

只复用：

- 独立分支/worktree；
- config-opt-in 与 legacy default；
- compact transition replay 的工程模式；
- 每 rank local replay、global batch 按 world size 等分、全局计数 all-reduce、所有 rank
  同 update count 的两卡所有权模式；
- target-shadow/replay/RNG resume 的问题清单；
- 首 rollout sync；
- fixed observation/fixed noise parity；
- trainable/frozen 参数 delta；
- formal→smoke/resume 批准包；
- 完整实现/测试/smoke 流水账。

禁止继承：

```text
32D Gaussian latent
repeat-H latent
H=50 / N=20
-1/0 reward
gamma**N 的旧 DSRL 数值
UTD20
warm-up 500 global macro transitions
10Q 与 25k replay 作为“因为 DSRL 已验证”的默认值
stochastic eval
DSRL target-shadow checkpoint schema
```

即便 QAM 也可能使用 10Q，来源应写“官方 QAM 10Q”，不是“抄 DSRL 10Q”。

### 6.2 RLT

入口：

- `docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md`
- `docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md`

只复用：

- source lock 和 source→adapter→test 可追溯性；
- action、norm stats、resume contract fingerprint；
- 首 rollout full sync；
- `冻结 / 待事实验证 / 待运行批准 / 延后` 四态；
- 先收口高信息量语义，再实施主体批次。

禁止继承：

```text
RL token / z_rl
C10
Stage 1/Stage 2
reference/student route
BC+Q
deterministic student eval
per-rank warm-up 数值
RLT manifest/checkpoint schema
```

## 7. RoboTwin 当前 chunk 事实

2026-07-29 19:22 CST 服务器只读审查来源：

- `/root/autodl-tmp/RLinf/rlinf/envs/robotwin/robotwin_env.py` 的
  `RoboTwinEnv.chunk_step`；
- RoboTwin qpos 路径
  `RoboTwin_RLinf/envs/_base_task.py::gen_sparse_reward_data`；
- `EnvOutput.final_obs`、`_handle_auto_reset` 与 policy-version 数据结构；
- DSRL worktree 的
  `rlinf/workers/actor/fsdp_sac_policy_worker.py` 两卡 replay 所有权。

已确认：

- `chunk_step` 调用一次 `venv.step(chunk_actions)`；
- qpos 路径先把全部 planned waypoints 组装成一条 TOPP trajectory，再交给 simulator；
- 上层 `obs_list` 只 append 最终 observation；
- `chunk_rewards` 是 `[num_envs, chunk_step]`；
- termination/truncation 只写最后 slot；
- `_cal_chunk_rewards()` 当前把 `n_steps_to_run` 写死为 0，不能提供 planned-action
  `realized L`，并把 success reward 对齐到 query final slot；
- `_elapsed_steps` 按配置 chunk width 增加；
- `EnvOutput.final_obs`、`infos["final_observation"]` 和 policy version 基础设施存在。

因此生产 M2 定义为固定 N query-level macro，不再要求或伪造 $L$；
$R_{\rm macro}=\sum_{i=0}^{N-1}\gamma_{\rm slot}^i r_i$，
bootstrap 使用 $\Gamma_N=\gamma_{\rm slot}^N$；`slot` 是逻辑 planned waypoint/reward
位置，不是测得的 simulator primitive duration。实施仍需验证：

- resolved `use_rel_reward/use_custom_reward` 与 raw `[N]` reward vector；
- success/live/timeout 的 bootstrap mask；
- nonterminal next critic/policy observation view 与 timeout true-final view 的传递；
- planned normalized/env action、end 与 policy version 的 query 对齐。

M1 只有在 fixed-N credit 被实验证明过粗时，才讨论暴露 primitive observations；这不是
M2 的前置条件。

## 8. 模型、环境、数据源和运行资产

### 8.1 稳定资产路径与动态路由

动态 GPU/进程/RAM/磁盘/Git 状态只读根 `HANDOFF.md`，本文不复制时间快照。稳定定位是：

| 资产 | 路径/边界 |
|---|---|
| π0 SFT checkpoint | `/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle` |
| norm stats | checkpoint 内 `physical-intelligence/robotwin/norm_stats.json`；实施前重刷 hash |
| baseline | `/root/autodl-tmp/RLinf`；实施时重刷 HEAD/status |
| DSRL / RLT worktree | `/root/autodl-tmp/RLinf_fastwam_rlinf` / `/root/autodl-tmp/RLinf_rlt_pi0_robotwin` |
| QAM planned worktree | `/root/autodl-tmp/RLinf_qam_pi0_robotwin`；当前尚未创建 |
| shared clean-50 source | `/root/autodl-tmp/datasets/robotwin2/source/<revision>/dataset/adjust_bottle/aloha-agilex_clean_50.zip`；可选诊断，不是 v1 QAM 依赖 |
| QAM data | 由 QAM rollout 产生的 online macro replay；不复用 PPO/GRPO replay 或 clean-50 派生标签冒充 |
| production venv | `/root/autodl-tmp/RLinf/.venv`，只读复用；官方 oracle 另建独立 venv |

### 8.2 clean-50 下载 source lock

主来源：

- [RoboTwin π0 官方数据流程](https://robotwin-platform.github.io/doc/usage/Pi0.html)；
- [官方数据仓 TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)；
- Hugging Face tree API：
  `GET /api/datasets/TianxingChen/RoboTwin2.0/tree/<revision>/dataset/adjust_bottle?recursive=false&expand=true`。

2026-07-29 锁定值：

| 字段 | 值 |
|---|---|
| revision | `9dc9299c163db059931898a9f0852098a61155a1` |
| file | `dataset/adjust_bottle/aloha-agilex_clean_50.zip` |
| exact size | `298,659,710` bytes（284.82 MiB） |
| expected file SHA-256 (`lfs.oid`) | `5554b6b30e37c6ed2f0bbc48079e8ad79d9512e9d4f910a5e71b0d5ad8fbe50e` |
| pointer blob oid | `50d380b9c6a4dc5921a6ff9816ddc669c54e9d36`，不能作文件 SHA-256 |
| xet hash | `e2efedb79d7fd979d15c8a984af006e8a8a222e7ae1527c90048258cef4059c4`，不作本地校验 |

RLT owner 已将该对象下载到上述版本化 source 路径，并验证 exact size、file SHA-256、
ZIP 完整性、50 组 `pkl+hdf5+mp4+instruction` 与 archive path safety。QAM 只读复用，
不重复下载。**当前 online-only 推荐不依赖它，不为 QAM 解压、转换或写 sidecar。**

### 8.3 converter 与字段 provenance

官方 RoboTwin converter source：

- upstream `RoboTwin-Platform/RoboTwin@13c3c47ff4312dd62484bcd51be034af55c062d1`；
- server checkout `c3ddfa8b97d5519efa828b075999bd0006778e5e`；
- 本任务核对的四个 converter 文件在两个 commit 间逐字节相同。

| 文件 | SHA-256 |
|---|---|
| `policy/pi0/process_data_pi0.sh` | `868c92fbb76b9b7b2a8d1ecd630237df8d320c7aa3a37b5bd0e704d8c0d49bbe` |
| `policy/pi0/scripts/process_data.py` | `b462918bf3f41f6d2fc30c3498381ac3cc7d8ce7a8bd6333fafb925e7d9d5590` |
| `policy/pi0/generate.sh` | `d0a065051bbc1db08bd0486150b5908e15c6682df4df343c9723708d5bbf1eee` |
| `convert_aloha_data_to_lerobot_robotwin.py` | `b8f0829329e099b7246b3d6467cec3ea4d60767eedd219b825d0b7f26bb7c373` |

这些 source/converter 事实解释了 clean-50 为何不进 v1 Q replay：

- raw HDF5 按 converter 预期含逐帧三相机与 qpos，但单 episode key/shape 尚未实物检查；
- converter 用 `qpos[t+1]` 作为 `obs[t]` 的 action target，它不是显式记录的
  `env.step(action)` command；
- 数据没有逐步 reward、terminated、truncated、failure/timeout、π0 query boundary 或
  `policy_version`；
- LeRobot 转换还会丢掉构造末条 transition 所需的最后一帧 observation。

这些字段可以人为派生，却不能冒充 observed online transition。若未来其他任务需要转换，
converter 会删除重建目标 `repo_id`，因此仍必须由唯一 owner 执行；这不是当前 QAM
实施步骤。

## 9. 旧调查的去重路由

2026-07-28 的 DSRL/RLT/QAM 调查已归档在：

`docs/rlinf-robotwin-pi0-traditional-rl/01_FULL_REFERENCE_HISTORY_20260728.md`

其中 QAM 有用的历史判断只保留为来源线索：

- RLinf 当时没有 QAM；
- 官方实现是 JAX/Flax OGBench；
- 先做官方 oracle，再做 PyTorch 数值核，再接 RLinf；
- SAC-Flow 是工程桥，不是方法来源。

这些结论已经在本专题重新核对、分类并提升为当前主计划；后续不要把历史全文复制回来。

## 10. 远程认证与凭据处理

本专题不复制一套认证教程；沿用根 `PROJECT_CONTEXT.md`/`AGENTS.md`：Paramiko
password auth、关闭 key/agent、先做只读身份探针、只对 banner/EOF/timeout 有界重试，
凭据只驻留当前进程且不得写入文档、Git、日志或 artifact。

## 11. 上下文压缩后的问答索引

| 问题 | 先读 |
|---|---|
| 主要抄谁？官方是什么模型/仿真还是真机？ | 方法指南 §1–§2；主计划 §0 |
| QAM 数学、FM/AM/adjoint/VJP 到底是什么？ | 方法指南 §3；主计划 §2；本文 §4 |
| π0 是 ODE，为什么 QAM 文档还有 SDE？ | 方法指南 §3.4；主计划 §2.2：SDE 只用于 AM 辅助轨迹，执行/TD 用 fine ODE |
| Q 是不是给每个去噪步做 FM 监督？ | 不是；精确的 FM→终点 Q gradient→VJP→AM 链见方法指南 §3.2–§3.6 |
| plain/F/E、B1/B2、F1/F2 有何区别？ | 方法指南 §4–§5；主计划 §4.2 |
| 官方为何没有 critic encoder 可抄？C1/C2/C3 是什么？ | 方法指南 §6；主计划 §4.3 |
| 当前收束主线是什么、还剩哪些事实？ | 主计划 §4.4、§11；方法指南 §5.5、§13 |
| π0 的时间为什么要翻转？ | 方法指南 §8；主计划 §3.1；本文 §5.2 |
| normalized model action、Q action、env action 有何区别？ | 主计划 §3.3；方法指南 §7.2；本文 §5.2 |
| H=50/N=20 能否直接用？fixed-N/L/mask 怎么分？ | 主计划 §3.3、§6；方法指南 §7 |
| RoboTwin 为何不是官方 exact Q-chunk？ | 方法指南 §7；主计划 §3.4；本文 §7 |
| v1 的 Q 数据从哪里来？ | RoboTwin online execution；见方法指南 §9、主计划 §7 |
| clean-50 是否进入 v1？ | 否；只作可选诊断资产，精确 source lock 见本文 §8.2 |
| venv/worktree 怎么隔离？ | 方法指南 §10；主计划 §7.1 |
| Q 是什么、与 SAC 有何异同、一次 update 怎么走？ | 方法指南 §3.2 |
| 10-Q、bootstrap、target、terminal gradient 如何区分？ | 方法指南 §3.7–§3.8 |
| QAM 是不是 Diffusion Policy？ | 方法指南 §2.3 |
| 官方与计划调用链、逐文件改动是什么？ | 方法指南 §11–§13；主计划 §5、§9 |
| 还有没有 π0/VLA/真机 QAM 可参考？ | 本文 §4.4；方法指南 §1 |
| DSRL/RLT 能抄什么？ | 本文 §6 |
| F1/C1/payload/两卡/超参实施合同在哪里？ | 主计划 §4.2–§6、§10–§11；方法指南 §13 |
| 测试和 smoke 怎么管？ | 主计划 §8、§10 |
| 当前允许做什么？ | 主计划 §12；根 `HANDOFF.md` |
