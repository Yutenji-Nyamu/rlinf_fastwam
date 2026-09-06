# 官方 baseline 调用链与旧 RLT → 新 upstream 迁移图

> 主体调用链最初锁于 `RLinf/RLinf@89b0cd5fce528180559bbd50d055fb2e21a25bb5`（2026-08-19）；
> 2026-08-21 已把 main 刷新到 `7d07a4212ee6858cc333e1d4fab7a37256d1f839` 并对本任务相关
> 文件做增量复核。开始 clone 时仍需再刷新一次并冻结，不把移动的 `main` 当 revision。

## 0. 2026-08-21 latest 增量与迁移量结论

比较锚点：共同旧 upstream `6d0db56bf26f972cd27fa29535f5eb939e80e5bf`、旧个人 RLT 终态
`2b8199d8ab2e7b110994fd3234bf7007196c3af9`、今日 official main `7d07a421...`。

| 问题 | 逐文件结果 | 对选择的含义 |
|---|---|---|
| PPO 要不要移植算法？ | `robotwin_env.py`、RoboTwin π0 PPO YAML、`model/pi0.yaml`、env YAML、patch syncer 在 `6d` 与 `7d` blob 相同 | 不要；新 main 只需重装环境、compose/model/sim 验证 |
| GRPO 主体是否重写？ | 旧 `compute_grpo_advantages` 与 actor loss 未改；新树另加 grpo-video/OPD，不替换旧主体 | 恢复精确私有 config/接缝并复测，预计小改 |
| 旧个人 RLT 有多大？ | `48a775db...→2b8199d8...` 恰为 20 个非文档文件、约 `+3400/-24`：7 core source、7 config/seed、2 tests、4 tools | 不是一整个 framework fork；可以按合同分批迁移 |
| 文本冲突多大？ | 7 core files 对 `7d` 的 three-way patch check 全 clean；应用后 13 个 Python 文件 AST compile 通过 | 初步移植不是全面重写，但尚未证明 import/compose/runtime/语义 |
| 新版直接收益 | causal AR reconstruction+单测、expert-takeover ref-chunk fix、exact π0+RoboTwin `openpi_rlinf` SFT/eval loader/converters、`rlinf-openpi==0.1.1`、runner global-step `.wait()` 同步修复 | reconstruction/route/source pin 对本任务直接有用 |
| 新版非即时收益 | RLT TD3-ManiSkill、RTC、Evo1/RoboCasa、π0.5 专属路线等 | 只作未来能力，不拿来证明当前 π0 baseline 必须升级 |
| 新版环境代价 | official stack 已前移到 Torch 2.11 / CUDA 12.8 方向 | 对 H100 更贴近当前栈，但也是必须实测的安装/兼容破坏面 |

因此迁移成本当前评为：PPO **零算法移植**；GRPO **小**；RLT **中等、低文本冲突但高语义验证要求**。
若保留旧 OpenPI adapter，核心 patch 可以作为实现草稿；若采用 current `openpi_rlinf` wrapper，adapter 必须
按新 public contract 重写，不能把 clean apply 当成完成证据。

## 1. 当前官方 RoboTwin π0 PPO recipe

### 1.1 官方启动入口

官方文档给出的训练入口是：

```bash
export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/absolute/path/to/RoboTwin-RLinf_support
bash examples/embodiment/run_embodiment.sh \
  robotwin_adjust_bottle_ppo_openpi
```

`run_embodiment.sh` 会：

1. 设置 `EMBODIED_PATH`、`REPO_PATH`、`MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`、`ROBOTWIN_PATH` 和 `PYTHONPATH`；
2. 选择 config name；
3. 创建 `${REPO_PATH}/logs/<timestamp>-<config>`；
4. 组装并执行：

   ```text
   python examples/embodiment/train_embodied_agent.py
     --config-path examples/embodiment/config/
     --config-name robotwin_adjust_bottle_ppo_openpi
     runner.logger.log_path=<absolute log dir>
   ```

5. 将 exact command 与 stdout/stderr 写入 `run_embodiment.log`。

环境变量 `STEPS`、`SAVE_INTER`、`NODES` 可作为 launcher overrides；smoke 是否使用它们，要在 resolved-config packet 中显式展示。

### 1.2 当前 checked-in PPO config

`examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml` 的关键值：

| 区域 | 当前值 | 解释 |
|---|---|---|
| defaults | RoboTwin adjust_bottle + `model/pi0` + FSDP + patch syncer | 精确 π0，不是 π0.5 |
| cluster | `num_nodes=1`；actor/env/rollout=`0-7` | 正好对应单机 8 GPU |
| runner | `max_epochs=1000`、`max_steps=-1` | 完整默认预算很大 |
| eval/save | 每 10 epoch | smoke 会只改预算，不改语义 |
| algorithm | GAE + actor-critic；group 1；update epoch 2 | PPO 主体 |
| reward/logprob | chunk-level | 不能与 primitive count 混为一谈 |
| entropy | token-level；bonus 0 | 记录但不迁 GRPO 设置 |
| clip/value | ratio .2/.2、value clip .2、Huber 10 | 当前官方参数 |
| discount | `gamma=.99`、`gae_lambda=.95` | 当前官方参数 |
| train env | 256 env、rollout epoch 4、episode/max rollout 200 | 需单独换算 interaction budget |
| eval env | 128 env、fixed reset IDs、episode 200 | fixed eval 主体 |
| actor batch | micro 32、global 2048 | 8-GPU默认 |
| actor model | SFT path；action chunk 50；action dim 14；value head on | RoboTwin π0 合同 |
| OpenPI | `pi0_aloha_robotwin`、3 images、detach critic input | model adapter 合同 |
| offload | env/rollout/actor 均 enabled | 先保持官方值，按深圳实测判断 |
| optimizer | actor LR `5.6e-6`；value LR `1.1e-4` | 不能从旧 run 覆盖 |

当前 YAML 和个人旧 RLT 终态中的同名官方 PPO YAML blob 相同；这说明 baseline config 本身相对稳定。变化主要发生在 launcher/runner、RLT/OpenPI 和 data/worker 接口，因此 clean PPO 应先原样验证。

### 1.3 训练调用链

```text
run_embodiment.sh
  └─ train_embodied_agent.py
       ├─ Hydra compose + validate_cfg
       ├─ Cluster(cfg.cluster)
       ├─ HybridComponentPlacement
       ├─ actor worker
       │    └─ default PPO: EmbodiedFSDPActor
       ├─ rollout worker
       │    └─ MultiStepRolloutWorker
       ├─ env worker
       │    └─ EnvWorker -> RoboTwinEnv
       └─ EmbodiedRunner
            ├─ init_workers
            └─ rollout → reward → GAE → actor/value update
                 → periodic eval/checkpoint
```

`train_embodied_agent.py` 现在会根据 `algorithm.loss_type` 分派 PPO、SAC、RLT AC、RLT TD3、DAgger、NFT 等 worker。旧代码直接覆盖该入口很危险；新 RLT 应使用官方分派扩展点。

### 1.4 官方 eval 路线

config：`evaluations/robotwin/robotwin_adjust_bottle_openpi_eval.yaml`。

候选入口（尚未执行）：

```bash
export ROBOT_PLATFORM=ALOHA
export ROBOTWIN_PATH=/absolute/path/to/RoboTwin-RLinf_support
bash evaluations/run_eval.sh \
  robotwin robotwin_adjust_bottle_openpi_eval \
  rollout.model.model_path=/absolute/path/to/SFT
```

官方 eval config 当前为 env/rollout GPU `0-7`、128 fixed-seed env、action chunk 50、14D、3 images。官方 eval guide 又说明 `adjust_bottle` 有 150 个 success seeds，默认 128 env×1 fixed epoch 只覆盖约 128 条。因此 engineering baseline 可先保留 checked-in default；formal 报告必须显式写清 128 子集、全 150 或动态轮转，不能把不同分母的成功率直接比较。`run_eval.sh` 默认 simulator OpenGL 是 `osmesa`，训练 launcher 默认 `egl`；深圳 packet 必须显示最后实际值并做 Vulkan/EGL/OSMesa 探针，不能把两条链的 renderer 混写。

### 1.5 必须保存的 baseline provenance

- RLinf / RoboTwin / model exact revision；
- installer command 和生成的 lock/package versions；
- driver/CUDA/Torch/Python/Ray/SAPIEN/Vulkan；
- env vars 与完整 resolved YAML；
- official config → SZ overlay 的精确 diff；
- fixed seed 文件 SHA；
- episode、primitive step、policy query/chunk、trajectory sample、optimizer update；
- output path、checkpoint tree、exit code、资源监控。

### 1.6 当前官方 installer 的已核实边界

候选 SHA `89b0cd5...` 的 `requirements/install.sh` 不是一条可直接交给普通账号盲跑的无副作用命令：

| 现场 | 源码事实 | 深圳决策 |
|---|---|---|
| OpenPI runtime | `openpi + robotwin` 安装 `rlinf-openpi==0.1.1`，再应用 RLinf 的 OpenPI runtime pins | runtime provenance 记录 package/version/lock；OpenPI main SHA仅为背景 |
| 系统依赖 | 不带 `--no-root` 会调用 `sys_deps.sh`；其开头要求 passwordless sudo，随后 apt 安装并可能写 EGL/Vulkan ICD | `chenyiteng` 先做只读 package/render preflight；已齐则 `--no-root`，缺失才提交 `toom` 管理员动作清单 |
| venv | Python minor 或 system-site-packages 模式不符时直接 `rm -rf "$VENV_DIR"` 后重建 | 只使用已核实不存在的新 env 路径；禁止复用/指向共享 env |
| Git 源依赖 | PyTorch3D 固定 `v0.7.9`；Curobo URL 没有 commit pin | 安装后保存 VCS `direct_url.json`/commit 与完整 freeze，必要时在 source-lock 批次再窄化 pin |
| mirror | `--use-mirror` 写 `git config --global ... insteadOf`；后续 NVIDIA Torch rewrite 会把 cleanup trap 替换为 pyproject restore trap | 直连优先；确需 mirror 时设置任务专属 `GIT_CONFIG_GLOBAL`，结束后核验并保留日志 |

这不是偏离官方路线，而是把官方命令放入深圳的权限、网络和可复现性边界。任何系统包安装或 material download 仍需另行展示影响并获批。

## 2. 当前官方 RLT 结构

### 2.1 Stage 1

官方当前例子：

- `examples/sft/config/maniskill_rlt_stage1_sft_openpi_pi05.yaml`
- `examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml`

当前 tree 另外已有 `examples/sft/config/robotwin_sft_openpi_rlinf.yaml` 和 `pi0_aloha_robotwin` dataconfig：它们给出 RoboTwin 精确 π0、14D action、50-step chunk、3 images、norm stats/checkpoint 的官方 SFT 骨架。它还没有 RLT `use_rlt`/reconstruction 组合，故不是现成 RoboTwin RLT Stage 1 recipe，但说明不需要从零重建 π0/RoboTwin SFT 数据入口。

语义：VLA 表征进入 RLT encoder/decoder；训练 RL token reconstruction，可选同时做 VLA SFT；输出完整 Stage 1 actor checkpoint，Stage 2 需要其中的 RLT token module，不能只拿裸 safetensors。

当前 SFT wrapper 的目标是 `rlt_loss + rlt_alpha * vla_loss`，官方 RLT 示例取 `rlt_alpha=1.0`。旧 RoboTwin Stage 1 则明确 `rlt_train_vla=False`、`rlt_alpha=0`，冻结 base π0；把 alpha 设为 0 并不自动等价于从 optimizer/FSDP 中真正冻结参数。这是算法语义选择，不能在接口迁移中顺手决定。

当前实现已在 2026-08-10 修复 reconstruction 逻辑（`8587bac4`），同时补 unit test。旧分支的 token reconstruction 代码不应覆盖这一版。

### 2.2 Stage 2

官方当前例子：

- `maniskill_rlt_stage2_ac_mlp.yaml`
- `maniskill_rlt_stage2_td3_mlp.yaml`
- `realworld_rlt_stage2_ac_mlp.yaml`

数据状态：

```text
curr_obs = {z_rl, proprio, ref_chunk}
action   = 实际送给环境执行的 action chunk
next_obs = {next_z_rl, next_proprio, next_ref_chunk}
```

Stage 2 冻结 feature model；rollout 侧生成 `z_rl` 和 reference chunk；小 actor/critic 从 replay 更新。ManiSkill 例子有 automatic critical-phase switch 和可选 expert intervention；real-world 例子有 keyboard switch。这些 switch 是环境专属，不能原样搬到 RoboTwin。

### 2.3 当前官方数据与 worker 接口

新版把 embodied I/O 收敛到：

- `rlinf/data/schema/embodied_types.py`
- `rlinf/data/schema/embodied_trajectory_builder.py`
- `rlinf/data/storage/replay/`

文档公开的主要类型包括 `EnvOutput`、`PolicyOutput`、`ChunkStepResult`、`EmbodiedTrajectoryBuilder` 和 `Trajectory`，并已容纳 `rlt_switch_flags` 等 RLT 字段。旧分支曾自己补 route/transition/trajectory 字段；迁移时先证明新版缺什么，不能复制旧 schema。

### 2.4 OpenPI 模型层变化

当前 upstream 同时存在：

- `rlinf/models/embodiment/openpi/`：官方 π0 PPO baseline 使用；
- `rlinf/models/embodiment/openpi_rlinf/`：JAX-aligned PyTorch 路线；当前 RLT 可执行例子以 π0.5 为主，但 official RoboTwin SFT 已有精确 π0 config；
- `rlinf/models/embodiment/openpi_pytorch/` 等其他实现。

因此最大的未闭合技术问题不是“有没有 RLT worker”或“有没有 π0/RoboTwin SFT 入口”，而是**怎样把这个现成 SFT 骨架接到当前 RLT reconstruction/feature/replay 基础设施**。先实测现有 π0 config/checkpoint 是否暴露 RLT 所需 embedding/z/ref interface；缺什么只加窄 adapter。不能为省事把目标改成 π0.5，也不能回滚 `openpi_rlinf` 整包。

## 3. 旧个人 RLT 的可迁移合同

旧 AutoDL 实现可作为规格的部分：

| 合同 | 旧值/语义 | 新版动作 |
|---|---|---|
| task | RoboTwin `adjust_bottle` | 保留 |
| VLA | 精确 π0 | 保留，先核新 model wrapper |
| Stage 1 objective | 旧：冻结 π0、token-only；新 official：joint VLA+RLT | 建两个有名字的语义，不用 `alpha=0` 假装冻结；默认推荐先跑 current-official baseline，待用户确认 |
| demonstration | historical clean-50 | 用新官方 processed dataset 候选重新锁 revision/schema |
| normalization | Stage 1/2/reference/student 共用同一 `norm_stats.json` | 保留并加 manifest/hash |
| prefix | image-only masked prefix；历史 `[B,768,2048]` | 以当前模型真实张量重新验证，不硬编码为事实 |
| `z_rl` | 2048 | 与当前 RLT config/model 对齐后冻结 |
| VLA horizon | H=50 | 保留候选 |
| execution chunk | C=10 | 保留候选 |
| state/action | canonical `[10,14]` | 保留 |
| actor input | `z_rl + proprio + ref_chunk` | 与 current upstream 一致，可复用 |
| critic input | `z_rl + proprio + action_chunk` | 与 current upstream 一致，可复用 |
| replay action | 实际 executed action | 与 current upstream 一致，必须验收 |
| switch | full-task reference warmup 后 student | RoboTwin 专属逻辑需重做；不搬 ManiSkill gate |
| eval | deterministic student mean | 保留 |
| resume | replay/schedule/model/optimizer + first-action initial sync | 保留合同，按新 checkpoint API 重写 |

这些值是旧实现合同，不是当前 upstream 已验证事实。Stage 1 data inspection 和真实 tensor probe 后才能写进 resolved config。

特别是旧 Stage 1 YAML 顶层 `num_action_chunks=10`、nested OpenPI `action_horizon=50`；current `openpi_rlinf` 的 RoboTwin SFT config 顶层是 50，并由该值构造模型 horizon。新 Stage 1 顶层必须先按 **H=50** 解释；Stage 2/环境每次执行 **C=10** 是另一个维度，不能机械抄旧顶层 H10。

## 4. 复用 / 重写 / 废弃矩阵

### 4.1 直接保留 upstream

| 模块 | 决策 | 原因 |
|---|---|---|
| `rlt_token_transformer.py` | 保留新版 | 已有 2026-08 reconstruction fix 和 unit test |
| `rlinf/algorithms/rlt/*` | 保留新版主体 | route/transition 已随新 schema 修订 |
| RLT AC/TD3 worker | 保留新版 | 新 runner 已正式分派；含后续功能/修复 |
| embodied schema/storage | 保留新版 | 2026-08 data refactor，旧路径已删除/迁移 |
| PPO baseline config/worker | 原样先跑 | 同名 YAML 稳定，当前目标有官方 recipe |
| current checkpoint/weight sync framework | 保留新版 | 旧 resume 合同在新 API 上实现 |

### 4.2 选择性适配

| 模块 | 需要做什么 | 首要验收 |
|---|---|---|
| π0 → RLT feature path | 从 official RoboTwin π0 SFT 骨架接入当前 RLT embedding/z/ref interface | prefix shape、dtype、mask、frozen params |
| RoboTwin Stage 1 data | official `pi0_aloha_robotwin` dataconfig + clean-50 processed data + norm stats | image/state/action/task schema一致 |
| RoboTwin Stage 1 config | 在 `robotwin_sft_openpi_rlinf.yaml` 上按 current RLT example 增加 opt-in 字段 | compose、only RLT trainable、完整 actor checkpoint load |
| RoboTwin Stage 2 env route | 把 env info/switch/executed action 映射到 current route/schema | transition 的 curr/action/next 对齐 |
| RoboTwin Stage 2 config | 14D、H50/C10、feature model、replay/schedule | resolved config 与合同表一致 |
| full-task route | current 只按 simulator capability 选 simulator 或 real-world route；补 RoboTwin warmup→student 全任务语义 | route truth table、action source、record flag |
| transition replay capability | current 只对 `MANISKILL_RLT` 返回 true；加显式 RoboTwin opt-in，不复制旧 API | 每 env step 一条 curr/action/next，executed action正确 |
| canonical action adapter | 在 current `openpi_rlinf` output transform 周围重建 RoboTwin normalized/delta canonical route/decode | reference/student parity、14D env action |
| truncation bootstrap | 先写 current-schema 回归测试，再决定是否回植旧 time-limit 语义 | truncated 非 terminal、linked next_obs正确 |
| deterministic eval | fixed seeds + student mean + 无 train-only intervention | 分母、seed、action source 可追溯 |
| resume/initial sync | 映射到新版 checkpoint/runtime | restored update、rollout version、first action |

### 4.3 只迁移 tests/fixtures 的意图，不迁旧实现

- normalization 单一真值；
- H/C/action dimension 合同；
- executed-action replay；
- next feature/reference 对齐；
- deterministic eval；
- checkpoint/replay/schedule/resume/initial sync；
- legacy PPO compose 不受 RLT opt-in 影响。

测试要针对 current public symbols 重写。旧 `tests/unit_tests/test_robotwin_rlt_contract.py` 在新 official tree 中不存在，不能机械复制 import/path。

### 4.4 明确废弃

- 旧 `openpi_action_model.py` 整文件覆盖；
- 旧 `fsdp_rlt_ac_policy_worker.py` 766 行补丁整块覆盖；
- 旧 data/trajectory/replay 路径；
- 旧 RoboTwin RLT YAML 的 A800/2GPU/smoke/8env 资源值；
- AutoDL cache/env/network/root 路径；
- 旧 reconstruction 实现；
- 旧 branch 中 DSRL/Fast-WAM 前史；
- 旧结果阈值作为新 smoke gate。

旧 compact replay 与 robust resume/manifest 不是判错后整块迁移的对象：fresh smoke 首批后置；只有正式长跑要求完整续训，或深圳现场证明 replay payload 导致内存问题时，才依据 current schema 分别重做。

## 5. 为什么不能直接 merge/cherry-pick 旧分支

从 `2b8199d8` 到 `89b0cd5` 的 relevant name-status 中可见：

- 旧 `robotwin_*_rlt*.yaml`、旧 RLT toolkit/test 在 official main 侧不存在；
- official main 已修改 `train_embodied_agent.py`、`embodied_runner.py`、`env_worker.py`、`huggingface_worker.py`；
- official main 已修改全部 RLT route/transition/worker/token transformer；
- official main 新增 `openpi_rlinf/`、new schema/storage 和 TD3 worker；
- old/new 对同一文件均有语义修改，整提交 cherry-pick 会产生假冲突解决：代码可能能合并，却退回上游修复。

正确方法是：在新 branch 上先写 current-source map，然后按本矩阵以 feature/config/test 为单位实现一个连贯批次。

## 6. 最小高信息量验收序列

### 6.1 clean PPO

1. YAML parse + Hydra `--cfg job --resolve`；
2. imports/versions/CUDA/Ray；
3. RoboTwin renderer/import；
4. π0 checkpoint + norm stats load；
5. single env reset/observation/action；
6. fixed SFT eval；
7. one-update PPO + eval + checkpoint + reload；
8. legacy defaults/diff check。

### 6.2 RLT port

1. current Stage 1/2 official configs compose；
2. π0 feature wrapper fixed-tensor probe；
3. RoboTwin dataset/dataconfig sample probe；
4. Stage 1 forward/backward: 按用户选定的 joint 或 frozen/token-only 语义核精确 trainable set；
5. Stage 1 checkpoint → Stage 2 load；
6. one route step: ref/student switch + executed action；
7. transition/replay curr-action-next contract；
8. actor/critic update and weight sync；
9. checkpoint/resume + first action version；
10. legacy π0 PPO config compose unchanged。

这些 checks 不替代真实 simulator smoke。任何 model load、simulator、smoke 或训练仍按主计划审批边界执行。

## 7. 当前开放风险

| 风险 | 当前证据 | 处理 |
|---|---|---|
| RLT official model path偏 π0.5 | upstream #1435 与 configs 文件名 | 严格核 π0 support；不静默换模型 |
| Stage 1 目标语义分叉 | current 示例 joint VLA+RLT；旧实现 frozen/token-only | 用户确认；两个 config/结果必须显式命名，不互称复现 |
| upstream 仍快速更新 | 8 月连续 reconstruction/data/OpenPI/TD3 变更；HF 又新增数据/模型 | 每个实施批次固定 SHA；开始前只审差分一次 |
| HF server route受阻 | 深圳 official HF timeout，mirror仅首页验证 | 真实分片 probe + server-side bounded route search |
| renderer/system deps | 裸机与 Docker image不同 | UV installer audit、Vulkan/EGL check；缺包再列 admin action |
| official full PPO budget大 | 256 env、8 GPU、1000 epochs | 先 smoke，实测吞吐后审批 pilot/formal |
| 旧 RAM failure被误迁移 | AutoDL 240 GiB cgroup；深圳约 2 TiB | 仍监控 EnvWorker RSS，但不预设同一故障 |

## 8. 实施时的 diff 纪律

- baseline branch 只允许 source-lock、SZ path/resource overlay、运行/证据脚本；不改算法。
- RLT branch 的每项改动标注：upstream file/symbol、旧合同来源、为何 current upstream 仍缺、对应 test。
- 任何 collision 先保留 upstream，再重写最小 adapter；不以“旧实验跑过”为覆盖理由。
- config 默认 opt-in；`robotwin_adjust_bottle_ppo_openpi` 和其他 official recipes 必须保持可 compose。
- 完成一个连贯批次后才 server-check；不维护并行 fallback runtime。
