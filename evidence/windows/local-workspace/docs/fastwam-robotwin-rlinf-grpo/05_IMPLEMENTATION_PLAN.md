# Fast-WAM + RoboTwin + RLinf + GRPO 实施计划

更新时间：2026-07-18

> 2026-07-18 执行口径更新：实现已完成到 18 个目标文件；canonical smoke/formal 参数、单命令 launcher 与同步资源监控以 `evidence/IMPLEMENTATION_LOG_20260717.md` 第8节为准。本文中“必须先lr=0再非零”的旧P2描述降级为可选ratio诊断。

本文是唯一实施过程文档。前半部分维护阶段顺序，后半部分维护逐层、逐文件实现台账。总目标和当前共识见 [00_INDEX.md](00_INDEX.md)，张量语义以 [02_INTERFACE_CONTRACTS.md](02_INTERFACE_CONTRACTS.md) 为准，代码落点以 [03_CODE_AND_DIFF_MAP.md](03_CODE_AND_DIFF_MAP.md) 为准，最小通过门以 [06_TEST_ACCEPTANCE_MATRIX.md](06_TEST_ACCEPTANCE_MATRIX.md) 为准。

## 1. 实施原则

- 当前 π0 工作树、环境和 checkpoint 不原地修改。
- 服务器默认只读；每次写入、安装或训练前取得用户明确授权。
- 先连续完成一个职责连贯、相互依赖的主体实现批次，再集中做少量高信息量检查；不为每个小函数设置独立流程 gate，也不把备份、依赖大改、编译和训练塞进一条长命令。
- 只做能拦住实质错误的必要检查。失败时保留最小证据并修正根因，不预先维护多套复杂 fallback。
- 小型证据写入 `evidence/`；动态实验状态写入权威交接。
- 每个目标 symbol 都必须能追溯到锁定的 repo@commit/path::symbol；与来源不同的地方只能标成 RoboTwin 适配、RLinf 兼容性修复或明确实验变量。
- “最小 RLinf 核心 diff”只表示复用标准 runner/worker/loss/sync/DCP，不表示缩短 Fast-WAM 模型、另写简化 ODE/SDE、退回 singleton、关闭同步或删掉有效 trainable 模块。

## 2. 阶段 0：锁定官方 standalone

状态：**已完成**。用户于 2026-07-17 确认官方 Fast-WAM RoboTwin 推理已跑通，因此迁移主线可以进入阶段 1/2 的只读设计和准备。此前的 Hugging Face 401 已由恢复官方 ModelScope 组件源解决，不再是当前阻塞项。具体安装和 standalone 命令只维护在 [07_OFFICIAL_STANDALONE_RUNBOOK.md](07_OFFICIAL_STANDALONE_RUNBOOK.md)，本计划不重复。

进入服务器实施前仍要从本次成功 run 补采以下小型只读证据；这不是重跑安装：

1. Fast-WAM commit。
2. checkpoint 和 dataset stats 标识。
3. resolved config：服务器成功 run 已确认 action horizon `H=32`、replan `N=24`、denoise `S=10`、scheduler infer shift=5.0。
4. 一个黄金 fixture：三张图、14D qpos、task text、normalized/physical action。
5. 首个真实 RLinf runner 的资源峰值。

通过标准：官方入口能重复完成至少一个 RoboTwin episode，而不只是 import 或 model load。该标准已满足；2026-07-17 20:05 已现场确认官方 commit、checkpoint/stats、成功日志、视频与 resolved `H/N/S/shift`。实施前只剩小型黄金 fixture；资源峰值留在 P2 真实 runner 记录，不为此重跑 standalone。

## 3. 阶段 1：隔离代码和环境

取得服务器写入授权后：

1. 只读确认 RLinf `HEAD/status/origin`、当前 venv 版本和磁盘空间。
2. 保存当前 π0 venv 的 `pip freeze`、editable 路径、`pip check`、Python/Torch/CUDA/Ray/SAPIEN/MPLib/CuRobo 等关键版本，只作为联合环境的只读参考清单，不把该 venv 复制成待运行环境。
3. 用 `rsync -aH` 制作 π0 golden venv 完整备份，并用一次 dry-run 确认一致。该副本只用于恢复到原规范路径，不在新路径直接运行，也不在备份副本上安装 Fast-WAM。
4. 从已验证 RLinf commit 创建 `feat/fastwam-robotwin-grpo` 独立 worktree，并先锁定最终 venv 路径，避免 Python shebang/editable path 搬迁。
5. 在最终路径新建 Python 3.11 venv，先固定最终 Torch `2.7.1+cu128`/Torchvision `0.22.1+cu128`；后续所有安装都带同一 constraints，不能让 RLinf 默认 dependency group 把 Torch 重新覆盖到 2.6。
6. 以 π0 清单为基线安装 RLinf/RoboTwin 通用依赖，再补官方 Fast-WAM 差异依赖。联合环境明确固定 `transformers==4.49.0`、`hydra-core==1.3.2`、`omegaconf==2.3.0`、`numpy==1.26.4`、`torchcodec==0.5`、`setuptools==80.9.0`、`warp-lang==1.11.1`、CuRobo `v0.7.8` 和 `CUDA_HOME=/usr/local/cuda-12.8`。所有 CuRobo/CUDA extension 必须在最终 Torch 后安装或重编，不能复制旧 Torch ABI 产物。
7. editable 安装锁定 commit 的官方 Fast-WAM 和独立 RLinf worktree；Fast-WAM adapter 直接位于该 RLinf worktree 内，不再安装第三个外置扩展包。安装前后分别保存 freeze/check diff。
8. 依次检查 Torch/CUDA、RLinf compose、RoboTwin import/render 与 qpos env smoke、Fast-WAM checkpoint load 和官方黄金 fixture。

这个顺序有两类依据：Motus/LaWAM 证明了变更前清单和完整备份的价值，也实际证明修改共享 π0 venv 会分别造成 Transformers 冲突和 OmegaConf/Hydra 破坏；本计划保留清单/备份经验，但拒绝共享环境原地修改。社区 Fast-WAM×RLinf 的成功 run 使用 Python 3.11、Torch 2.7.1+cu128，并 editable 安装 Fast-WAM + RLinf；社区仓库默认依赖仍可能声明 Torch 2.6，因此只能把实际 run 环境作为兼容性证据，不能无审计安装整组默认依赖。

必要产物：

- 环境清单和安装 diff。
- π0 venv 备份路径及一致性结果。
- Fast-WAM×RLinf 联合 venv 的关键包版本。

通过标准：独立环境能同时加载 RLinf 和 Fast-WAM，并重现官方黄金 action。

## 4. 阶段 2：Deterministic eval 接入

实现 RLinf 内置 `rlinf/models/embodiment/fastwam/`：

1. builder：加载官方 model/checkpoint/stats/config；把重复 alias 收敛成唯一 `mot.*` 注册树。
2. RoboTwin adapter：三相机、14D qpos、prompt、norm/denorm。
3. `FastWAMPolicy(nn.Module, BasePolicy)` 显式实现`forward()`，并调用唯一`flow_sde_rollout(..., deterministic=True)`的batch inference；官方singleton `infer_action()`仅作测试oracle。
4. 按 RLinf 内置模型惯例在 `rlinf/models/__init__.py` 延迟导入 `get_model()` 并注册唯一 `fastwam_robotwin`；启动仍用官方 `train_embodied_agent.py`。
5. 当前服务器 pin 已确认 worker 不向通用 embodied policy 传 train/eval mode，因此加入默认关闭、由模型类属性显式启用的 capability shim；只有实施时目标代码已存在等价通用机制才省略。

必要检查：

- B=1 wrapper action 与官方 fixture 对齐。
- B>1 batched输出与逐样本oracle在记录的BF16/kernel容差内对齐。
- 同任务/seed 下 RLinf eval 与 standalone 无系统偏差。
- canonicalization 前后官方 action 对齐；`state_dict`/optimizer/sync 名称无重复逻辑参数。
- π0 默认 model type 和 YAML 行为不变。

此阶段不开 optimizer。

## 5. 阶段 3：Flow-SDE 与 actor replay

1. 不新增第二套 train denoise loop：在阶段 2 已验过的同一个 `flow_sde_rollout(..., deterministic=...)` 中开启 train 参数分支。
2. `deterministic=True`时保留官方model-dtype state并调用官方scheduler step，按记录容差与官方ODE parity；`False`仍走同一scheduler/conditioning/velocity循环，只在选中denoise step注入FP32 Flow-SDE transition。
3. 第一版按社区与 π0 的已验证方式保存完整 action chain、共享 denoise index、behavior old logprob 和 actor 重算输入；不保存 video chain/KV/debug。
4. actor 对同一 transition 重算 new logprob。
5. 保存 normalized proprio 和未拼 proprio 的 text context；第一版 proprio 冻结，但 replay schema 不把其输出永久 detach，便于后续独立实验。
6. 固定 `H=32`；`N` 取 standalone resolved 值。全 H 采样，只对实际执行的前 N 计算 old/new logprob 与 loss。
7. 断言 scheduler 的 resolved shift/timestep/delta 与 rollout/replay 完全一致；不静默忽略 `sigma_shift`。
8. training batch 路径 fail-fast，不复制社区 catch-all fallback。

必要检查：

- SDE 关闭时与官方 deterministic action 一致。
- mean/std/Gaussian logprob 的小型数值测试通过。
- 同权重 rollout/actor 的 old/new logprob 对齐。
- optimizer 前 ratio 接近 1、clip fraction 接近 0。
- 只有 canonical action expert 有梯度；proprio/video/VAE/T5 无梯度且 hash 不变。
- policy 返回逐元素 `[B,N,14]` logprob，不在 policy 内提前 sum/mean。

未通过这些检查，不进入真实 GRPO 更新。

## 6. 阶段 4：同步、checkpoint 与最小 GRPO smoke

第一版：

- 冻结 proprio、video expert、VAE、T5 和 stats。
- 只训练 canonical action expert。
- 使用RLinf pin现有FSDP2：禁用block/expert wrap、最后shard root，同时保留非tied Embedding的既有单独wrap；该FSDP2路径不使用`use_orig_params`。
- 保留 RLinf 常规初始权重同步；Fast-WAM 使用 bucket sync。社区 Fast-WAM 的默认 GPU bucket 是初始依据，但 2026-07-18 本机首次 smoke 证明当前 AutoDL 容器会在 CUDA IPC 上拒绝 `pidfd_getfd`，因此本机改用 RLinf 原生 CPU bucket。π0 的 patch YAML 不改。
- 从最低并发和最小 actor 布局开始。

顺序：

1. actor/rollout 初始参数 hash 对齐。
2. 完成一个 `lr=0` runner step。
3. 完成一个非零 lr GRPO step。
4. 保存 DCP；本阶段只确认 checkpoint 产出，不把生命周期验收塞进同一个 smoke。

必要检查：无 NaN/OOM；advantage 非退化；同步后 trainable 参数一致；冻结参数不变；DCP 完整。fresh-process resume 与 deploy export 统一归入 P3。

## 7. 阶段 5：10-step 学习验证

选择 standalone baseline 成功率约 0.3–0.8 的任务。

记录：

- step 0、5、10 的固定 seed deterministic eval。
- train success、ratio/KL/clip、grad norm。
- GPU、cgroup RAM、EnvWorker RSS 和 step time。
- 至少一个未训练任务或随机化条件，观察遗忘。

通过标准：独立 eval 出现可解释趋势，而不是只有带噪 train success 上升。

## 8. 后置工作

只有前五阶段通过后再讨论：

- 扩大 env、group 或 actor world size。
- replay 压缩和性能优化。
- proprio encoder 单独训练实验：actor 重跑 proprio，分别审计 action-context/video-context 梯度。
- video expert 联合训练。
- PPO/critic。
- RLinf 整体升级。

这些不是首版接入的前置工作。

## 9. 已冻结默认与仍需现场证据

已经形成的首版默认：

1. 不升级 RLinf；从 π0 known-good commit 派生独立 worktree，π0 原工作树和 venv 不动。
2. 官方 Fast-WAM 仓库保持独立、固定 commit；RLinf 内置 adapter只导入官方代码，不复制模型实现。
3. critic-free GRPO；只训练 canonical action expert，proprio/video/VAE/T5 全部冻结。
4. 复用现有 RLinf RoboTwin qpos env、reward、trajectory、actor loss、runner 和 DCP；不新增 Fast-WAM env 分支。
5. 内置模型 registry + 当前 pin 必需、默认关闭的 rollout-mode capability shim；不把 Fast-WAM 写进多处核心枚举或 worker 分支。
6. 单 stochastic denoise transition、真实 behavior old logprob、同一 transition actor replay；训练路径不做 catch-all fallback。
7. RLinf pin原生FSDP2：`wrap_policy.no_split_names=["__none__"]` 禁block/expert wrap，root shard并保留默认Embedding wrap；使用CPU-staged bucket sync以适配本机容器的CUDA IPC限制。同步集合由canonical action trainables和通用同步器要求的persistent buffers构成，不把“action-only训练”误写成“绝对只有action key被传输”。
8. `N=24` 的首个工程 smoke 固定使用 192-step episode/rollout，恰好执行 8 个完整 chunks；正式实验的短尾 mask 后置。
9. 模型落点固定为内置 `rlinf/models/embodiment/fastwam/`；stochastic `k` 从全部 `0..S-1` 均匀采样；首个非零探索 smoke 固定 `noise_level=0.1`。
10. 实施先完成主体实现，再集中做离线 parity、真实 runner smoke、同步/恢复/导出三组检查，不采用逐 symbol 的微型 gate 流程。
11. regular DCP 显式 `save_full_model_weights:false`；deploy export 优先复用 pin 自带的 `no_dist=True` DCP→model-state 转换器，`full_weights.pt` 只作兼容输入。

已经拍板三项的依据：

1. **模型落点**：RLinf 官方新增模型文档和 `register_model` 内置惯例是第一依据；社区 Fast-WAM、Motus、LaWAM 是重复工程证据。本项目维护独立 RLinf worktree，因此一条标准 registry diff 比外置 launcher/跨进程扩展生命周期更贴近基座。外置方案不并行维护。
2. **stochastic k 范围**：RLinf pin 的 OpenPI/π0 GRPO 概率实现是第一依据；Fast-WAM 只适配其 shifted/signed schedule。首步用下一 schedule 点作分母后 std 有限，所以没有依据再删首/末 step。若真实 resolved schedule 违反这些前提，才按现场证据调整。
3. **探索强度**：`noise_level=0.1` 不是宣称 Fast-WAM 官方最优值，而是首个非零更新的保守稳定性/ratio 探针；通过后是否比较 `0.1/0.3` 另定。

仍需现场证据或后续实验决定：

1. **replay 图像表示**：当前接口附录使用 BF16 `[-1,1]` image，最直接；也可以保存组合后的 uint8 image 以减半 CPU transport。首版先保持 BF16 parity，资源实测后再决定是否改 schema。
2. **T5 驻留**：首版 rollout 保存未拼 proprio 的 text context，actor 不重跑 T5。同一 registry builder 首版仍加载官方完整 deploy 组件，是否为 actor 独立省去 T5，等双 A800 实测后再做有证据的优化。
3. **首个学习任务**：阶段 2 固定用已经跑通 standalone 且有 π0 基线的 `adjust_bottle` 做接口 parity；阶段 5 再从 Fast-WAM baseline 约 0.3–0.8 的任务中选学习任务。

其余三项工程选择已经冻结：本机CPU bucket（bucket算法不变，只替换传输设备）；`eval_seed=0`仅用于eval，按官方生成一个singleton latent后沿B broadcast，train继续为每条trajectory独立生成随机量；P2采用第23.3节资源表。

`H/N/S=32/24/10` 与 infer shift=5.0 已从成功 standalone 的 resolved config只读确认；训练 YAML必须逐项断言，不再作为可自由覆盖的候选值。

## 10. 实现方法：两端对齐，不单向盲写

推荐顺序不是只从上往下，也不是只从下往上：

1. **先自上而下冻结合同**：入口、Hydra 配置、driver/Worker 注册、`BasePolicy` 返回、trajectory、actor 调用和 GRPO loss 的边界先写清楚。
2. **再自下而上编码**：先写相机/state/action 纯函数和 Flow-SDE 数学，再写 builder/canonicalization 和 policy，最后接 worker/runner。
3. **最后自上而下集中验收**：主体代码完整后，从启动命令贯通一次 `lr=0` step，再做一次非零更新；只在出现实质错误时停下，不同时启用多套 fallback。

这样做的原因是：RLinf 的外层调用链已经由 π0 跑通，Fast-WAM 的底层数值语义也由官方和社区实现给出；真正需要新写的是两者之间的 batch 编排、adapter、概率 replay 和模型树规范化。

## 11. 自上而下的目标调用链

```text
命令行
  python examples/embodiment/train_embodied_agent.py \
    --config-name robotwin_adjust_bottle_grpo_fastwam_a800_2gpu_smoke
    ↓
RLinf validate_cfg → Cluster → HybridComponentPlacement
    ↓
EmbodiedFSDPActor + MultiStepRolloutWorker + EnvWorker
    ↓
rlinf.models registry → fastwam.get_model() → builder.build_fastwam_policy(...)
    ↓
FastWAMPolicy.predict_action_batch(env_obs, mode)
  → fastwam_rl.flow_sde_rollout(deterministic = mode != "train")
    eval  → 同一 denoise loop 关闭随机 transition，不保存 replay
    train → 同一 loop 的选中 k 注入 Flow-SDE，保存 behavior old logprob + tensor replay
    ↓
RLinf RolloutResult → 现有 RoboTwin chunk_step → Trajectory stack/flatten
    ↓
FastWAMPolicy.forward → default_forward → 同一 transition new logprob
    ↓
现有 EmbodiedFSDPActor → GRPO advantage/loss → optimizer
    ↓
现有 bucket sync / DCP；需要部署时单独 export 官方 Fast-WAM payload
```

明确不新增 Fast-WAM 分支的层：`EnvWorker`、RoboTwin env/reward、trajectory、advantage registry、通用 GRPO/PPO loss、actor training loop、runner、通用 DCP manager。

## 12. 目标文件与第一版符号

模型适配建议位于独立 RLinf worktree 的内置目录：

```text
rlinf/models/embodiment/fastwam/
├── __init__.py
├── builder.py
├── robotwin_adapter.py
├── fastwam_rl.py
├── fastwam_policy.py
└── export.py

tests/models/embodiment/fastwam/
    ├── test_adapter_builder_parity.py
    ├── test_flow_sde_replay.py
    └── test_checkpoint_export.py
```

RLinf 功能分支还新增两个 YAML、一个内置注册；只有现场确认必要时才修改 worker：

```text
rlinf/models/__init__.py
examples/embodiment/config/model/fastwam_robotwin.yaml
examples/embodiment/config/robotwin_adjust_bottle_grpo_fastwam_a800_2gpu_smoke.yaml
rlinf/workers/rollout/hf/huggingface_worker.py  # 可选、默认关闭的 capability shim
```

| 文件 | 第一版符号/职责 | 直接依据 | 完成定义 |
|---|---|---|---|
| `rlinf/models/__init__.py` | `_build_fastwam()` + `register_model("fastwam_robotwin", ...)` | RLinf `6d0db56::_register_builtin_models`、`new_model_fsdp.rst` | driver/Worker 走同一内置 registry；官方训练入口不变 |
| `fastwam/__init__.py` | `get_model(cfg, torch_dtype)`；延迟调用 builder | RLinf 内置 embodied model 惯例 | 不导入 checkpoint/模型大对象到 registry import 阶段 |
| `builder.py` | `build_fastwam_policy()`、`compose_official_robotwin_cfg()`、`canonicalize_fastwam_module_tree()`、`freeze_action_only()`、`assert_model_inventory()` | 官方 `runtime.create_fastwam/FastWAM.load_checkpoint`；社区 `fastwam/__init__.py::get_model`；Motus 暴露模型给 FSDP 的模式 | 官方 checkpoint/stats 加载；唯一注册树；只有 action subtree 可训练；冻结 conditioner 维持 eval；固定输入 action 不变 |
| `robotwin_adapter.py` | `validate_env_obs()`、`compose_three_camera_image()`、`normalize_proprio()`、`denormalize_actions()`、`build_prompts()` | 官方 `deploy_policy.py::_resize_rgb/_build_robotwin_image_tensor/_normalize_state/_denormalize_action/_infer_action_chunk`；RLinf pin RoboTwin obs | B=1/B>1 精确 PIL 三相机、14D absolute qpos；无 wrist/prompt/state fallback或额外动作转换 |
| `fastwam_rl.py` | `flow_step_mean_std()`、`build_action_conditioning()`、`predict_action_velocity()`、`flow_sde_rollout(deterministic=...)`、`recompute_logprob()` | 社区 `fd780f0` 同名文件；官方 continuous scheduler/`infer_action`；π0/Motus model-side replay | 只有一套 ODE/SDE loop；deterministic parity、有限性、rollout/replay old-new parity 全通过 |
| `fastwam_policy.py` | `FastWAMPolicy`；显式 `forward()`、`train()`、`predict_action_batch()`、`default_forward()`；`rlinf_accepts_rollout_mode=True` | RLinf `BasePolicy/OpenPI`；社区同名 policy；Motus 后期 wrapper | eval/train 都调同一 RL core；冻结 conditioner mode不漂移；eval不留replay；train返回标准字段；actor返回逐元素logprob |
| `export.py` | `load_rlinf_model_state()`、`extract_official_fastwam_payload()`、`export_deploy_checkpoint()`、`main()`：pin原生 no-dist DCP/full state → 官方 payload | RLinf `convert_dcp_to_pt.py`；官方 `save_checkpoint/load_checkpoint` | exact schema通过；全新进程官方 loader 可加载；resumed RLinf与exported official action parity |

`FastWAMPolicy(nn.Module, BasePolicy)` 必须显式重写 `forward()`。否则 Python MRO 会先命中 `nn.Module.forward`，不能依赖 `BasePolicy.forward` 自动分发。

依赖方向保持单向：`adapter` 不算概率，`fastwam_rl` 不处理 RoboTwin，policy 不实现 GRPO loss，`builder` 只负责构建/规范化，`__init__.py` 只暴露 `get_model`。模块职责可以分开，但 eval/train 的 conditioning、scheduler、velocity 和 denoise loop 只能有一套。第一版不新增通用 `fsdp.py`、`checkpoint.py` 或第二套 runner。

## 13. 连贯实现批次与集中验收

| 批次 | 连续完成什么 | 完成后集中检查 |
|---|---|---|
| I0 现场锁定 | 只读确认 server HEAD/status/origin、成功 standalone `H/N/S`、黄金 fixture、现有 RoboTwin obs/chunk 契约；不修改服务器 | 只确认来源足以编码，不扩展安装或重跑 |
| I1 模型主体 | 内置 package/registry、`robotwin_adapter.py`、`builder.py`/canonicalization、`fastwam_rl.py` ODE/SDE 共核、`fastwam_policy.py` | P1：一个组合 fixture 覆盖官方 B=1、真实 B>1、模块树/trainables、schedule/old-new parity |
| I2 RLinf 接线 | 当前pin必需的capability shim、model/smoke YAML、FSDP/sync配置与DCP产出 | P2：真实 `lr=0` runner step + 一次 `noise_level=0.1` 非零更新，连带检查实际 wrap、同步、资源和 π0 零回归 |
| I3 生命周期 | fresh-process resume、pin原生no-dist DCP抽取、官方 deploy export，必要问题修正 | P3：恢复后的RLinf模型与导出后官方模型在同一fixture上的action parity；稳定后再做10-step学习验证 |

I1 内部允许连续实现相互依赖文件，不要求 adapter、builder、deterministic、stochastic 各停一次。官方逐样本 oracle 只用于组合测试，不是训练时异常 fallback；production B>1 失败时保留证据并修正 batch core，不能捕获所有异常后悄悄退回慢路径。

## 14. 一次 train rollout 的精确数据流

1. 现有 RoboTwin env 输出 `main_images [B,H,W,3]`、`wrist_images [B,2,Hw,Ww,3]`、`states [B,14]`、`task_descriptions`。
2. adapter 固定 `wrist_images[:,0]=left`、`[:,1]=right`；resize/拼图得到 `[B,3,384,320]`，并用官方 processor 归一化 14D qpos、构造官方 prompt。
3. rollout 计算未拼 proprio 的 T5 `text_context/text_mask`，再用冻结 proprio encoder 拼 token；VAE + video expert 构建 conditioning。
4. 完整采样 normalized action latent `[B,H=32,14]`。训练时 batch 共享一个 denoise index `k`，每条 trajectory 使用独立 initial latent 和独立 Gaussian noise；只选中的一步随机，其余步骤走 deterministic ODE。
5. 保存完整 chain `[B,S+1,32,14]` 和真实 behavior old logprob `[B,N,14]`；denorm 后只把前 `N` 个 physical qpos action `[B,N,14]` 交给环境。
6. `RolloutResult.actions` 用于环境执行；`forward_inputs["action"]` 保存同一 physical action 的 `[B,N*14]` 展平形式，供 trajectory 记录；二者不能混成 normalized action。
7. actor 收到 tensor-only replay 后，重建 frozen conditioning，取每条样本真实的 `x_k/x_{k+1}`，只让 canonical action expert 参与梯度，返回 new logprob `[B,N,14]`、可选 entropy、`values=None`。
8. 现有 RLinf chunk-level GRPO 汇总逐元素 logprob、计算 ratio/clip/advantage 并更新；policy 内不提前 sum/mean，也不另写 loss。

第一版 `forward_inputs` 字段和 dtype/shape 不在此重复，统一按 [02_INTERFACE_CONTRACTS.md](02_INTERFACE_CONTRACTS.md)。关键边界是：保存未拼 proprio 的 text context + normalized proprio，不保存 prompt 字符串、video chain、KV cache、velocity 或 debug 对象。

## 15. 已收敛的数值规则与仍需证据的结构细节

1. **Flow-SDE 首步已经收敛**：Fast-WAM 官方 scheduler 负责 shifted 网格；RLinf `6d0db56` OpenPI 负责 Flow-SDE mean/std。Fast-WAM 的 signed `delta<0` 先断言，再令 `dt=-delta>0`；当 normalized `t==1` 时，sigma 分母使用下一 schedule 点。禁止复制社区 `t<=0.98`，也禁止通过 `abs(delta)`、任意 std floor 或 `simple/full` fallback 改写首版概率。
2. **schedule/latent dtype 边界**：schedule始终按官方函数以model dtype生成。`deterministic=True`保留model-dtype latent并直接调用官方`scheduler.step()`；stochastic rollout/replay的chain、mean/std/logprob保持FP32，进入action expert前显式把`x/raw_t`转model dtype，velocity返回后再upcast FP32。两者仍在同一个loop中，仅以`deterministic`参数控制这个有依据的dtype差异；rollout/replay保存或重建同一configured/effective `sigma_shift`。
3. **canonicalization 具体动作仍需实证**：当前首选是在官方 checkpoint 加载后，从 `_modules` 移除重复注册的 `video_expert/action_expert/dit`，再以非注册兼容属性指向 `mot.mixtures.video/action` 和 `mot`。只有 state-dict、parameter object、optimizer/sync name 唯一且推理 parity 同时通过，才采用；否则停在 B3 讨论。

当前推荐和待决策项统一维护在 [00_INDEX.md](00_INDEX.md)，不另建重复的决策文档。

## 16. 历史反例形成的硬约束

下面不是额外兜底，而是 Motus、LaWAM、π0 和社区 Fast-WAM 已经用失败证明过的实现边界：

1. 官方 Fast-WAM singleton wrapper 只用于 B=1 oracle；生产 B>1 必须是真 batch，异常时 fail-fast，不逐样本 fallback。
2. 三相机、prompt、state、stats 缺失或 shape/顺序不符立即失败；不复制 wrist、不补空 prompt、不伪造 state。
3. eval、train rollout、actor replay 共用 adapter、conditioning、scheduler、velocity 和一个 `flow_sde_rollout(deterministic=...)`；阶段分开验收不等于函数分叉。
4. old在behavior rollout采样真实transition时原位用`flow_step_mean_std + gaussian_logprob`计算并冻结；new才由actor的`recompute_logprob`重放。两者共用的是schedule/conditioning/velocity/mean-std/Gaussian数学，不是都在actor重算。禁止用actor当前值替换behavior denominator来人为制造ratio=1。
5. `forward_inputs` 每个字段在 env、chunk 和决策时间维固定 shape；文本按官方固定长度进入 trajectory，不以当前 batch max 动态变化。
6. 保留 RLinf 常规初始同步与 CPU transport；不复制 Motus 临时关闭完整 init sync 的捷径。
7. 不为Fast-WAM另写FSDP实现：禁block/expert wrap，使用pin的root+非tied Embedding FSDP2布局；不ignore整个冻结大塔导致各rank复制，也不为dtype/FSDP报错临时冻结action expert内有效模块。实施时保存实际wrap inventory。
8. 不粗暴把整模型统一 cast；按官方模块 dtype 构建，并在 FSDP 前保存 parameter name/dtype/device/requires-grad inventory。
9. Fast-WAM 只使用现有 qpos env/trajectory/reward/loss/sync/DCP；LaWAM 的 endpose、EE action 和 planner patch 不迁移。
10. 一个 B=1 step、一次 backward 或一个 runner step 只证明工程接通。DCP fresh-process resume、官方 loader export parity、step 0/更新后固定 seed eval 都必须通过，才能声称实现正确或学到东西。

更细的“历史错误 → 不变量 → 集中验收”索引见 [01_REFERENCE_MATRIX.md](01_REFERENCE_MATRIX.md)，目标 symbol 的精确来源见 [03_CODE_AND_DIFF_MAP.md](03_CODE_AND_DIFF_MAP.md)。

## 17. 逐文件讨论顺序与状态

后续仍按下表逐个文件冻结，但编码时允许把相互依赖文件作为 I1 连续完成。每个文件先回答“直接来源、必要适配、禁止偏离、集中验收、已定选择”五件事。

| 顺序 | 文件 | 本轮状态 | 下一步讨论重点 |
|---|---|---|---|
| 1 | `fastwam_rl.py` | **第一版详细设计已冻结，见第 18 节** | 全部 k 与 `noise_level=0.1` 已拍板；实施时纳入组合数值 fixture |
| 2 | `robotwin_adapter.py` | **第一版详细设计已冻结，见第 20 节** | 实施时对官方 deploy 做 B=1/B>1 组合 parity |
| 3 | `builder.py` | **第一版详细设计已冻结，见第 21 节** | canonicalization 机制以实际对象树检查选定，不改模型数学 |
| 4 | `fastwam_policy.py` | **第一版详细设计已冻结，见第 22 节** | 实施时纳入 policy bridge 组合 fixture与真实 runner |
| 5 | `fastwam/__init__.py` + `rlinf/models/__init__.py` | **精确 registry diff已冻结，见第 22 节** | 实施延迟 import与唯一 `fastwam_robotwin` 注册 |
| 6 | `huggingface_worker.py` | **服务器 pin 已证实需要 capability shim，见第 22 节** | 仅当实施时现场已有等价通用机制才省略 |
| 7 | model/smoke YAML | **字段级方案已冻结，见第 23 节** | 按GPU bucket、固定eval singleton broadcast和P2资源表实施 |
| 8 | `export.py` + tests | **第一版详细设计已收敛，见第 24–25 节** | DCP resume 与官方 deploy export 分开验收 |

本轮已依次冻结 `robotwin_adapter.py`、`builder.py`、`fastwam_policy.py`、registry、worker mode shim、两份 YAML 的字段结构以及 export/P1–P3 边界。实现时上述 Python文件属于同一个 I1 主体批次，不因讨论顺序拆成多次机械停顿；第 26 节三项已经确认，可以进入完整实现。

## 18. `fastwam_rl.py` 第一版详细设计

### 18.1 该文件只解决什么

它负责四件事：构建 Fast-WAM action conditioning、预测 action velocity、沿官方 schedule 做统一 ODE/SDE rollout、对保存的真实 transition 重算 logprob。它不读取 RoboTwin 原始 obs、不做相机拼图/物理 action denorm、不实现 GRPO loss、不管理 FSDP/checkpoint。

### 18.2 唯一生产调用图

```text
build_action_conditioning(...)
  官方 encode_prompt(list[str]) 或 actor replay 的 base text context
  官方 proprio encoder + token append
  官方 WanVideoVAE.encode(batch API)
  官方 video_expert.pre_dit + mot.prefill_video_cache
                ↓
flow_sde_rollout(..., deterministic)
  resolve_action_schedule(..., sigma_shift)
  接收外层一次生成的 initial_latents / shared k / SDE epsilon
  for k in 0..S-1:
      predict_action_velocity()             # train/eval 同一个 grad-capable 函数
      flow_step_mean_std()                  # 官方网格 + RLinf OpenPI 概率
      deterministic=True  → 官方 model-dtype scheduler.step
      deterministic=False → 仅共享选中 k 使用 mean + std * eps
                ↓
  final [B,H,D] + train-only chain/behavior logprob

recompute_logprob(...)
  重建同语义 conditioning
  gather 每个样本真实 x_k/x_(k+1), t_k, delta_k
  同一个 predict_action_velocity + flow_step_mean_std
  输出逐元素 new logprob [B,H,D]
```

不另写 `deterministic_ode_rollout()`；阶段 B4/B5 分开验收，但生产函数不分叉。官方 `infer_action()` 只在测试里提供 B=1 oracle。

### 18.3 目标函数、来源和适配

| 函数 | 直接来源 | 第一版实现 | 不允许的做法 |
|---|---|---|---|
| `resolve_action_schedule()` | 官方 `scheduler_continuous.py::build_inference_schedule` | 传 `num_inference_steps/device/model dtype/shift_override`；返回 raw timestep、normalized `t`、signed delta、`dt=-delta`、下一点分母和 effective shift | 漏传 `sigma_shift`；自己造线性网格；`abs(delta)` 掩盖方向错误 |
| `prepare_initial_action_latents()` | 官方 `infer_action` 的 noise 构造 | 固定 eval 生成官方 `B=1, seed=0` singleton 后沿 B 复制，精确复现每个环境独立调用同 seed 的 deploy 语义；train 不走此分支，而由 `prepare_rollout_randomness()` 生成逐样本独立 latent；测试可显式注入 latent | 把 eval singleton 复制偷换进 train；每个 resource chunk 重置 seed；异常后逐 env fallback |
| `prepare_rollout_randomness()` | RLinf OpenPI共享k + 官方initial noise | 在逻辑batch分块前一次生成`initial_latents [B,H,D]`、共享`k`和`sde_epsilon [B,H,D]`；chunk只取slice | 每个chunk重采k/noise；分块改变策略样本 |
| `encode_first_frame_latents()` | 官方 `WanVideoVAE.encode` | 构造B个`[3,1,H,W]`video的list交给官方wrapper；wrapper自身逐项encode后返回stack，目标代码不二次stack；保持官方tiled/dtype | 未profiling就绕过wrapper；二次stack；catch exception逐env fallback |
| `build_action_conditioning()` | 官方 prompt/proprio/video pre-DiT/cache；社区同名函数 | 显式核对 B；首版 frozen conditioning 在 no-grad 中重建；返回 cache/mask/seq len，不保存到 trajectory | actor 直接复用 rollout KV cache；缺 wrist/context 时伪造 |
| `predict_action_velocity()` | 官方 `_predict_action_noise_with_cache` + 社区grad-enabled twin | 调用图逐行同构并移除decorator；显式`x_model=x_fp32.to(model_dtype)`、`raw_t_model=raw_t.to(model_dtype)`，所有timestep为`[B]`；velocity返回后upcast FP32算概率 | FP32 latent直接喂BF16 expert；`__wrapped__`/monkeypatch；train/eval两份velocity |
| `flow_step_mean_std()` | RLinf `6d0db56` OpenPI `sample_mean_var_val` + 官方 Fast-WAM schedule | 下述精确公式，FP32；返回 `mean/std/mean_ode` | 社区 `0.98` hard clamp；std floor；legacy `simple` |
| `gaussian_logprob()/entropy()` | RLinf 逐元素 Gaussian | FP32、保留 `[B,H,D]`；selected step std 非正/非有限即失败 | policy 内提前 sum/mean；用 floor 静默改变 density |
| `flow_sde_rollout()` | 社区同名函数 + RLinf OpenPI rollout | 一个函数由 `deterministic` 控制；train共享 k、样本噪声独立；eval不保存 chain | full-chain SDE；catch-all fallback；eval保存大 replay |
| `recompute_logprob()` | 社区同名函数 + RLinf OpenPI/Motus model-side replay | 每样本 gather真实 transition；actor batch可有不同 k；只对 action expert保留梯度 | actor伪造 old logprob；用最终 action反推 chain；重采样 transition |

### 18.4 schedule 与 Flow-SDE 精确公式

官方 Fast-WAM 网格：

```text
u_i = linspace(1, 0, S+1)
sigma_i = shift * u_i / (1 + (shift - 1) * u_i)
raw_t_i = sigma_i * num_train_timesteps          # i=0..S-1
signed_delta_i = sigma_(i+1) - sigma_i < 0
mean_ode = x + signed_delta_i * v
```

RLinf OpenPI 使用正的 `dt`，因此适配只做一次有方向的转换：

```text
t = raw_t / num_train_timesteps
assert signed_delta < 0
dt = -signed_delta
denom_t = next_t if t == 1 else t

x0 = x - t * v
x1 = x + (1 - t) * v
sigma = noise_level * sqrt(t / (1 - denom_t))

mean = x0 * (1 - (t - dt))
     + x1 * (t - dt - sigma^2 * dt / (2*t))
std = sqrt(dt) * sigma
mean_ode = x + signed_delta * v
```

首步 `t=1` 时只替换 sigma 分母中的 `t` 为下一 schedule 点，分子仍为当前 `t=1`；这逐项对应 RLinf OpenPI 的 `denom_timesteps = where(timesteps == 1, timesteps[1], timesteps)`。社区 `t.clamp(..., 0.98)` 只是历史修补，不进入目标实现。

官方 config 的 action `infer_shift=5.0`；standalone `sigma_shift=null` 表示使用 scheduler 自身的 5.0。若 resolved config 提供 override，rollout 和 replay 都必须把同一值传给官方 schedule，并记录 configured/effective shift。`sigma_shift` 不是新实验超参，不能自行选择。

### 18.5 k、chain 与 batch 语义

- 完整保存 `chains [B,S+1,H,D]`；behavior logprob只记录选中 transition 的逐元素值，供前 N 个 action token进入 loss。
- rollout batch按 RLinf OpenPI惯例共享一次采样的 `k`；每条 trajectory 的 initial latent 和 SDE epsilon 独立。
- 若启用固定`model_forward_batch_size`，外层先调用`prepare_rollout_randomness()`，再把同一逻辑batch的随机Tensor切片给每个chunk；`flow_sde_rollout`不在chunk内重采样。
- trajectory flatten/actor microbatch 后，不假设所有样本仍有同一个 k；`recompute_logprob` 接收 `[B] denoise_inds` 和 `[B] timestep`。
- 首版已冻结从全部 `k∈[0,S-1]` 均匀采样，不忽略首/末步。首步已经由 RLinf OpenPI 分母规则变为有限；社区最终成功实现与 RLinf `ignore_last=false` 也提供交叉证据。
- train 模式要求 `noise_level>0`；selected std必须严格正且有限。eval只走 `mean_ode`，不计算 Gaussian density。

### 18.6 batch 推理难点与控制

难度是中等，不在模型结构本身，而在四个边界：

1. 官方 public `infer_action()` 写死 B=1，但底层 prompt/proprio/VAE返回、video/action expert、KV cache和 scheduler均保留 B；因此只 batch 化编排层。
2. VAE官方 wrapper内部逐样本编码后stack。首版接受这一官方语义；真正重的 video/action MoT仍按 B 一次执行。
3. KV cache显存按`B × layer × video_tokens × hidden`增长。用显式`model_forward_batch_size`做固定资源分块，失败即停；共享k/initial/SDE epsilon在分块外一次生成。社区成功run的chunk 8只作起始证据，不直接当本机结论。
4. rollout plain BF16、actor FSDP BF16和不同 batch partition可能放大微小 velocity差异成巨大 ratio漂移。必须同时做同进程 parity和真实 rollout→actor、optimizer前 parity；绝不能以 actor自产 old logprob掩盖。

### 18.7 `fastwam_rl.py` 纳入 P1 的高信息量检查

1. 一个固定随机 fixture 同时检查 B=1 官方 `infer_action()` oracle、B=2 batch/permutation/cache 隔离和实际采用的一种固定分块；不穷举所有 batch/chunk 组合。
2. 同一 fixture 检查 official scheduler/effective shift/signed delta、首步 denominator、FP32 mean/std/logprob；无 `0.98`、无 std floor、无 NaN/Inf。
3. rollout 保存的 `x_k/x_(k+1)` 在同权重 actor 重算后逐元素 old/new 一致；轻微修改 action expert 后 new 变化、old不变。
4. 真实 P2 runner 再记录 optimizer 前 ratio/clip、实际 FSDP partition 与只有 canonical action expert 有梯度；不在离线阶段复制大量微型 gate。

## 19. 用户已拍板的三项

1. **代码落点**：采用内置 `rlinf/models/embodiment/fastwam/`，不再维护外置 `extensions/fastwam_rlinf/`。
2. **stochastic k 范围**：首版对全部 `0..S-1` 均匀采样，不设 `ignore_first/ignore_last`。
3. **首个非零探索 smoke**：使用`noise_level=0.1`，只验证稳定性和真实ratio。在已确认的`S=10, shift=5`网格下，按本节公式各step的元素std约为`0.0445–0.1079`、首步为`0.1`，属于保守工程起点。通过后再根据任务baseline决定是否比较`0.1/0.3`，这不是论文最终超参冻结。

其余事项不作为“自由选择”：`sigma_shift`服从 standalone resolved config；首版使用官方 VAE wrapper；batch chunk由实测峰值决定；第一版只训练 action expert；`N=24`工程 smoke固定192 steps。

## 20. `robotwin_adapter.py` 第一版详细设计

### 20.1 输入合同与来源

adapter 只把 RLinf pin `6d0db56:rlinf/envs/robotwin/robotwin_env.py::_extract_obs_image` 的 batch observation 变成官方 Fast-WAM RoboTwin deploy 语义：

```text
main_images       uint8 [B,Hm,Wm,3]      = head_camera.rgb
wrist_images      uint8 [B,2,Hw,Ww,3]    = [left_camera.rgb, right_camera.rgb]
states            float [B,14]            = [left arm 6, left gripper 1, right arm 6, right gripper 1]
task_descriptions list[str], len=B
```

`validate_env_obs()` 要求四项同时存在、batch 对齐、图像为 HWC RGB uint8、state 为有限 14D、instruction 为非空字符串。它不限定原始 H/W，因为官方 deploy 本来就 resize；不自动猜 CHW、不把 float 图像猜回 uint8、不截断 state、不复制缺失 wrist、不补空 prompt。

### 20.2 三相机拼图

`compose_three_camera_image()` 逐样本严格复用官方 `deploy_policy.py::_resize_rgb/_build_robotwin_image_tensor`：PIL bilinear 将 head resize 为 W=320/H=256、left/right 各 W=160/H=128；先左右横拼，再接到 head 下方，得到 `[384,320,3]`，batch 后转 `[B,3,384,320]`。随后保持官方顺序：

```python
x = uint8_tensor.to(device=device, dtype=model_dtype)
x = x * (2.0 / 255.0) - 1.0
```

首版保留这个轻量 CPU batch loop，而不换成 `F.interpolate`：官方训练数据 pipeline 与 standalone deploy 的 resize 路径并非逐像素相同，本迁移首先以已经跑通的 deploy 为 oracle。目标配置保持 `center_crop=false`；只有实测证明预处理成为瓶颈后，才另做向量化并重新 parity。

### 20.3 state、action 与 prompt

- `normalize_proprio()`：按官方 deploy 把 `[B,14]` FP32 physical qpos 包成 `{"state":{"default": states}}`，依次调用 `processor.action_state_transform` 和 `processor.normalizer.forward`。当前锁定配置为 singleton key `default`、global z-score、forward 后 clamp `[-5,5]`；即使 transform 当前是 identity，也不手抄成另一条公式。
- `denormalize_actions()`：对 `[B,H=32,14]` normalized FP32 直接调用 `processor.normalizer.normalizers["action"]["default"].backward`，返回 CPU FP32 physical absolute qpos。禁止引入 π0 delta/关节转换、LIBERO gripper invert/binarize 或 LaWAM EE/endpose。
- `build_prompts()`：精确使用官方 `DEFAULT_PROMPT`：`A video recorded from a robot's point of view executing the following instruction: {task}`。只包装 env 已选 instruction，不重新随机选句子，不“修正”官方 tokenizer/mask 语义。

adapter 不实现 action queue、reward/done、trajectory 或概率。policy 对完整 `[B,32,14]` denorm 后截前 `N=24` 给现有 `RoboTwinEnv.chunk_step`；后 8 个丢弃，与官方 `生成32→queue前24→重观测` 等价。

### 20.4 目标符号与集中检查

```text
validate_env_obs(env_obs) -> ValidatedRobotWinBatch
compose_three_camera_image(main, wrist, device, dtype) -> [B,3,384,320]
normalize_proprio(states, processor) -> fp32 [B,14]
denormalize_actions(model_actions, processor) -> cpu fp32 [B,32,14]
build_prompts(task_descriptions) -> list[str]
adapt_robotwin_observation(...) -> image, normalized_proprio, prompts
```

P1 用一个组合 fixture 覆盖 B=1 官方 deploy 的图像/prompt/norm/denorm和 B=2 的 batch/left-right/permutation；再用一个错误 shape 证明 strict fail-fast即可，不为每个 helper 单建 gate。

## 21. `builder.py` 第一版详细设计

### 21.1 构建顺序与配置权威

`build_fastwam_policy(cfg, torch_dtype)` 只编排官方对象，不复制模型源码。顺序固定为：

1. 在隔离的 Hydra compose 上下文中组合官方 `configs/sim_robotwin.yaml` + `task=robotwin_uncond_3cam_384_1e-4`，并应用 RLinf YAML 中允许的路径/资源参数；compose 前后清理 `GlobalHydra`，不让官方 config 污染已经 resolved 的 RLinf config。
2. 与 standalone deploy 一样，把模型配置的 `load_text_encoder=True` 后调用官方 `fastwam.runtime.create_fastwam`；首版 actor/rollout走同一 registry 构建图，先保证权重与 state-dict 一致，actor虽不重跑 T5也不先做角色特化。
3. checkpoint 与 dataset stats 路径必须显式存在；依次调用官方 `model.load_checkpoint()`、构造 `FastWAMProcessor`、`load_dataset_stats_from_json(path)` 与 `processor.set_normalizer_from_stats(stats)`。官方 processor 没有 `load_dataset_stats()`；不允许缺 checkpoint 时随机初始化，也不猜另一份 stats。
4. 在 checkpoint 已加载、FSDP 尚未 wrap 时 canonicalize 模块树，再冻结全模型、仅开放 canonical action expert，最后包成 `FastWAMPolicy`。

权威优先级是：官方 resolved config决定模型结构、H/D、scheduler/shift和 processor；RLinf YAML决定文件路径、`N`、`S`、`noise_level`、batch chunk、placement/FSDP/sync。与 H=32、D=14、三相机或 stats schema冲突的 override直接报错，不用“RLinf cfg wins”静默覆盖。

builder 不修改 `HF_ENDPOINT`/`DIFFSYNTH_DOWNLOAD_SOURCE`，不复制社区强制 Hugging Face 的环境设置；本机已跑通的官方 ModelScope组件路径由启动环境提供。

### 21.2 模块树 canonicalization

官方对象同时注册 `video_expert`、`action_expert`、`mot` 和 `dit=mot`，其中专家又是 `mot.mixtures.video/action` 的同一对象。RLinf sync 会枚举 `remove_duplicate=False`，所以这是必须有证据的 RLinf兼容修复，而非模型改造：

1. 先断言三个 identity：`video_expert is mot.mixtures.video`、`action_expert is mot.mixtures.action`、`dit is mot`。
2. 以 `model.mot.mixtures.video/action` 为唯一注册树，从 `_modules` 去掉重复注册路径；需要兼容官方内部访问时只保留非注册 alias。
3. 断言 state-dict、parameter object、optimizer/sync name无逻辑重复；固定 latent 的 action 在变换前后保持一致。

具体使用 `object.__setattr__` 非注册 alias 还是小型 property，由实施时的真实对象树测试决定；选择标准只有“官方访问仍工作 + 注册树唯一 + action parity”，不为这点维护两套长期路径。

### 21.3 冻结范围与 train/eval mode

`freeze_action_only()` 先全部 `requires_grad_(False)`，再只开放 `model.mot.mixtures["action"]`。processor/stats、proprio、video、VAE、T5全冻结。policy 的 `train(mode)` 还必须在 `super().train(mode)` 后恢复冻结模块为 eval，仅让 action expert处于 train；MoT外壳若需 `training=True` 以启用官方 mixed-attention/checkpoint语义，则显式设置后再次把 video/proprio/VAE/T5设回eval。这样 actor训练不会把冻结 conditioner意外切进 train/dropout模式，也不影响 action梯度。

官方 resolved `mot_checkpoint_mixed_attn` 首版不由 builder 静默改写；gradient checkpointing是实测显存后的资源旋钮。若启用，做一次 backward 确认 action梯度即可，不把它变成算法分支。

### 21.4 registry 与 P1/P2 检查

`fastwam/__init__.py::get_model()` 只延迟调用 builder；`rlinf/models/__init__.py` 按 pin 的 `_register_builtin_models()` 增加唯一 `fastwam_robotwin`。driver、rollout、actor因此走同一构建入口；不新增 launcher、模型枚举或 monkeypatch。

P1 用一个 builder/inventory 组合检查覆盖：官方 checkpoint+stats成功加载、canonicalization前后固定 action一致、name唯一、仅action trainable、冻结模块mode正确。真实 FSDP wrap、bucket sync和driver/Worker双构建留到 P2 runner一并观察，不在构建前铺多套仿真 gate。

## 22. `fastwam_policy.py`、registry 与 rollout mode 第一版详细设计

### 22.1 在调用链中的位置

这三部分共同构成 RLinf 到 Fast-WAM 数值 core 的桥：registry 在最上方负责构造，worker 负责把 rollout 算法 mode传入，policy把环境 batch与actor replay接到已冻结的 adapter/Flow-SDE。

```text
train_embodied_agent.py / Hydra
  -> rlinf.models registry
  -> fastwam.get_model() -> builder.build_fastwam_policy()
  -> huggingface_worker.predict(mode)
  -> FastWAMPolicy.predict_action_batch()
       -> robotwin_adapter.py
       -> fastwam_rl.flow_sde_rollout()
  -> result dict -> RolloutResult -> RoboTwinEnv.chunk_step / trajectory

trajectory -> EmbodiedFSDPActor
  -> FastWAMPolicy.forward(ForwardType.DEFAULT)
  -> default_forward()
  -> fastwam_rl.recompute_logprob()
  -> RLinf通用GRPO loss
```

直接上游分别是 RLinf registry、`huggingface_worker.py` 和 actor 的通用 `self.model(...)`；直接下游是 `builder.py`、`robotwin_adapter.py`、`fastwam_rl.py` 与 RLinf `RolloutResult`。env、trajectory、actor、loss、sync、DCP都不增加 Fast-WAM 专用分支。

### 22.2 `FastWAMPolicy` 的四个显式 symbol

类定义固定为 `FastWAMPolicy(nn.Module, BasePolicy)` 并声明类属性 `rlinf_accepts_rollout_mode=True`。由于此 MRO会先命中 `nn.Module.forward`，目标类必须显式实现 `forward()`；首版只分发 `ForwardType.DEFAULT -> default_forward()`，其他类型直接 `NotImplementedError`，不顺带实现 SFT或critic。首版完全不创建 `value_head` 属性：当前 worker用 `hasattr` 判断bootstrap，社区的 `value_head=None` 会触发无谓的额外推理。

`predict_action_batch(env_obs, mode="eval", **kwargs)` 使用 `@torch.no_grad()`，严格只接受 train/eval：

1. adapter得到 `image [B,3,384,320]`、normalized `proprio [B,14]` 与 prompts；一次调用官方 `encode_prompt(list[str])` 得到未追加 proprio 的固定 `text_context [B,128,4096]` 和 mask。
2. 在任何固定资源分块之前，为整个逻辑 batch一次生成 initial latent、共享 `k` 与逐样本 SDE epsilon；然后调用同一个 `flow_sde_rollout(deterministic=(mode=="eval"))`。
3. 完整 normalized action为 `[B,32,14]`；按官方 stats全量 denorm成 physical qpos，再截前24，返回 CPU FP32 `actions [B,24,14]`。
4. eval返回 `(actions,{})`，不保留 replay。train返回 `prev_logprobs [B,24,14]`、`prev_values=None` 与下述 Tensor-only `forward_inputs`；worker随后构造 `RolloutResult`。

```text
chains             [B,11,32,14]  fp32
denoise_inds        [B]           int64
image               [B,3,384,320] model dtype
text_context        [B,128,4096]  model dtype，未追加proprio
text_context_mask   [B,128]       bool
proprio             [B,14]        fp32 normalized
action              [B,24*14]     fp32 physical qpos
model_action        [B,32*14]     fp32 full normalized action
```

`chains[:,0]` 是 initial Gaussian；`chains[:,k] -> chains[:,k+1]` 是 behavior policy真实采样的 transition。chain和model action保留完整 H=32；只有环境 action与old/new logprob截前 N=24。`action` 供 EnvWorker/trajectory记账，actor概率只重放chain，不从physical或final action反推。字符串、KV cache、video chain、velocity和debug Tensor都不进 replay。

`default_forward(forward_inputs, compute_logprobs=True, compute_values=False, compute_entropy=False, **kwargs)` 验证固定 Tensor合同，用保存的 image、基础text context/mask与normalized proprio重建冻结conditioning，按每个样本的 `denoise_inds` gather真实 transition并调用一次 batch-native `recompute_logprob()`。首版要求 logprob且拒绝 values；返回逐元素 `logprobs [B,24,14]`，仅请求时返回同shape entropy，不在policy内sum/mean。`use_cache=False`等actor通用参数由`**kwargs`接收但不改变数值路径。

`train(mode)` 解决 PyTorch module mode，而不是 rollout随机性：先 `nn.Module.train(self, mode)`；`mode=True` 时保持 policy/model/MoT/action expert为train，再把 video expert、VAE、T5与proprio encoder恢复为eval；`mode=False` 时全部eval。`predict_action_batch(mode="train")` 则始终是 rollout no-grad下的SDE/replay选择，rollout module仍为eval。禁止用`self.training`猜 rollout类型，也禁止调用会主动 `eval()`、no-grad且写死B=1的官方`infer_action()`作为生产路径。

首个 smoke 固定 `gradient_checkpointing:false`，因此第一版不以社区 no-op 掩盖未启用能力。若资源实测要求开启，`FastWAMPolicy.gradient_checkpointing_enable()` 必须显式接通 action expert 的 `use_gradient_checkpointing` 与 MoT 的 `mot_checkpoint_mixed_attn`，再用真实 backward 验证；在此之前不把 YAML 的 `true` 写成已生效。

### 22.3 registry 的唯一两处与零 diff边界

`fastwam/__init__.py::get_model(cfg, torch_dtype)` 只延迟导入并转交 `builder.build_fastwam_policy()`。`rlinf/models/__init__.py::_register_builtin_models()` 增加一个延迟 builder和一条：

```python
register_model(
    "fastwam_robotwin",
    _build_fastwam_robotwin,
    category="embodied",
    force=True,
)
```

锁定 pin 的 `register_model()` 已同时更新 `_MODEL_REGISTRY`、`SupportedModel` 与 `EMBODIED_MODEL`；因此 `config.py`、driver、actor、BasePolicy、async worker、RoboTwin action utils均零 diff。`fastwam_robotwin` 明确绑定三相机/14D qpos，避免与社区 LIBERO 的泛化名称混用。actor和rollout都经同一 `get_model()`；训练时 Fast-WAM结构参数只放在 `actor.model`，rollout配置按RLinf既有复制机制取得，only-eval配置才需自足。YAML必须显式 `is_lora:false`。

### 22.4 `huggingface_worker.py` 的必要 capability shim

服务器 `6d0db56` 已确认 worker虽知道 train/eval mode，却只把它传给硬编码模型列表；且 rollout模型始终 `.eval()`。所以当前 pin必须在现有硬编码分支后、`return_obs`独立分支前增加默认关闭的 `elif`：

```python
elif getattr(self.hf_model, "rlinf_accepts_rollout_mode", False):
    loss_type = self.algorithm_cfg.get("loss_type", "actor")
    kwargs = dict(kwargs)
    kwargs["mode"] = "eval" if loss_type == "embodied_dagger" else mode
```

现有硬编码分支逐字保留，故π0仍收到原来的纯mode字典；无capability旧模型仍只有原sampling kwargs；Fast-WAM保留worker已选的sampling kwargs再追加mode；DAgger仍强制eval。`dict(kwargs)`避免改写共享配置。async worker继承该方法，不做第二处修改。只有实施时现场代码已经出现等价通用机制，才省略这条diff；不能用`self.training`、默认固定mode、伪装OPENPI或异常fallback规避。

### 22.5 本批集中检查与下一批

P1 的一个 policy bridge组合 fixture即可覆盖：B=1 eval同官方 oracle、B=2/permutation/固定分块；eval result为空；train replay全部字段shape/dtype；同权重old/new逐元素一致且微改action expert后只有new变化；`policy.train()`后仅canonical action expert处于train且有grad。再参数化检查三种worker dispatch：π0原分支不变、普通旧模型无mode、Fast-WAM保留sentinel sampling kwarg并收到train/eval/DAgger mode。

P2真实`lr=0` runner再统一覆盖 `RolloutResult` CPU transport、trajectory flatten、FSDP后的ratio/clip、registry双构建与train rollout确有chain；不为每个helper新增独立gate。两份YAML、export和P1–P3的最终设计接续如下。

## 23. 两份 YAML、FSDP、sync 与 runner 第一版设计

### 23.1 调用链位置

```text
model YAML + smoke YAML
  -> train_embodied_agent.py / Hydra                         [不改]
  -> HybridComponentPlacement                               [不改]
  -> EnvWorker / MultiStepRolloutWorker / EmbodiedFSDPActor [不改]
  -> FastWAM registry / policy
  -> RolloutResult / Trajectory / GRPO loss                  [不改]
  -> FSDP2 optimizer -> bucket sync -> next rollout          [不改]
```

两份 YAML 是调用链最上游的唯一 Fast-WAM 实验接线。训练时完整模型结构参数只放 `actor.model`；当前 worker 从它复制 rollout 构造配置，并只用 `rollout.model.model_path/precision` 覆盖路径与精度。因此不能维护 actor/rollout 两份会漂移的 Fast-WAM 配置。

### 23.2 `model/fastwam_robotwin.yaml`

首版字段合同：

```yaml
model_type: fastwam_robotwin
model_path: /root/autodl-tmp/models/fastwam/release/robotwin_uncond_3cam_384.pt
precision: bf16
is_lora: false

checkpoint_path: ${.model_path}
dataset_stats_path: /root/autodl-tmp/models/fastwam/release/robotwin_uncond_3cam_384_dataset_stats.json

action_dim: 14
action_horizon: 32
num_action_chunks: 24
num_inference_steps: 10
sigma_shift: null
text_cfg_scale: 1.0
negative_prompt: ""
rand_device: cpu
tiled: false
model_forward_batch_size: 2
eval_seed: 0
add_value_head: false

rl:
  noise_level: 0.1

fastwam:
  config_name: sim_robotwin
  task: robotwin_uncond_3cam_384_1e-4
```

`H/N/S=32/24/10`、14D、`text_cfg_scale=1.0`、空 negative prompt、CPU随机数生成与 `sigma_shift:null -> effective shift=5.0` 来自成功官方 resolved config；builder要断言而不是静默接受冲突。`model_forward_batch_size` 只切模型前向资源，不重新采样 k/initial/epsilon，不构成概率分支。社区的 `noise_method`、`sde_sampling`、`batched`、gripper转换和异常 singleton fallback全部不进入此 YAML。

`eval_seed=0`只用于eval：一次生成官方 B=1 singleton latent 后沿 batch broadcast，以保证 singleton oracle 与样本置换等价。train rollout不用该固定 latent，而由 `prepare_rollout_randomness()` 对每条 trajectory产生独立随机量。

### 23.3 `robotwin_adjust_bottle_grpo_fastwam_a800_2gpu_smoke.yaml`

首轮P2资源表冻结如下；它是工程 smoke，不宣称最终训练超参：

| 项 | 冻结值 | 依据/作用 |
|---|---:|---|
| placement | `actor,env,rollout: 0-1` | 现役 π0 两卡基座；双 A800 |
| train env / group / rollout epoch | `16 / 8 / 1` | 两个真实GRPO group；不让过滤静默清空 |
| train/eval episode与rollout limit | `192` | 同时写 `max_episode_steps`、`max_steps_per_rollout_epoch`、`task_config.step_lim`；恰好8个24-action chunks |
| eval env / rollout epoch | `2 / 1` | P2默认不跑val；P3固定eval的低并发起点 |
| actor global/micro batch | `128 / 2` | `16 env × 1 epoch × 8 decisions=128`；两卡可整除，先留显存余量 |
| model forward batch | `2` | 首个真实batch而非singleton fallback |
| update epoch | `1` | smoke只验证一条真实更新链 |
| actor/rollout offload | `false/false` | 先保留GPU常驻；OOM后再按证据启用 |
| train env offload | `true` | 沿用现役 RoboTwin 两卡经验 |
| gradient checkpoint | `false` | 官方 RoboTwin resolved关闭；不以no-op伪装生效 |
| filter rewards | `false` | 防止全成/全败group让smoke loss静默归零 |
| runner | `max_steps=1`, `weight_sync_interval=1`, `save_interval=1`, val关闭 | 一步更新后立即有DCP，生命周期另在P3验收 |
| optimizer | 先`lr=0`，新进程再`lr=1e-6`；`clip_grad=1.0` | 零更新测behavior-old一致性；非零只证明更新/同步，不是最终LR |

192必须在 train/eval 的三个层级同时覆盖；只改外层两个值会让 RLinf rollout 与 RoboTwin 内部终止条件不同。`group_size=8` 和 `num_action_chunks=24` 分别是GRPO组大小与每次环境动作数，不能混用。

FSDP2字段固定为：

```yaml
actor:
  training_backend: fsdp
  fsdp_config:
    strategy: fsdp2
    gradient_checkpointing: false
    enable_gradient_accumulation: true
    wrap_policy:
      no_split_names: ["__none__"]
    save_full_model_weights: false
    mixed_precision:
      param_dtype: bf16
      reduce_dtype: bf16
      buffer_dtype: bf16
```

`["__none__"]` 是可执行的“禁止block/expert wrap”：当前 pin仍会单独wrap非tied `nn.Embedding`，再wrap root；root的 `reshard_after_forward=False` 在代码中硬编码，YAML的同名值只影响被单独wrap的Embedding。`use_orig_params`在这条FSDP2路径未使用，首版不把它当能力。regular checkpoint必须显式 `save_full_model_weights:false`，因为pin默认可能为true；里程碑deploy从DCP转换，不让每步额外写约一份完整权重。

bucket算法与本机device均已固定：

```yaml
weight_syncer:
  type: bucket
  bucket:
    bucket_size: 536870912
    bucket_dtype: null
    bucket_device: cpu
    is_agent: false
    load_instant: true
```

社区 Fast-WAM的直接成功证据是未显式device、因而在当前pin默认走accelerator的bucket。项目初版据此选择GPU bucket；但 2026-07-18 首次双 A800 smoke 在模型均成功构建后，于初始 actor→rollout 广播的 `torch.storage._new_shared_cuda()` 报 `pidfd_getfd: Operation not permitted`。这说明失败位于本机容器的CUDA IPC传输层，尚未进入rollout、loss或backward。修正为RLinf原生CPU bucket：保留同一bucket算法、同步集合、dtype和512 MiB分桶，只把staging/transport移到CPU；保留`expandable_segments:True`以降低大模型显存碎片。依据是RLinf的CPU/GPU bucket能力、本机π0成功使用CPU transport，以及本次现场错误链。

## 24. `export.py` 第一版详细设计

### 24.1 调用链位置与来源

```text
RLinf global_step_N/dcp_checkpoint
  -> pin convert_dcp_to_pt.py 的 no_dist=True model-state抽取
  -> export.py exact canonical subtree映射
  -> 官方 {mot, proprio_encoder?, step, torch_dtype}
  -> fresh FastWAM.load_checkpoint()
```

直接上游是RLinf DCP，直接下游是锁定commit的官方loader。目标pin `6d0db56:rlinf/utils/ckpt_convertor/fsdp_convertor/convert_dcp_to_pt.py` 已使用 `FileSystemReader + _EmptyStateDictLoadPlanner(keys={"fsdp_checkpoint.model"}) + no_dist=True`，所以不实现matching-world-size gather，不新增checkpoint manager，也不要求regular run保存`full_weights.pt`。

### 24.2 目标符号与严格边界

```text
resolve_actor_checkpoint(path)
load_rlinf_model_state(path)
extract_official_fastwam_payload(state_dict, base_checkpoint, step)
validate_official_schema(payload, base_checkpoint)
export_deploy_checkpoint(...)
main()
```

1. `resolve_actor_checkpoint()`只接受明确的`global_step_N`、`dcp_checkpoint`或`full_weights.pt`，不做模糊全盘递归。
2. `load_rlinf_model_state()`以pin的no-dist DCP语义为主，full weights只作兼容输入。
3. P1 inventory冻结prefix后，只接受唯一canonical `model.mot.*`与可选`model.proprio_encoder.*`；未知wrapper、重复alias、缺key或额外logical key立即失败。
4. 用原官方checkpoint作schema oracle，exact比较key、shape与dtype；官方`load_checkpoint()`对`mot`是`strict=False`，所以“没报错”不能替代严格校验。
5. 输出只含`mot`、可选`proprio_encoder`、`step`、`torch_dtype`；不含optimizer、wrapper、VAE/T5或`dit` alias。`mot`必须包含冻结video与更新后的action，不能只导出trainable子树。
6. `export.py`本身提供CLI，不再新增shell/python launcher。

P3 parity的正确对象是“fresh-process从DCP恢复的RLinf模型”与“同一DCP导出后由fresh官方模型加载的模型”。非零更新后的action本来就不应继续等于原base checkpoint；只有`lr=0` checkpoint应保持base parity。

## 25. 完整改动面、来源与集中验收

首版实际为15个新增文件、3个小改，共18个代码/配置/测试/运行文件；第15个是实施中补出的高信息量 wiring 回归，最后三项是正式配置、launcher和资源监控：

| # | 文件 | 调用链位置；直接上下游 | 主要来源 | 首版适配/完成定义 |
|---:|---|---|---|---|
| 1 | `examples/.../robotwin_adjust_bottle_grpo_fastwam_a800_2gpu_smoke.yaml` | 顶层；启动命令→Hydra/runner | π0 resolved、RLinf pin、社区h240 | 23.3字段；三项资源选择落盘 |
| 2 | `examples/.../model/fastwam_robotwin.yaml` | 模型合同；smoke YAML→builder | 官方sim/task/data config、社区字段布局 | 23.2字段；官方H/N/S/shift/14D断言 |
| 3 | `rlinf/models/__init__.py`（改） | registry；worker→Fast-WAM package | pin `register_model/_register_builtin_models` | 一条延迟builder与唯一`fastwam_robotwin`注册 |
| 4 | `fastwam/__init__.py` | package入口；registry→builder | RLinf内置模型惯例、社区get_model | 薄`get_model()`；import不加载大模型 |
| 5 | `builder.py` | 构建；get_model→官方对象/policy/FSDP | 官方runtime/load/deploy、社区compose | Hydra隔离、官方load、stats、canonicalization、action-only freeze与inventory |
| 6 | `robotwin_adapter.py` | 数据边界；policy→processor/RL core | 官方deploy、pin RoboTwin obs | `ValidatedRobotWinBatch`与7个纯helper；严格三相机/14D/prompt/norm |
| 7 | `fastwam_rl.py` | 数值core；policy→MoT/replay | 官方scheduler/MoT/VAE、pin OpenPI、社区final、Motus | 4个轻量dataclass与schedule/conditioning/velocity/ODE-SDE/recompute函数；单一loop |
| 8 | `fastwam_policy.py` | RLinf桥；worker/actor→adapter/RL core | BasePolicy、社区policy、Motus wrapper | 显式`forward/train/predict/default_forward`与capability；无value_head属性 |
| 9 | `huggingface_worker.py`（改） | mode分发；EnvWorker→policy | pin现有硬编码分支 | 默认关闭capability `elif`；旧模型与π0不变 |
| 10 | `export.py` | 部署边界；DCP→官方loader | pin no-dist converter、官方save/load | 第24节strict schema与同文件CLI |
| 11 | `test_adapter_builder_parity.py` | P1 unit | 官方deploy、Motus/LaWAM反例 | adapter/processor、module mode与严格tensor-only replay schema；不冒充真实checkpoint parity |
| 12 | `test_flow_sde_replay.py` | P1 | 官方schedule、OpenPI、社区final | k全覆盖、first-step、chunk parity、same-weight old/new与微改权重 |
| 13 | `test_checkpoint_export.py` | P3离线部分 | pin converter、官方schema | synthetic DCP/full输入、prefix/schema失败用例、官方payload |
| 14 | `rlinf/hybrid_engines/fsdp/strategy/fsdp2.py`（改） | actor FSDP调用边界；actor worker→policy replay | pin FSDP2 `MixedPrecisionPolicy`、Fast-WAM FP32概率链 | 新增通用`cast_forward_inputs`配置，默认`true`保持旧模型；Fast-WAM设`false`，由policy只在模型边界显式cast BF16 |
| 15 | `test_rlinf_wiring.py` | P1 wiring | public registry、pin HF worker、pin FSDP2 | 验证lazy registry、capability mode分发，以及旧模型默认cast=true/Fast-WAM显式false |
| 16 | `examples/.../robotwin_adjust_bottle_grpo_fastwam_a800_2gpu.yaml` | 顶层正式配置；launcher→Hydra/runner | 成功π0 resolved、社区Fast-WAM、真实smoke资源 | 100-step formal、64 trajectories与本机offload/CPU bucket；运行目录resolved config为实验真相 |
| 17 | `examples/embodiment/run_fastwam_robotwin_grpo.sh` | 运行入口；shell→RLinf标准driver/monitor | 现有RLinf入口与旧π0实验日志习惯 | smoke/train单命令；保存命令、resolved config、PID和run log，不另写runner |
| 18 | `examples/embodiment/monitor_resources.py` | 旁路观测；launcher→cgroup/NVIDIA/process RSS | 旧π0 PPO/GRPO资源CSV格式 | 与训练同生命周期写CSV/peak，不参与模型、reward或优化 |

真实 checkpoint parity 使用服务器上可复现生成的固定 synthetic RoboTwin observation，不把模型、图像大文件或机器路径写进源码测试；命令和结果写入`evidence/IMPLEMENTATION_LOG_20260717.md`。P3再对恢复后的RLinf模型和导出后官方loader使用同一fixture。

明确零diff：`train_embodied_agent.py`、`config.py`、`BasePolicy`、RoboTwin env/reward、`embodied_io_struct.py`、trajectory、actor worker、advantage/policy loss、weight syncer、runner、通用DCP manager和launcher。FSDP2仅有上述默认兼容开关：若保持原硬编码`true`，actor会在policy收到replay前把FP32 chains/proprio/action递归量化为BF16，既触发严格dtype失败，也破坏old/new logprob；因此这是有直接证据的必要通用修正，不是Fast-WAM数学特判。

三组检查足够：

1. **P1离线组合**：unit/synthetic测试覆盖adapter、module/replay schema、schedule/old-new、export和wiring；另用真实checkpoint进程覆盖builder、canonical tree、官方B1、真实B2/permutation和real replay。两类证据分开记录。
2. **P2真实runner**：独立`lr=0`与`lr=1e-6/noise=0.1`各一步，192步/8 chunks；记录wrap、grad/update/frozen hash、sync、ratio/clip/KL、GPU/RAM。社区已实证plain-BF16 rollout与FSDP-BF16 actor的小数值差会被高维chunk logprob放大；若lr=0在optimizer前已有实质clip/ratio漂移，必须停下修数值一致性，不能用actor自产old掩盖。
3. **P3生命周期**：fresh-process DCP resume恢复step/model/optimizer/scheduler/RNG，首次rollout前同步；no-dist导出；strict schema；resumed RLinf↔exported official固定fixture parity。10-step学习趋势是P3之后的尾项。

## 26. 实施就绪度与已冻结的最后三项

主体模型、数据、概率、policy、registry、worker shim、FSDP布局、DCP/export和验收边界已经达到可编码粒度；无需继续逐helper开会。用户已经确认：

1. **bucket device**：设计时首选GPU bucket；2026-07-18 本机真实 runner 在 CUDA IPC 的 `pidfd_getfd` 上被容器拒绝，因此当前落地为 RLinf 原生 CPU-staged bucket 512 MiB。只改变传输层，不改变同步张量、bucket算法或模型数学。
2. **固定eval latent**：`eval_seed=0`生成一个官方singleton latent并沿B broadcast；只用于eval，train仍为每条trajectory独立生成随机量。
3. **P2资源表**：固定采用23.3的`16 env / group8 / rollout_epoch1 / global128 / micro2 / model-forward2 / actor-rollout offload关闭 / env offload开启`；OOM时再一次只调资源旋钮。

三项已经冻结，I0/I1主体已实施到独立worktree/venv；真实checkpoint的official B1、B2 permutation、train replay和action-only inventory均已通过，joint venv 50项测试及静态检查通过。P2真实一步 runner 已通过，正式100-step训练已启动；当前实际资源方案为4并发env、rollout_epoch16、group4、rollout offload开启、CPU-staged bucket。canonicalization已采用非注册alias；是否提高forward chunk、启用checkpoint/offload只能由真实资源与数值证据决定，不再预设第二套路径。

## 27. 2026-07-18 正式训练效果、实现与换任务审计

本节是当前实验判断的权威增量，覆盖第23、26节中尚未经过真实 runner 验证的早期资源假设。只读现场、运行目录的 resolved config、官方 Fast-WAM/RoboTwin、当前 RLinf pin、成功 π0 GRPO 和社区 Fast-WAM×RLinf 共同作为依据。

### 27.1 当前训练不是崩溃，而是任务饱和且缺少独立评估

- run：`/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260718_020324-robotwin_adjust_bottle_grpo_fastwam_a800_2gpu`。
- 2026-07-18 09:12 CST 只读刷新：已完成 step 24，step 25 rollout 进行中；driver/Ray 存活，无 traceback、OOM 或 OOM kill；step10/20 DCP 已按计划产生。
- step1–24 的 train rollout `success_once` 总均值约 `97.14%`；前10步约 `97.97%`，最近10步约 `96.56%`。这不是固定seed deterministic eval，不能据此证明 checkpoint 变好或变差。
- 24步中 step 2/4/13/19/20 为全成功并产生零梯度；其余有效步的 `ratio_abs`、KL和clip总体很小，说明 replay/old-new 概率链没有出现明显断裂证据。
- 资源峰值：GPU0/1=`63036/62511 MiB`，cgroup RAM=`241932/245760 MiB`（98.44%），但 `oom=0/oom_kill=0`；当前两卡显存均衡，主机内存余量窄。

Fast-WAM论文 Table 3 中 `adjust_bottle` 为 clean/randomized=`100/100`。本项目即使把 RoboTwin 官方400步上限缩到192步，仍约97%成功，进一步说明该任务不适合判断GRPO是否能学涨。

对二元成功奖励、组大小为 `g`，组内至少同时出现成功和失败的概率是 `P(mixed)=1-p^g-(1-p)^g`。它在 `p=0.5` 最大，但“适合GRPO”不要求精确50%；约30%–80%通常都有可用对比信号。当前 `p≈0.9728,g=4` 时，每组mixed概率约10.4%，16组/step只期望约1.67个有效组；单独把同样64条轨迹改为group8并不能解决饱和。

### 27.2 当前、成功π0和社区成功Fast-WAM的配方差异

| 项 | 当前 Fast-WAM RoboTwin | 成功 π0 GRPO | 社区成功 Fast-WAM LIBERO |
|---|---:|---:|---:|
| env × rollout_epoch | `4×16` | `16×16` | `32×8` |
| trajectories / RL step | 64 | 256 | 256 |
| group size / groups | `4 / 16` | `8 / 32` | `8 / 32` |
| actor transition samples | 512 | 1024 | 13312（LIBERO 520/10） |
| global batch / update epoch | `128 / 1` | `512 / 2` | `64 / 1` |
| sample presentations / step | 512 | 2048 | 13312 |
| LR / noise | `5e-6 / 0.3` | `5.6e-6 / 0.5` | `2e-5 / 0.3` |
| group-std normalization | 开，当前pin硬编码 | 开 | 成功run关闭 |
| fixed deterministic eval | 无 | 无 | base/step5/step10 |

训练量比π0和社区少是次要风险，但当前首要瓶颈仍是97%饱和：提高LR会在极少有效组上放大更新，社区报告也显示饱和策略在更激进RL后可能遗忘。首轮不同时更换任务、LR和advantage定义。

### 27.3 逐层实现审计结论

调用链为：launcher/Hydra → registry/builder → RoboTwin adapter → conditioning/scheduler → ODE/SDE rollout chain → tensor replay → FSDP actor/通用GRPO loss → bucket sync/DCP/export。

1. **launcher/config**：复用RLinf标准driver；保存命令、resolved config和资源监控。当前服务器resolved config是实验真相，本地旧YAML不能替代。
2. **registry/worker/FSDP三个小diff**：Fast-WAM显式opt-in；旧π0路径保持原行为；`cast_forward_inputs=false`防止FP32 chain在actor入口被BF16量化。
3. **builder/adapter**：官方checkpoint/stats、三相机顺序、PIL bilinear、`[-1,1]`、14D qpos归一化、action反归一化和prompt与官方deploy一致。
4. **scheduler/conditioning/ODE-SDE**：官方shifted时间网格、H/N/S=`32/24/10`、VAE/proprio/video-prefill共核；训练逐样本随机initial/SDE noise，eval才广播官方singleton latent。
5. **rollout/replay**：保存完整 `[B,11,32,14]` chain，只重算真实 `x_k→x_{k+1}`，不从physical action反推，不合成old logprob；通用RLinf在chunk维聚合loss。
6. **数值证据**：官方B=1、B=2 permutation、rollout→actor same-weight replay均为零误差；真实runner有非零梯度且ratio/KL/clip正常。未发现能解释“不涨”的P0概率链缺陷。

仍需保留的工程风险：fresh-process DCP resume、真实DCP export后官方loader parity、非零更新后的action expert delta与冻结子树hash。`gradient_checkpointing`当前未实现Fast-WAM显式能力，不能只改YAML开启。

### 27.4 两个有依据但不能混在本轮里的算法选择

- **proprio**：当前action-only不是无依据遗漏。官方SFT训练DiT+proprio；社区RL虽然把proprio设为trainable，但actor replay复用了rollout阶段已经detach的context，实际成功梯度路径仍是action-only。若要真正训练proprio，必须在actor replay重新运行proprio encoder，不能只切`requires_grad`。作为新任务后的独立ablation处理。
- **no-std GRPO**：当前pin在advantage中始终除组内std；社区成功run用`grpo_norm_by_std=false`，且把开关贯穿env/actor。仅在YAML写字段不会生效。它是应补齐的可控复现差异，但不应与首次换任务同时修改。

### 27.5 域随机化的准确结论

当前pin的基础 `env/robotwin_adjust_bottle.yaml` 片段确实默认开启了RoboTwin随机背景、杂物、桌高和光照；但当前formal顶层配置又显式覆盖为clean：`random_background=false`、`cluttered_table=false`、`clean_background_rate=1`、`random_table_height=0`、`random_light=false`、`crazy_random_light_rate=0`。所以答案是“RLinf示例fragment默认开，但本次resolved run没有开”；这些字段都能在配置层修改，并非框架硬编码。

Fast-WAM论文总体 clean/randomized=`91.88/91.78`，randomized并非所有任务都更低；例如`move_stapler_pad=77/64`、`open_microwave=62/45`下降明显，但`hanging_mug=58/62`、`pick_diverse_bottles=80/85`反而上升。因此先换clean任务验证学习，再把randomized作为第二个单变量，不把任务难度和域偏移同时引入。

官方 vendored `demo_randomized.yml` 的对应字段是：background/clutter/light开启，`clean_background_rate=0.02`、`random_table_height=0.03`、`crazy_random_light_rate=0.02`；head-camera distance保持0。

### 27.6 首选新任务与代码改动面

论文较低成功任务包括：`open_microwave 62/45`、`hanging_mug 58/62`、`turn_switch 61/59`、`place_can_basket 71/69`、`move_stapler_pad 77/64`、`pick_diverse_bottles 80/85`。

服务器只读核对发现，当前RLinf train/eval seed JSON中只有 `move_stapler_pad` 同时具备现成成功seed（train 1000、eval 260）；其他更低任务缺seed。第一项建议因此是：

- `move_stapler_pad` clean。用户拍板首轮保持当前正式配置的 `192=24×8`，不因任务切换同时改变 horizon；`move_stapler_pad` 与 `adjust_bottle` 的 RoboTwin 官方 step limit 都是400，因此这是严格的单变量任务对照，而不是声称复现论文完整400步评估。
- 复制新增 `env/robotwin_move_stapler_pad.yaml`，只改 `task_name`。
- 新增顶层 `robotwin_move_stapler_pad_grpo_fastwam_a800_2gpu.yaml`，从服务器当前 resolved formal 对应源配置复制，只改 train/eval env 引用与实验名；三处step limit继续统一为192，其他训练、资源和随机化参数原样保留。
- model YAML、checkpoint/stats、builder、adapter、Flow-SDE core、actor/loss/sync/DCP均不改；`fastwam.task=robotwin_uncond_3cam_384_1e-4`是官方模型配置名，不是环境任务名。
- launcher增加显式config-name/task选择，或新增轻量模式；不复制policy实现。

`turn_switch`在论文上更接近50%，但当前缺train/eval成功seed；先补seed会引入额外环境数据变量，所以不作为第一项。

### 27.7 下一轮实验判断顺序

1. 用户决定中断饱和的 `adjust_bottle` run，直接切换到 `move_stapler_pad` clean；旧run保留日志和checkpoint，不改写产物。
2. 新任务正式配置完整保留当前配方：4 env×rollout_epoch16=64 trajectories、group4、global128/micro2、noise0.3、lr5e-6、192步、clean随机化、rollout/env offload和CPU bucket。这样首轮只改变任务名，不同时改变 horizon、并发、样本量或优化器。
3. 该任务按论文clean `p≈0.77` 估算，64 trajectories/group4期望约10.9个mixed groups；相较 `adjust_bottle` 更有GRPO组内差异。论文值来自完整官方评估，当前192步下的真实base成功率必须由新run现场测量，不能预设为77%。
4. 若mixed groups充分但固定eval仍不涨，再单变量移植/比较社区`grpo_norm_by_std=false`。
5. 之后再比较 `5e-6` 与社区有依据的 `2e-5`；不在饱和`adjust_bottle`上先升LR。
6. randomized作为clean跑通后的第二阶段；proprio训练作为独立实现/ablation。

### 27.8 `move_stapler_pad` 配置落地（2026-07-18）

- 服务器新增 `examples/embodiment/config/env/robotwin_move_stapler_pad.yaml`：相对 `robotwin_adjust_bottle.yaml` 只改 `task_config.task_name`。
- 服务器新增 `examples/embodiment/config/robotwin_move_stapler_pad_grpo_fastwam_a800_2gpu.yaml`：相对当前 `adjust_bottle` formal只改两条env defaults和`experiment_name`。
- launcher保持原 `smoke|train` 默认行为，并新增可选第二参数`config-name`；执行前检查对应YAML存在。
- Hydra compose/resolve已验证 train/eval task均为`move_stapler_pad`，三处step limit均为192，数量关系仍为4×16=64 trajectories、64/4=16 groups、64×(192/24)=512 actor transitions；模型、adapter、Flow-SDE、actor/loss/sync/DCP均无改动。
- 社区 Fast-WAM LIBERO 配置每步为32 env×8 rollout_epoch=256 trajectories、group8=32 groups；本配置64 trajectories是其1/4。两者任务horizon不同，不能仅按trajectory数等同计算总actor样本。

### 27.9 Git与代码归档方案

- 不使用旧 `RLinf_wamppo_backup_20260714_step57_lastdcp40`：现场确认它不是Git仓库，只是历史快照。
- 当前服务器 `/root/autodl-tmp/RLinf_fastwam_rlinf` 的 `feat/fastwam-robotwin-grpo` 是实现真相；保存前先把服务器实际formal YAML同步回本地镜像，避免归档旧的16×4/group8配置。
- 跟踪：首版GRPO的15个新增文件与3个兼容小改（含launcher/monitor），以及第28节PPO增量的Fast-WAM core/policy/builder/export、三份新配置、PPO launcher和测试；同时跟踪source lock、环境复现说明和小型fixture。不跟踪venv、模型、checkpoint、assets、数据、日志、TensorBoard/Ray、视频、cache和编译产物。
- 推荐提交拆分：核心policy/Flow-SDE与测试；RLinf wiring/FSDP；实验配置/launcher/monitor；文档与证据。官方Fast-WAM仓库保持独立并锁commit，不复制5B模型代码和权重进RLinf Git。

## 28. 2026-07-19：下一版GRPO参数与Fast-WAM PPO设计及落地

本节先记录参数和PPO实现的设计依据，随后记录2026-07-19已经落地到本地镜像与服务器工作区的实现。现役旧GRPO训练未停止、未重启；新增配置和代码只供后续新进程使用。

### 28.1 参数证据的优先级不能只分成两类

后续参数选择按以下四层取信，不把任何一份参考配置整套照搬：

1. **Fast-WAM模型固有合同**：`H/N/S=32/24/10`、官方shifted scheduler、action expert/MoT前向、三相机384输入、Flow-SDE随机量与replay，优先官方Fast-WAM；社区Fast-WAM只作RLinf拼接和已跑通数值证据。
2. **RLinf通用算法合同**：GAE/PPO loss、clip/value clip、group advantage、FSDP actor、optimizer/sync/DCP，优先当前RLinf pin与本机已跑通的π0 PPO/GRPO。
3. **Fast-WAM已验证训练配方**：`noise_level`、Fast-WAM学习率候选、no-std正向证据，参考社区真实成功run，但不能把其LIBERO horizon、4卡布局或未验证PPO小配置当成本机默认。
4. **本机资源参数**：并发env、`micro_batch_size`、`model_forward_batch_size`、offload、FSDP2、CPU bucket和checkpoint频率，优先本机两张A800和245.8GB cgroup的真实峰值；π0和社区硬件不能覆盖现场证据。

因此，`update_epoch`和GRPO std属于算法/估计器参数，不是Fast-WAM架构参数；`micro_batch_size`和`model_forward_batch_size`也不是通用算法参数，而是模型×硬件资源参数。

### 28.2 GRPO数量关系与下一版候选

| 配方 | trajectories/step | group / groups | transitions/trajectory | actor transitions | global/micro | optimizer updates | update epoch | group std | LR/noise |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 当前Fast-WAM RoboTwin | `4×16=64` | `4 / 16` | `192/24=8` | 512 | `128/2` | 4 | 1 | on | `5e-6/0.3` |
| 已跑通π0 RoboTwin | `16×16=256` | `8 / 32` | `200/50=4` | 1024 | `512/32` | 4 | 2 | on | `5.6e-6/0.5` |
| 社区成功Fast-WAM LIBERO | `32×8=256` | `8 / 32` | `520/10=52` | 13312 | `64/4` | 208 | 1 | off | `2e-5/0.3或0.6` |
| 下一版π0对齐候选 | `4×32=128` | `4 / 32` | 8 | 1024 | `512/2` | 4 | 2 | 首轮on | `5e-6/0.3` |

`4 env×32 rollout_epoch`是合法而且有依据的折中：并发仍为4，顺序采样128条轨迹，unique actor transitions恰好与已跑通π0的1024相同；在成功率约45%时，单step二项成功率的95%波动由约±12.2pp降为±8.6pp。代价是rollout墙钟时间和CPU trajectory/replay缓存大致翻倍；并发显存不应成倍增加，但当前DCP保存已接近99% RAM，所以首个checkpoint仍有真实OOM风险。

`4×32 + group8`在当前RLinf中不是合法配置。启动校验要求：

```text
total_num_envs / env_worker_world_size / pipeline_stage_num
  能被 env.train.group_size 整除
```

当前env只放在一个worker，`4 % 8 != 0`；`rollout_epoch`是顺序批次，不能把两个不同reset/seed的epoch拼成同一个8-sibling GRPO group。要用group8至少需要8个并发env；此前8并发已把RAM推到极窄余量，因此首选保持group4。当前成功率附近，group4的mixed概率约86.7%，32个group预计约27.7个mixed group，信号并不稀缺。

这里必须把五个不同层次的量分开，不能只比较`optimizer.step`次数：

1. `trajectories`和`actor transitions`决定每个RL step看多少新数据；下一版Fast-WAM与π0都是1024个unique transition。
2. `group_size/groups`决定GRPO组内相对优势；Fast-WAM的`group4×32 groups`与π0的`group8×32 groups`都保留32个group，但组大小不同。
3. `update_epoch`决定同一批rollout被完整复用几轮；设为2才与π0一致。
4. `global_batch_size`是两张actor GPU合计的一次有效梯度batch，决定一次`optimizer.step`聚合多少transition；它不是trajectory数，也不是每卡batch。
5. `micro_batch_size`和`model_forward_batch_size`主要控制单次前后向/推理的资源峰值，不应拿来替代算法batch或update epoch。

公式为：

```text
T = 每个RL step的flattened actor transitions
G = actor.global_batch_size
E = algorithm.update_epoch
W = actor world size
m = actor.micro_batch_size

optimizer.step / RL step = T / G × E
每卡gradient accumulation = G / (m × W)
```

因此，若目标是优先对齐已跑通π0的通用训练动态，下一份独立formal配置应为：

```yaml
env.train.total_num_envs: 4
env.train.rollout_epoch: 32
algorithm.group_size: 4
env.train.group_size: ${algorithm.group_size}

actor.global_batch_size: 512
actor.micro_batch_size: 2
actor.model.model_forward_batch_size: 2
rollout.model.model_forward_batch_size: 2
algorithm.update_epoch: 2

# 首轮保持当前代码硬编码的std-on；此时不写尚未接通的伪配置键。
# no-std分支实现后，才新增并显式设置 algorithm.grpo_norm_by_std。
actor.optim.lr: 5.0e-6
actor.model.rl.noise_level: 0.3
runner.save_interval: 10
```

其中：

- `global512/update2`使Fast-WAM与π0都成为`1024/512×2=4`次optimizer update、2048次transition presentation；这比先前讨论的`global128/update1`更完整地对齐π0。先前方案只有8次小batch更新和一次样本使用，属于偏工程吞吐的选择，现由本结论覆盖。
- RLinf先按`global_batch/world_size`切片，再按micro batch切片，并在每个micro batch才搬GPU；因此把global从128升到512不会把512个transition一次放入显存，单次activation峰值仍主要由`micro=2`决定。它把每卡gradient accumulation从32提高到128，减少Adam更新频率并降低单次梯度噪声。
- 风险不是显存突然变成4倍，而是一次optimizer step要连续累计128个micro batch，墙钟更长，BF16深累积的数值稳定性需要一步实跑确认；若实测不稳，`global256/update2`可作工程折中，但它每RL step有8次optimizer update，不能称为π0等价。
- std首轮保持on，是为了只改变rollout数并延续RLinf/π0默认。社区no-std是高价值第二个实验变量；当前pin必须增加真实配置分支，不能只在YAML写一个无效字段。
- `micro=2`来自本机Fast-WAM BF16/FSDP2实测；π0的32和社区4卡Fast-WAM的4都不能直接搬。
- `model_forward_batch_size=2`是Fast-WAM policy内部的模型前向分块：一个logical batch先统一生成prompt conditioning、initial latent、`k`和SDE noise，再每2个样本执行VAE/video cache/action denoise或actor replay。chunk不能重采样；该参数只控制Fast-WAM前向峰值，不改变trajectory数、global batch、梯度累积或optimizer update，π0没有同名等价项。
- 新配置保持旧formal的`save_interval=10`，以满足“相对旧formal严格只改四个字段”的单变量合同；它不改变算法更新，只决定DCP触发频率。

若目标从“可归因”改为“尽快贴近社区正向配方”，可在另一份独立配置中同时比较`grpo_norm_by_std=false + lr=2e-5`；不得把这个三变量版本的结果归因给单一参数。

### 28.3 PPO分歧：不是head挂载位置，而是critic feature取点

各实现的`ValueHead`最终都作为policy/model的顶层可训练子模块，以便FSDP、独立`value_lr`、同步和DCP发现它。真正分歧是从主网络哪一层抽特征：

| 实现 | feature取点 | critic语义 | 现有证据 |
|---|---|---|---|
| RLinf π0 PPO | 融合后的action suffix hidden，action output projection之前，`[B,H,1024]`后mean | action/noise-conditioned | 本机PPO全链跑通 |
| π0.5可选分支 | observation-side VLM/image-language token，`[B,L,2048]`池化 | 更接近`V(s)` | RLinf参考实现 |
| Motus PPO | 同时支持action token和VLM token；本机formal实际选`value_after_vlm=true` | 当前formal更接近π0.5 | PPO backward跑通，效果不作为正向基线 |
| LaWAM PPO | 多模态VLM后的flow-query placeholder token，flow DiT之前 | observation-side | B=1 PPO工程链跑通 |
| 社区Fast-WAM PPO | `video_expert.pre_dit()`的`video_pre["tokens"].mean(1)` | k-independent但语义浅 | 仅小型PPO验证；正向学习证据来自GRPO |

社区Fast-WAM的pre-DiT token只是当前观测帧VAE latent经过patchify/Conv3d后的video patch embedding；文本context单独保存，proprio/text/action要到后续DiT/MoT attention才融合。因此它虽与denoise k无关、old/new value容易一致，却没有融合任务文本、proprio或world/action信息；社区代码本身也把它标为较弱critic feature并推荐GRPO。

用户倾向采用社区Fast-WAM与π0.5一侧的observation critic，不让value读取action latent。结合官方Fast-WAM实际接口，首版推荐进一步收敛为：

```text
image + text + proprio
  → official build_action_conditioning()
  → official model.mot.prefill_video_cache(...)
  → video_kv_cache[-1]["v"] [B,120,3072]
  → token mean [B,3072]
  → ValueHead 3072→1024→512→256→1
  → FP32 value [B,1]
```

准确语义是“最后一层video value-cache的token mean”，不是最终video hidden。RoboTwin三相机拼图为`[B,3,384,320]`；官方VAE与`(1,2,2)` patchify后得到120个video token；Wan2.2 video expert为24 heads×128维，所以cache V是`[B,120,3072]`。运行时必须断言token数、hidden dim和finite，而不是把3072静默硬猜成任意宽度。

选择这个取点的依据：

1. 社区raw `video_pre["tokens"]`只是VAE latent经patchify/Conv3d后的视觉embedding，尚未进入MoT，也没有融合文本/proprio；可保留为后续消融，但不作为首版默认。
2. official proprio先append到text context；video prefill的每层都做video self-attention和对该context的cross-attention。最后一层cache V的输入已通过前29层，因此比raw pre-DiT token更接近π0.5/Motus的post-VLM observation feature。
3. 官方mask没有video→action边；该feature严格不读取action latent、denoise timestep、`k`或SDE noise，语义更接近`V(s)`，bootstrap也不带selected-k随机性。
4. 取V而非K，因为V没有RoPE，token mean有清晰含义；K已施加位置旋转，不宜直接均值。
5. 理论上最终video token还多一层融合，但官方`prefill_video_cache()`不返回最终`x`。为了多这一层去修改锁定Fast-WAM仓库或复制整段30层prefill，侵入和漂移风险不值当；last-layer V已经存在且是action expert真实读取的observation memory，不增加模型前向。

设置并冻结`detach_critic_input=true`：当前`build_action_conditioning()`本就处于`no_grad`，video expert、T5和proprio encoder保持冻结；critic loss只训练value head，actor loss仍只训练action expert。rollout old value、actor replay new value和bootstrap都由同一image/text/proprio重建该feature，不把feature写进replay，也不另写简化denoise路径。

### 28.4 PPO逐文件实现合同

1. **PPO top-level YAML与launcher**（调用链最上游：用户命令→Hydra/Runner；下游builder和RLinf workers）：新增独立Fast-WAM PPO smoke/formal配置和清晰的PPO launcher，不把GRPO launcher变成含糊的多算法分支。设置`adv_type=gae`、`loss_type=actor_critic`、`group_size=1`、`critic.use_critic_model=false`；actor/rollout两边都显式收到相同的`add_value_head`和critic feature合同。
2. **`model/fastwam_robotwin.yaml`**（上游top-level插值；下游builder）：GRPO默认继续`add_value_head:false`；增加默认关闭的critic合同字段。PPO actor显式覆盖true，rollout必须用`${actor.model.add_value_head}`等插值同步，不能只给actor创建head，否则bootstrap和weight-sync schema不一致。
3. **`builder.py`**（上游registry/model YAML；下游policy/FSDP）：`build_fastwam_policy()`解析`add_value_head`、只允许首版`video_cache_last_v_mean`和`detach_critic_input=true`。先保持现有`freeze_action_only(model)`，PPO再在policy顶层创建RLinf `ValueHead(3072,(1024,512,256),1,relu,bias_last=True)`并移到模型device/dtype；GRPO完全不创建`value_head`属性，不能写`None`，避免HF worker的`hasattr`误触发bootstrap。官方model inventory和deploy schema仍只审计Fast-WAM本体。
4. **`fastwam_rl.py`**（上游policy tensor输入；下游官方conditioning/MoT/Flow-SDE）：`ActionConditioning`增加可选`observation_value_features`；`build_action_conditioning(return_value_features=False)`只在PPO从现有last-layer cache V生成`[B,3072]`；`recompute_logprob(..., return_value_features)`把同一次conditioning的feature带回actor。`predict_action_velocity()`、`flow_sde_rollout()`、H/N/S、scheduler、Flow-SDE mean/std、chain和真实selected transition全部零改动。
5. **`fastwam_policy.py`**（上游HF rollout/actor worker；下游Fast-WAM core）：构造时只在传入非空head时注册`self.value_head`；新增统一value helper，显式detach、匹配head dtype/device并输出FP32 `[B,1]`。`predict_action_batch(compute_values=True)`从每个model-forward chunk的conditioning生成并拼接`prev_values`；`default_forward(compute_values=True)`重建同一conditioning并返回当前`values`。replay schema仍只存image/text context/mask/proprio/chain/k，不新增feature缓存；GRPO默认输出仍为`prev_values=None`。
6. **`export.py`**（上游RLinf DCP；下游官方Fast-WAM deploy checkpoint）：DCP必须保留value head供PPO resume；strict deploy export则把顶层`value_head.`加入允许忽略的非官方前缀，继续只导出官方Fast-WAM schema并验证动作parity。
7. **集中测试**（上游上述所有symbol；下游一步PPO runner）：扩展现有Fast-WAM测试而不为每个小函数设置单独gate。覆盖GRPO无head/原hash不变、feature对action latent/k/noise不变且对image/text/proprio敏感、same-weight old/new value一致、critic/actor梯度路由、bootstrap、optimizer双LR组、bucket同步head hash、DCP resume和deploy排除head；最后只做一次两卡BF16真实PPO一步smoke。
8. **RLinf通用层零diff**：HF worker现有`prev_values/bootstrap_values`、EnvWorker trajectory、GAE、clipped actor+Huber value loss、按`value_head`命名使用`value_lr`、FSDP2、CPU bucket sync和DCP均复用。实施时只核对head确实进入optimizer、sync与DCP，不因尚未观察到问题预改worker/loss/runner。

old/new/bootstrap合同：behavior rollout用已同步head产生old `prev_values`；actor用同一observation重建feature并以当前head产生new `values`；final observation以同一路径产生bootstrap。首次更新前old/new应在BF16容差内一致；更新后由PPO value clip比较。当前HF bootstrap可能为了取得value仍完整生成一次action，这是性能冗余而非语义错误，首版不为此修改通用worker；后续再单独考虑`predict_value_batch`优化。

### 28.5 PPO配置参数来源

PPO外层算法优先复用已跑通π0，Fast-WAM数值和资源项复用当前GRPO：

| 参数 | 首版建议 | 依据 |
|---|---:|---|
| `adv_type/loss_type/group_size` | `gae/actor_critic/1` | π0、Motus、社区Fast-WAM PPO一致；PPO不用GRPO组优势 |
| `normalize_advantages` | true | π0 PPO；这是GAE batch归一化，不是GRPO group std |
| `update_epoch` | smoke 1，formal 2 | formal直接参考已跑通π0；社区Fast-WAM PPO只是小型验证 |
| `gamma/gae_lambda` | `0.99/0.95` | π0、Motus、社区PPO一致 |
| policy/value clip | `0.2/0.2` | π0和社区PPO一致 |
| Huber delta | 10 | π0、Motus、社区PPO一致 |
| reward/logprob/entropy | chunk/chunk/chunk | 当前Fast-WAM接口；entropy bonus为0，entropy聚合暂不影响梯度 |
| `filter_rewards` | false | PPO/GAE不丢全成功/失败轨迹；π0和社区PPO一致 |
| actor/value LR | `5e-6/1.1e-4` | actor接近π0的5.6e-6且等于当前Fast-WAM；value LR直接取π0/Motus |
| Adam/WD/clip grad | `.9/.95/1e-8/.01/1.0` | π0、当前Fast-WAM和社区正向GRPO共同支持 |
| noise / H/N/S | `0.3 / 32/24/10` | 当前Fast-WAM与社区正向GRPO / 官方Fast-WAM |
| global/micro/model-forward | smoke `32/2/2`；formal `512/2/2` | formal的global512与新GRPO、已跑通π0对齐；smoke只有32个真实transition，故global32恰好完成一次更新。`micro/forward=2`来自本机Fast-WAM资源实测，二者保持相同并发峰值合同 |
| env/rollout offload | true/true | 当前Fast-WAM能运行的资源布局 |
| actor/FSDP CPU offload | false/false | 避免进一步占用已接近上限的RAM |
| FSDP | FSDP2/full-shard/root wrap | 当前Fast-WAM MoT实现与社区结构证据 |
| gradient checkpoint | false | 当前wrapper未接通显式能力；不能只改YAML |
| BF16 / cast inputs | BF16 / false | 模型前向BF16；FP32 Flow-SDE chain/logprob不在FSDP入口被量化 |
| CPU bucket 512MiB | 保持 | 本机pidfd/CUDA IPC限制 |
| save/val interval | smoke `1/-1`；formal `10/-1` | smoke必须落step1 DCP；formal与新GRPO保持每10步DCP。固定seed eval仍串行执行 |

PPO formal现与新GRPO共享采样和有效batch合同：`4 env×rollout_epoch32=128 trajectories`、`128×8=1024 transitions`、`global512/update2=4`次optimizer update和2048次样本呈现。PPO与GRPO的算法差异只保留在`group1 + GAE + actor_critic + value head/value_lr`，不再额外改变数据量或更新动态。

PPO smoke保持与formal相同的4路环境并发、两卡placement、`micro2`、`model-forward2`和offload布局，只把顺序采样降为`rollout_epoch1`：`4×1×8=32 transitions`。因此smoke必须使用`global32/update1`，才能由当前RLinf EmbodiedFSDPActor完整切成一个global batch并完成一次真实optimizer update；若仍写global512，rollout batch小于一个global batch，不满足actor切分合同。这是smoke数据预算缩减，不是模型前向或并发萎缩。

### 28.6 已拍板并落地的设计项

1. 下一版GRPO采用π0对齐版本：`4 env×32 epoch`、group4、global512、micro2、model-forward2、update2、std-on、lr5e-6、noise0.3。该结论覆盖本节早先的global128/update1候选；no-std和lr2e-5继续作为后续单变量/独立多变量实验。
2. PPO首版value feature采用`video_kv_cache[-1]["v"].mean(1)`的observation-side critic；不采用selected-k action hidden，也不直接采用社区raw pre-DiT patch。社区raw feature与最终video token只作为后续消融候选。
3. PPO从官方Fast-WAM base冷启动，`runner.resume_dir=null`、`runner.ckpt_path=null`，模型权重继续由官方release checkpoint路径加载；不从当前GRPO DCP暖启。
4. PPO smoke固定`4×1` trajectories、global32/update1，保持4路并发并完成恰好一次optimizer update；formal固定`4×32`、global512/update2，每个RL step四次optimizer update。formal与新GRPO都使用1024个unique transition、2048次样本呈现，便于把后续差异归因于GRPO与PPO本身。

### 28.7 2026-07-19实现落地与集中验收

调用链已经按本节合同实现：

1. `robotwin_move_stapler_pad_grpo_fastwam_a800_2gpu_pi0_aligned.yaml`相对旧formal经结构化diff确认严格只改四项：实验名、`rollout_epoch 16→32`、`global_batch_size 128→512`、`update_epoch 1→2`；旧formal保留不动。
2. 新增PPO smoke/formal YAML和独立`run_fastwam_robotwin_ppo.sh`。actor/rollout两侧均显式启用同一head合同；launcher复用现有资源监控，支持`smoke|train|formal`。
3. `model/fastwam_robotwin.yaml`默认仍为`add_value_head:false`；PPO顶层配置才覆盖为true，因此GRPO对象不注册`value_head`属性。
4. `builder.py`只在PPO创建顶层`ValueHead(3072→1024→512→256→1)`。一次独立审查发现原草案的FP32 head与`cast_forward_inputs=false`/FSDP2 BF16存在dtype不一致风险，已改为head参数跟随模型`torch_dtype`；rollout和actor都用BF16 head计算，wrapper再把标量value转FP32交给GAE/value loss。
5. `fastwam_rl.py`只在PPO显式请求时，从官方`prefill_video_cache()`的最后一层`v`校验并池化`[B,120,3072]→[B,3072]`；scheduler、H/N/S、Flow-SDE、chain、selected transition和logprob计算零改动。
6. `fastwam_policy.py`在rollout返回CPU FP32 `prev_values[B,1]`，actor replay重建同一observation feature并返回可训练`values[B,1]`；feature不写replay且显式detach。`export.py`只在官方deploy导出时忽略`value_head.*`，RLinf DCP仍完整保存head。
7. RLinf通用HF worker、EnvWorker、GAE/PPO loss、FSDP optimizer分组、bucket sync和DCP没有新增PPO专用diff；现有路径按顶层`value_head`命名使用`value_lr`并同步/保存它。

2026-07-19随后按用户拍板修正PPO外层预算：formal由`4×16/global128/update2`改为`4×32/global512/update2`，与新GRPO对齐为1024 transitions、4次optimizer update和2048次样本呈现；smoke由`4×4/global128`改为`4×1/global32`，保持4路并发但只做一个顺序epoch和一次真实更新。formal DCP间隔同步由20改为10。该修改不触及value feature、head、Flow-SDE、actor/loss/sync/DCP实现。

服务器联合环境已对修正后的三份配置重新做Hydra `--resolve`：smoke预算为`(4 trajectories, 32 transitions, 1 update, 32 presentations)`，PPO formal与GRPO π0对齐版均为`(128, 1024, 4, 2048)`；actor/rollout head与forward2合同一致。配置合同`7 passed`（额外锁定PPO↔GRPO与smoke↔formal严格diff），Fast-WAM集中测试`61 passed, 1 warning`；现役旧GRPO driver未被停止或重启。

服务器联合环境已完成但不启动训练的验收：两个launcher `bash -n`、Fast-WAM包与测试`compileall`、三份新配置Hydra `--resolve`均通过；resolved结果确认GRPO actor/rollout均无head、PPO两侧均有相同head/forward2；Fast-WAM集中单元测试`59 passed`。现役旧GRPO PID仍在。尚未冒充完成的是两卡真实PPO rollout→GAE→backward→sync→step1 DCP smoke，以及后续DCP resume；按用户约定由下一次单独运行验证。
