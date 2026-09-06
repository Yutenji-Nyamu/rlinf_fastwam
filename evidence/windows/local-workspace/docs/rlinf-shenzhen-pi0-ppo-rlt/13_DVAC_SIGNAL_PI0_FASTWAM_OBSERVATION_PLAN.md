# SZ current RLinf π0 与 official Fast-WAM：DVAC 信号观测计划

> 状态：**两侧default-off观测实现、focused tests与独立审查均已完成；π0已push，Fast-WAM已local commit；真实P0/P1尚未采集。**  
> 目的：在不改变策略动作与随机数状态的前提下，从两条 official runtime 记录同一定义的 flow-denoising endpoint variance，并把模型内的 horizon-position 结构、query-level 强度和局部 horizon 结构分开分析。  
> 结论先行：π0 侧属于“**小的模型信号改动 + 中等的现有 worker/writer 接线**”，不是重新实现算法；Fast-WAM 也能记录同一个 endpoint 几何量，但它不是 π0 的同构 runtime，原始方差数值不能直接横向比较。

## 1. 目标与非目标

### 1.1 本计划要回答

1. 在 SZ source-locked current RLinf π0 中，如何复用旧 AutoDL 的 DVAC telemetry，而不搬入训练侧 DVAC、R-only、GRPO 或其他后续算法增量。
2. 在 official standalone Fast-WAM 中，能观测到哪些 action denoising chain 量，哪些不能被说成 π0 的同一内部过程。
3. 如何用统一的 raw signal contract，离线拆成 `Position / Residual / S / I`，并与 query、视频帧、成功/失败和独立任务阶段对齐。
4. 如何用几十个 episode 得到第一批可解释数据，同时保持两套环境、依赖和结果目录隔离。

### 1.2 明确非目标

- 不实现 DVAC 在线 adaptive chunking，也不改变 `N_exec` / replan 策略。
- 不迁移旧 DVAC 训练加权、R-only、G35、GRPO、Fast-WAM GRPO 或 RLinf trajectory schema。
- 不把 Fast-WAM 集成进 RLinf；两条 official standalone/runtime 路径保持独立。
- 不用 variance 生成任务阶段标签，不把它称作已校准的不确定性、安全分数或因果重要性。
- 不比较两个 policy 的 raw `V` / `y` 绝对大小，也不把不同 RoboTwin tree 上同名 seed 当成 paired state。
- 本文不描述服务器当前 GPU、RAM、进程或网络状态；实施前必须现场刷新。

## 2. Source locks 与证据边界

### 2.1 深圳目标 runtime

| 对象 | source lock | 本计划用途 |
|---|---|---|
| [RLinf official](https://github.com/RLinf/RLinf/tree/7d07a4212ee6858cc333e1d4fab7a37256d1f839) | `7d07a4212ee6858cc333e1d4fab7a37256d1f839` | π0 current port 的唯一代码基线 |
| RoboTwin `RLinf_support` | `0008ae6800df9f75fc8de7098bacb01735fd8fd2` | π0 evaluator compatibility tree |
| π0 RoboTwin SFT checkpoint | revision `92684e50c8a3dcf13b76a06713e3152625967be1` | `adjust_bottle` common-task policy |
| [Fast-WAM official](https://github.com/yuantianyuan01/FastWAM/tree/7faa71108368fbb3b6885649f112af607427a2d4) | `7faa71108368fbb3b6885649f112af607427a2d4` | standalone Fast-WAM 唯一代码基线 |
| Fast-WAM vendored RoboTwin | `bf44be51cf5717a5595ce59447f2cf5263d2aa95` | Fast-WAM official evaluator tree |
| [Fast-WAM HF release](https://huggingface.co/yuanty/fastwam/tree/8eaceeb24c3cc92ff2a9c9a9d266a4941b836705) | `8eaceeb24c3cc92ff2a9c9a9d266a4941b836705` | checkpoint 与 dataset stats |
| Fast-WAM checkpoint | SHA-256 `776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63` | 已锁定的 official 权重 |
| Fast-WAM stats | SHA-256 `7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095` | action z-score normalization |

Fast-WAM 已有的 source/runtime 决策由 [official standalone plan](../fastwam-robotwin-rlinf-grpo/09_SHENZHEN_OFFICIAL_STANDALONE_PLAN.md) 与 [完成记录](../fastwam-robotwin-rlinf-grpo/11_SHENZHEN_FROM_ACT_TO_OFFICIAL_FASTWAM_NOTES.md) 提供；本文只增加“观测信号”规划，不把其历史执行状态当作新的现场事实。

### 2.2 旧 AutoDL 只作为增量来源

| 对象 | Git object | 使用边界 |
|---|---|---|
| π0 standalone telemetry base | `6d0db56bf26f972cd27fa29535f5eb939e80e5bf` | 旧 baseline |
| π0 telemetry increment | `61996e15cc7f5a32bd6012b61b20893d94636c82` | 本次主要复用对象 |
| 后续 DVAC training implementation | `052a2ee8902c51595caa997736f1ec76699ad8df` | 只作训练研究上下文，不迁移 |
| R-only child / base | `3061872e30cfb496eb296354d30274d35b66576e` / `145fa810…` | 后续算法上下文，不迁移 |
| RoboTwin control trace | `43696bbab85fef3dd98074c5ba0ccb90786d0e94` | 只在需要更细 π0 控制对齐时选配 recorder/hooks |

旧 telemetry 对 base 的精确增量是 **6 个文件，1099 insertions / 9 deletions**。其中模型内 capture 只有约 35 行；大部分代码量来自 writer、索引、join contract 与测试。因此真实判断是：

- **信号算法本身很小**；
- **current 集成不能只复制 YAML**；
- 两个新工具文件和 YAML 可按语义复用，三个已演进文件必须在 `7d07a421` 上重放，而不是盲目 cherry-pick。

### 2.3 一手资料

- DVAC official paper v1：[Denoising Tells When to Replan: Denoising-Variance Adaptive Chunking for Flow-Based Robot Policies](https://arxiv.org/html/2606.03847v1)，`arXiv:2606.03847v1`，2026-06-02。
- Fast-WAM：[paper](https://arxiv.org/abs/2603.16666)、[project](https://yuantianyuan01.github.io/FastWAM/)、[exact source](https://github.com/yuantianyuan01/FastWAM/tree/7faa71108368fbb3b6885649f112af607427a2d4)。
- 本计划检索到的 DVAC paper metadata、正文和 exact-title 结果均未给出 official code/project 链接；因此 signal definition 以论文公式为准，不声称存在可直接复制的官方实现。

## 3. 哪些是 DVAC 论文原定义，哪些是我们的分析

| 内容 | 来源 | 本计划如何使用 |
|---|---|---|
| `z_i = x_i - t_i v_i` clean-endpoint estimate | **DVAC 论文** | 两个 flow policy 的共同原始几何量 |
| 最近 `L` 个 denoising steps、对每个未来位置和 action dim 做 population variance，再按 dim 求和 | **DVAC 论文** | `V_L(q,h)` 的唯一主定义 |
| `V_total` 只是标量诊断；在线方法按阈值首次越界选执行长度 | **DVAC 论文** | 本次只记录前者，不实现后者 |
| 论文默认 `L=5`，并使用 rolling local scale；RoboTwin 报告 `H=50` | **DVAC 论文** | Fast-WAM 可做 `L=5` sensitivity；π0 只有 4 个 endpoint，不能伪造 L=5 |
| MOVING / OPERATING 由前/腕相机帧独立标注 | **DVAC 论文实验方法** | 我们同样要求 phase label 独立于 variance |
| `y=log(V+eps)`、horizon-position robust center/scale | **我们的离线分析** | 消除“某些 h 天然更大”的位置结构 |
| `R`、query scalar `S`、within-query pattern `I` | **我们的离线分解** | 区分整条 query 强弱与 horizon 内局部结构 |
| π0 与 Fast-WAM 用 `L_common=3` 做标准化结构比较 | **我们的跨模型协议** | 只比较 shape/rank/association，不比较 raw scale |

论文把 denoising variance 定位为**经验稳定性 proxy**，不是 calibrated uncertainty 或 safety estimate。本文所有图和结论沿用这一边界。

## 4. 统一 raw signal contract

设 query 为 `q`，action horizon 位置为 `h`，active action 维为 `d`，denoising step 为 `i`：

```text
z_i(q,h,d) = x_i(q,h,d) - tau_i * v_i(q,h,d)

V_L(q,h) = sum_d [ (1/L) * sum_{i in tail(L)}
                    (z_i(q,h,d) - mean_tail(z(q,h,d)))^2 ]

y(q,h) = log(V_L(q,h) + 1e-12)
```

约束：

1. 只用 active 14D action；padding 维不进入 variance。
2. `x/z` 保存在各 policy 的**模型原生归一化空间**。解码后的物理单位 action 另存，不混入 `V`。
3. 保存完整 chain，`V/y/R/S/I` 全部离线派生；未来修改分析不需要重跑模型。
4. π0 使用其 4 个实际 flow times；Fast-WAM 的 `tau_i` 是 `step_t_action / num_train_timesteps`，不能把 `0…1000` scheduler timestep 直接代入公式。

### 4.1 我们的 Position / Residual / S / I

位置统计必须分别在 `(policy, checkpoint, task, h)` 内估计：

```text
b_h     = median_q y(q,h)                         # Position center
s_h     = 1.4826 * MAD_q y(q,h)                  # Position scale
r(q,h)  = y(q,h) - b_h                           # raw residual
R(q,h)  = r(q,h) / max(s_h, scale_floor)          # standardized residual
S(q)    = mean_h R(q,h)                           # query-wide intensity
I(q,h)  = R(q,h) - S(q)                           # within-query horizon pattern
```

精确恒等式是 `y=b+r` 和 `R=S+I`。**不能**把标准化后的 `S/I` 写成 `y=b+S+I`。`scale_floor` 只用于数值稳定，必须写入 manifest，不能按结果调参。

### 4.2 跨模型可比与不可比

主比较用 `L_common=3`，因为两侧都真实拥有至少 3 个 endpoint；另报：

- π0：`L=2/3/4` sensitivity；
- Fast-WAM：`L=3/5`，其中 `L=5` 对齐 DVAC paper 默认。

能比较的是标准化 `R/S/I` 的分布、rank、时间/phase/outcome association。不能直接比较 raw `V/y`，原因包括：π0 与 Fast-WAM 的 normalization、flow schedule、step 数、H、checkpoint 和 simulator tree 都不同。

## 5. π0 current RLinf：适配量与接缝

### 5.1 current 与旧 telemetry 的差异

`7d07a421` 相对旧 base 已演进：

- `openpi_action_model.py` 增加 RTC、`model_actions` 与 RLT 相关重构，但普通 π0 denoising loop 的核心仍可在 `x_t_prev / v_t / Euler update` 周围插入只读 capture。
- HuggingFace worker 的 predict/evaluate 路径和 Env worker 的 eval metadata / `PolicyOutput` / `EnvOutput` 接口已重排。
- 新 YAML、独立 telemetry helper 和独立 test 在三方 merge 结构上是 clean addition；模型和两个 worker 属于双方都改过的 semantic replay。

因此 current port 预期仍是 **6 个逻辑文件：3 个现有文件 + 3 个新增文件**：

| 类别 | 文件 / symbol | 计划改动 |
|---|---|---|
| 现有 1 | `rlinf/models/embodiment/openpi/openpi_action_model.py::{sample_actions,predict_action_batch}` | 默认关闭；在 Euler update 前 detach 已有 `x/v/t`，返回 raw chain；普通 SFT eval only，RTC telemetry 非目标 |
| 现有 2 | `rlinf/workers/rollout/hf/huggingface_worker.py::{HuggingFaceRolloutWorker.__init__,predict,_predict_rollout_actions,evaluate,_merge_obs_batches}` | current API 下保留 model telemetry result，并交给 writer；不改动作值 |
| 现有 3 | `rlinf/workers/env/env_worker.py::{EnvWorker.__init__,_build_rollout_input_data,_build_eval_query_metadata,env_evaluate_step,evaluate}` | 生成 query/episode/env metadata 和 join key；保持 side-channel，不改 training trajectory schema |
| 新增 1 | `rlinf/utils/dvac_telemetry.py` | writer、manifest、rank shard、CSV/NPZ join contract |
| 新增 2 | `evaluations/robotwin/robotwin_adjust_bottle_openpi_dvac_eval.yaml` | 以 current official eval YAML 为父语义，只加 `enabled: false` telemetry block 与 source locks |
| 新增 3 | `tests/unit_tests/test_dvac_telemetry.py` | off/on 等价、公式、writer round-trip 等少量高信息量检查 |

无需改 actor、critic、loss、advantage、placement、rollout training batch 或 `rlinf/data/schema/embodied_trajectory_builder.py`。旧 training-weighting 文件不进入这个 branch。

### 5.2 π0 实际 chain 和控制边界

| 量 | 当前 common-task 计划值 |
|---|---|
| action horizon | `H=50` |
| active dims | `D=14`（模型张量总维 32，padding 排除） |
| denoising steps | `M=4`，`tau=[1.0,0.75,0.5,0.25]` |
| raw shapes / query | `x_chain [5,50,14]`；`z_endpoint [4,50,14]` |
| fixed chunk | 计划沿现有 official π0 eval 的 `C=50`，终止时可提前结束 |

RLinf 现有 tiled MP4 是 **query-boundary video**：旧 metadata 可精确表示 `video_pre_frame=query_idx`、`video_post_frame=query_idx+1` 及 tile/local env slot，但不能把 `h=0…49` 声称为 50 个逐动作视频帧。

若首批数据证明必须做更细的 π0 phase/horizon 对齐，才选配 `43696bba…` 中的 default-off RoboTwin control recorder/hooks。它用较慢 arm 的 TOPP progress 近似 `h`，并不提供精确 model waypoint lineage；只复用 recorder/hooks，不带入其后续行为或 step-limit 改动。

## 6. official Fast-WAM：可观测量与边界

### 6.1 action denoising chain

在 [`FastWAM.infer_action`](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/fastwam.py) 中：

1. `latents_action` 从本地 `torch.Generator` 采样，shape 为 `[1,H,D]`；
2. 每步得到 action flow velocity `pred_action`；
3. scheduler 以 `latents_action = sample + delta * pred_action` 更新；
4. 因而可在更新前记录 `x_i=sample`、`v_i=pred_action`、归一化 `sigma_i`，并计算 `z_i=x_i-sigma_i v_i`。

official RoboTwin resolved runtime 的关键量是：

| 量 | Fast-WAM |
|---|---|
| action horizon | `H=32` |
| active dims | `D=14` |
| denoising steps | `M=10` |
| raw shapes / query | `x_chain [11,32,14]`；`z_endpoint [10,32,14]` |
| execution / replan | 每次最多执行前 `C=24`，尾部 `h=24…31` 不执行 |
| scheduler compatibility | released checkpoint 使用 `sigma_shift=5.0`；不能静默退回 source 当前默认值 |

telemetry 必须记录 resolved sigma/delta schedule，而不是只写配置名。

### 6.2 不能直接等同于 π0 的部分

- official `infer_action` 先用当前观测帧建立 video KV cache，随后只 denoise action latents；本计划**没有**可称为“future-video denoising chain”的对应量。
- 不默认导出 video latent、KV cache 或 hidden states：它们不是 DVAC endpoint 定义所需量，且存储/同步开销大。
- Fast-WAM action 用 dataset stats 做 z-score，π0 normalization 不同；即使都取 active 14D，raw scale 仍不可比。
- Fast-WAM 每 query 使用现有本地 generator/seed 行为。telemetry 不增加任何随机采样，也不“修正”其 seed 方式。

### 6.3 Fast-WAM 预期插桩位置

预期 **4 个 runtime/config 文件 + 1 个 focused test**：

| 文件 / symbol | 计划改动 |
|---|---|
| `src/fastwam/models/wan22/fastwam.py::FastWAM.infer_action` | opt-in capture/return既有 `x/v/timestep/delta` 与 step 后 `x_next`；`z` 离线计算，不改变既有 update 顺序 |
| `experiments/robotwin/fastwam_policy/deploy_policy.py::{WorldActionRobotWinPolicy._infer_action_chunk,_fill_action_queue,step,reset}` | query/episode writer、action-slot 与视频 join、成功边界 |
| 新增 `experiments/robotwin/fastwam_policy/dvac_telemetry.py` | 与 π0 同字段语义的独立 writer/helper，不引入 RLinf dependency |
| `configs/sim_robotwin.yaml::EVALUATION.dvac_telemetry` | `enabled: false` 与输出/采样配置 |
| 新增 focused test | off/on action 与 generator state 等价、`z` 公式、writer round-trip |

不改 vendored RoboTwin，不改 Fast-WAM training/GRPO，不把 π0 worker 搬进来。若 official compile 路径改变了 Python-side capture 边界，首批仅支持当前 official non-compiled inference；不为未观察到的问题预设第二套实现。

## 7. Query、动作和视频帧对齐

### 7.1 Fast-WAM：当前 official 视频只到 query 级；新 run 可用官方开关取得 action 级帧

vendored evaluator 每次从 pending queue 取一个 action；`_base_task.take_action` 对每个 action 写一个
**pre-action frame**，若该 action 内成功，再写一个 terminal-success frame。但当前已成功 official run 的
resolved config 是 `skip_get_obs_within_replan=true`：同一 24-action replan 窗内不刷新 `now_obs`，所以
MP4 虽然每个高层 action 都写帧，画面实际是同一 query observation 的重复/低帧率表示。该现成视频只能
可靠用于 query/chunk-level phase 对齐，不能拿来解释每个 `h` 的执行画面。

若新的 telemetry run 显式使用 official 已支持的 `skip_get_obs_within_replan=false`，则每个高层 action
前都会 fresh `get_obs()`；在无早停时，query `q` 从 action slot `s=24q` 开始，实际执行到 exclusive
end `e`：

```text
standard pre-action video frames = [s, e)
terminal success frame           = e   # 仅成功发生在最后一次 action 时存在
```

通常 `e=min(s+24, episode_end)`。`h=0…e-s-1` 可与真实执行 action/frame 对齐；`h=24…31` 永远是该 query 的未执行 future tail，**不得伪造视频帧或任务阶段结果**。它仍可用于 model-internal horizon-position / `I` 分析。

`skip=false` 不增加 policy query 或改变 action queue，但会增加逐 action render/observation 开销；它必须写进
resolved config，并用同一设置做 telemetry off/on parity。不能把 `skip=true` 的旧成功视频与
`skip=false` 的新 telemetry run 当作逐帧 paired data。

### 7.2 π0：query-boundary 精确，horizon 内默认不精确

默认记录 query 输入三相机图、query index、episode/env slot、chunk 起止及 tiled video 的 pre/post query frame。由于一条模型 action 可经低层轨迹执行为多个 simulation frames，不能仅靠 tiled MP4 建立 `h ↔ physics frame` 一一对应。

因此第一批共同分析采用：

- query-level `S` ↔ query-boundary 图像、episode time、success-before/after；
- `I(q,h)` ↔ 模型 horizon position，只做代表 query frame strip；
- 不把 π0 的 `I(q,h)` 画成已精确对齐的逐 action task phase。

只有用户批准 optional control trace 后，才对少量代表 episode 增加近似 `h` / physics timeline。

### 7.3 Task phase 与 outcome

1. phase label 独立于 `V/y/R/S/I`，先看视频/状态定义标签，再做关联分析，避免循环论证。
2. 首版沿 DVAC paper 的 coarse semantics：`MOVING` 与 `OPERATING`；可另加 task-specific 标签，但不替代 coarse 列。
3. 保存 `phase_source`、annotator/version、confidence、frame/action range；不确定区间允许 `UNKNOWN/TRANSITION`。
4. success/failure 取 evaluator 的 episode outcome；另存 `first_success_action_slot/query`，不能由视频文件名猜测。
5. 跨 policy 的同名 seed/reset id 只作各自运行记录。两套 RoboTwin commit 不同，因此 episode 不做 paired test。

## 8. 产物 schema

建议根目录（实施时才创建）：

```text
/data/chenyiteng/results/dvac-observation/<run_id>/
├── run_manifest.json
├── traces/
│   └── rank*_part*.npz
├── queries.csv
├── episodes.csv
├── query_images/                 # lossless three-view input frames
├── videos/                       # evaluator 原视频或稳定引用
├── derived/
│   └── query_horizon.parquet     # 无 parquet 依赖时写 CSV
└── control_trace/                # π0 optional；默认不存在
```

### 8.1 `run_manifest.json`

必须包含：policy/model、全部 source/checkpoint/stats locks、精确命令和 resolved config、`H/C/M/D`、flow times 或 sigma/delta、`L` 集合、normalization、seed/reset 语义、视频写帧 contract、telemetry schema version、host/runtime 版本与输出路径。

### 8.2 raw trace

每个 query 至少保存：

- `x_chain`, `times_or_sigmas`, `deltas`；π0 可沿既有 observer 同时保存 `z_endpoint`，Fast-WAM 原始旁路
  另存 `v_chain` 并在离线阶段用 `z=x-(timestep/1000)*v` 生成 `z_endpoint`；
- `final_model_action`（原生归一化 active 14D）；
- `env_action`（实际交给环境的解码 action）；
- 可得时的 `robot_state`；
- join key：`run_id/policy/task/rank/env_slot/episode_id/query_idx`。

π0 的 `x/z` 是 `[5,50,14] / [4,50,14]`；Fast-WAM 是 `[11,32,14] / [10,32,14]`。两侧 raw action chain 每 query 只是几十 KB 量级，主要空间来自 lossless query images 与视频；无需导出 KV/video latent。

### 8.3 query / episode / derived 表

`queries.csv`：join key、source seed/reset id、query start/end action slot、executed length、video pre/post/frame range、success before/after、trace shard/row。  
`episodes.csv`：task、policy、episode/reset id、accepted seed、success、总 action slots/queries、first success slot/query、video path。  
`derived/query_horizon.*`：join key、`h`、`L`、`V/y/b/s/r/R/S/I`、executed-tail flag、phase 与 phase provenance。

## 9. 分阶段采集与 episode 预算

所有 seed/block 在运行前写入 manifest；不能看到 outcome 后挑 episode。

| 阶段 | 数据量 | 目的 | 是否进入主分析 |
|---|---:|---|---|
| P0：机制/等价 | `adjust_bottle`，π0 2 episodes + Fast-WAM 2 episodes | 验证 raw chain、join、视频边界和 off/on 行为相同 | 否 |
| P1：首个固定 block | 两侧各 16 episodes | 第一次看 Position、S/I、成功/失败样本数和阶段覆盖 | 是 |
| P2：第二个固定 block | 两侧各再 16 episodes | common-task 主集达到各 32、降低 episode-level 方差 | 是 |
| P3：Fast-WAM-only 扩展（可选） | 3 tasks × 16 episodes = 48 | 看信号结构是否跨任务形态存在，不作 π0 对照 | 是，单独报告 |

默认执行总量是 116 episodes，其中 4 个 P0 不计入统计，common-task 主分析 64 个，Fast-WAM-only 扩展 48 个。首批执行顺序为 **P0 → P1**；P1 结果 schema 完整后再决定是否启动 P2/P3。

P3 的 3 个任务应在 Fast-WAM official coverage 内预先各选一种形态：approach/rotation、transport/place、bimanual/contact。具体任务名由用户在实施前锁定。

如果 common-task 32 episodes 某一 outcome 类少于 5：

- 不补挑“刚好失败/成功”的 seed；
- 先如实写“当前不支持 outcome contrast”；
- 或由用户批准下一个固定 16-seed block 后整体追加。

official RLinf RoboTwin collection 当前可直接用于本计划的 π0/π0.5 task-matched checkpoint 是 `adjust_bottle`；其他公开 task 项主要是别的 policy family。因此 P3 不能被写成 π0-vs-Fast-WAM multi-task comparison。若要多任务 π0 对照，需要另有 source-locked π0 checkpoints 或训练，这是新任务。

## 10. 最小分析图组

1. **Model-native schedule + endpoint convergence**：分别画 π0 `tau`、Fast-WAM `sigma/delta` 与 tail endpoint spread，不混坐标尺度。
2. **Position profile**：每 policy/task 的 `b_h`、`s_h` 及置信区间，直接暴露 horizon-position 主效应。
3. **Raw vs residual heatmap**：同一 query 的 `y(q,h)` 与 `R(q,h)` 并排，证明去位置结构后的变化。
4. **S timeline**：`S(q)` 对 query/action time，叠加独立 phase、success boundary 和 query frames。
5. **I heatmap + frame strip**：成功/失败各取按预注册规则选择的代表 episode，不人工挑“最好看”样本。
6. **Outcome / phase aggregation**：episode 内先聚合，再做 episode-level bootstrap；避免把 horizon cell 当独立样本。
7. **Cross-policy standardized panel**：只比较 `R/S/I` 的 distribution/rank/association；图注明 raw scale excluded。
8. **L sensitivity**：π0 `2/3/4` 与 Fast-WAM `3/5`；共同主结果固定 `L=3`。

这些图能回答“信号是不是主要由位置决定、query-wide 还是局部、与独立阶段/outcome 是否相关”，但不能单凭 observational association 宣称 action importance 或 adaptive chunking 有收益。

## 11. 行为不变与少量高信息量检查

telemetry 的硬约束：

- 默认 `enabled=false`；关闭时不构建 writer，不改返回类型或既有控制流。
- 开启时只 detach/copy 已经计算的 tensor；不增加 forward、RNG draw、dtype cast、update reorder 或 action truncation。
- raw 产物写唯一新目录，不覆盖 official eval 视频/结果。

每侧只做以下 focused checks：

1. synthetic/fixed input 下 off/on action bitwise 或严格数值等价，并比较 RNG / local generator state；
2. `z=x-tau*v`、population variance 与 final chain endpoint 检查；
3. writer round-trip 与 query/episode join 完整性；
4. 一个 fixed-seed real chunk 的 off/on parity，然后才采 P0。

这四项直接保护“观测不改变行为”这一语义边界；不预设额外 fallback 或诊断树。

## 12. Worktree、环境与资源隔离

实施时建议两个独立、source-locked worktree/branch：

```text
π0:       codex/sz-current-pi0-dvac-observe       # base 7d07a421...
Fast-WAM: codex/sz-fastwam-dvac-observe           # base 7faa711...
```

- 不在 clean official checkout 上直接开发，不共享 branch、Python env 或可写结果目录。
- π0 继续 current RLinf env；Fast-WAM 继续 official standalone env；不交叉 pip install。
- 原始资产/checkpoint 只读复用，结果写数据盘 `/data/chenyiteng/results/...`，不写根分区。
- 默认顺序执行两侧采集，不与正式训练并发。真正执行前刷新 GPU/RAM/disk/process，再按空闲资源选卡；本文不预占具体 GPU。
- telemetry tensor 在 action 已确定后转 CPU/写盘；不在 GPU 保留跨 query 历史。

## 13. 一批连贯实现的顺序

1. 用户先锁定本计划第 14 节的选择并授权创建两个 worktree/branch。
2. 在 π0 current `7d07a421` 上按第 5 节一次性语义重放 6 文件增量；不混入训练算法。
3. 在 Fast-WAM `7faa711` 上按第 6 节一次性加入 action-chain side channel。
4. 两侧各跑 focused tests 与一条 fixed-seed action parity；产出 review packet。
5. 在真正 P0/P1 前给出 resolved config、精确命令、输出目录、seed/block、资源、预计时长和停止条件，取得明确批准。
6. P0 通过后执行预注册 P1；离线生成 Position/Residual/S/I 与最小图组。
7. 用户看 P1 后再决定 P2、P3 或 optional π0 control trace。

## 14. 实施前需用户决定 / 授权

1. 是否批准两个独立 worktree/branch，还是先只实现 π0 或 Fast-WAM 一侧。
2. 首批是否按 `P0 + P1`：两侧各 2 个机制 episode，再各 16 个统计 episode。
3. common-task 是否固定为现有 `adjust_bottle`；建议是。
4. 是否先不做 optional π0 control trace；建议先不做，只有 P1 确认 query-level 对齐不够时再加。
5. P3 是否需要；若需要，锁定 3 个 Fast-WAM official tasks。它只回答 Fast-WAM 信号跨任务结构，不是跨 policy 性能对比。
6. phase 标注采用人工双检、固定外部 VLM，或二者结合；无论哪种，都必须独立于 DVAC signal 并保存 provenance。

上述决定之前，本计划不授权 server 写操作、依赖安装、数据采集或训练。

## 15. 本地上下文路由

- [DVAC signal/data contract](../rlinf-robotwin-pi0-dvac-telemetry/01_SIGNAL_AND_DATA_CONTRACT.md)
- [旧 π0 telemetry implementation + smoke review](../rlinf-robotwin-pi0-dvac-telemetry/03_IMPLEMENTATION_RESULT_AND_SMOKE_REVIEW.md)
- [旧首批数据分析](../rlinf-robotwin-pi0-dvac-telemetry/04_FIRST_DATA_ANALYSIS.md)
- [Position / Residual 与相关工作边界](../rlinf-robotwin-pi0-dvac-telemetry/10_ACTION_WEIGHT_POSITION_RESIDUAL_AND_RECENT_CREDIT_LITERATURE_20260821.md)
- [RoboTwin control trace note](../rlinf-robotwin-pi0-dvac-telemetry/14_ROBOTWIN_CONTROL_TRACE_IMPLEMENTATION_NOTE_20260822.md)
- [R-only / G35 后续机制链；本次不迁移](../rlinf-robotwin-pi0-dvac-telemetry/15_R_ONLY_G35_SIGNAL_CREDIT_AND_MECHANISM_CHAIN_20260822.md)
- [current RLinf source/port map](02_OFFICIAL_SOURCE_AND_PORT_MAP.md)
- [current port + Git strategy](10_CURRENT_RLINF_PORT_AND_GIT_STRATEGY.md)
- [Fast-WAM interface contracts](../fastwam-robotwin-rlinf-grpo/02_INTERFACE_CONTRACTS.md)

## 16. 2026-08-22 实施结果与当前边界

### 16.1 current RLinf π0

- branch：`codex/sz-current-pi0-dvac-observe`；base=`7d07a421...`；telemetry commit=
  `f7cf0f6092b92e0ab2b813bc8acd8ee132a288e5`；当前head/parity commit=
  `800baf80d6eab64169cf0e691eb04a681a093ee9`，已普通push到用户个人RLinf仓同名branch，
  local/remote/upstream一致、ahead/behind=`0/0`、worktree clean且没有force。
- 实际仍为计划中的exact 6 logical files：3 modified + 3 added，`+1135/-9`。默认关闭时不构造writer、
  不传telemetry kwarg、不增加metadata；开启时只保存已有Euler chain，不改action/RNG/training schema/loss。
- Ruff、format、compile、4个focused tests和off/on Hydra compose通过；telemetry实现独立review无阻塞。
  另新增一个359行real-query parity工具，CPU/static checks与第二次独立review均PASS；它锁1次真实reset、
  同一观测2次policy query、0次environment action，并比较action/chain/logprob/value/post-RNG及`x-v*t`。
  真实checkpoint parity和episode尚未运行。
- GPU2的P0草案锁定两个预注册fixed reset IDs `100100052,100100066`：最多2 episodes、400 action
  slots、8 queries；real single-query parity harness已经补完/审查/提交，启动前仍须用户批准完整packet。

逐命令账和P0命令草案见
[`evidence/13_PI0_DVAC_CURRENT_IMPLEMENTATION_LEDGER_20260822.md`](evidence/13_PI0_DVAC_CURRENT_IMPLEMENTATION_LEDGER_20260822.md)。

### 16.2 official Fast-WAM

- branch：`codex/sz-fastwam-dvac-observe`；base=`7faa711...`；telemetry commit=
  `fc652fb49cd32350eca15734b5c7124c0b8c2c02`；当前local head/parity commit=
  `c63dc9b5384d6637a93cc862dbe2815d0332801d`，parent精确且worktree clean。
- 实际为exact 7 files，`+675/-3`：模型Euler capture、policy/writer接线、父子Hydra透传、配置、独立writer
  与focused test；没有改vendored RoboTwin、训练或RLinf。
- exact Fast-WAM runtime下compile/import、4个focused tests和独立review均通过。默认关闭时返回仍只有action；
  telemetry只复制既有`x/v/t/delta/x_next`，并明确拒绝compile+telemetry组合。
- 另有exact 3-file、`+384/-1` real-query parity增量；accepted seed已锁`4300001`，CPU checks与focused
  rereview均PASS。预算口径是0次policy `take_action`加1次official expert feasibility rollout；真实GPU尚未运行。
- 尚未push：Fast-WAM与`rlinf_fastwam`是不同Git历史。需先建立`Yutenji-Nyamu/FastWAM` fork并登记已经
  生成的独立read/write deploy key；不能复用前一仓deploy key，也不会把无关历史塞入RLinf仓库。
- 真实GPU/model/simulator parity与P0仍未执行；resolved packet由Fast-WAM专题流水维护。

Fast-WAM逐命令账见
[`../fastwam-robotwin-rlinf-grpo/evidence/SHENZHEN_DVAC_IMPLEMENTATION_LEDGER_20260822.md`](../fastwam-robotwin-rlinf-grpo/evidence/SHENZHEN_DVAC_IMPLEMENTATION_LEDGER_20260822.md)。
