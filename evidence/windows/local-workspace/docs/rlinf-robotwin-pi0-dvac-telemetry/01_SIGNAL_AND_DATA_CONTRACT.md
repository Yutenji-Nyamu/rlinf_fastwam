# π0 DVAC 信号、粒度与首轮数据合同

最后更新：2026-08-20  
适用范围：RoboTwin `adjust_bottle` 原始 π0 SFT。RLT、PPO 和训练期 telemetry 当前均不在范围内。

## 1. 先把“action 粒度”说清楚

这里有三种粒度，容易被都叫成 action：

1. `d`：一个 action vector 内的坐标，例如左右臂关节或 gripper 维；
2. `h`：chunk 中第几个未来 action vector，例如 `h=0..49`；
3. query/chunk：模型在一个 observation 上一次性生成整段 `H=50`。

π0 在去噪的每一步都同时维护完整 action chunk：

```text
x_i, v_i, z_i: [B, H, D_model]
RoboTwin π0:   H=50, D_model=32, D_active=14
```

所以答案是：**不仅能做 action-level；DVAC 的核心本来就是逐 `h` 计算。** chunk-level scalar 只是最后
为了总览再聚合一次。

## 2. `x_i`、`v_i`、`z_i` 分别是什么

- `x_i`：当前 flow step 仍带噪的整段 action tensor；
- `v_i`：模型在相同 `[H,D]` 网格上给出的 velocity；
- `t_i`：当前 noise time；
- `z_i=x_i-t_i v_i`：站在第 `i` 步，按当前 velocity 外推到 `t=0` 时预测的 clean action endpoint。

本地 π0 源码已经在 value 路径中计算同一表达式 `x0_pred = x_t - v_t * t_input`。当前 `M=4` 的
采样时间为：

```text
t_i = [1.00, 0.75, 0.50, 0.25]，随后到 t=0
```

每次更新都保持 `[B,50,32]`；output transform 最后才截取执行 chunk 和环境实际 14 维。因此记录
active slice 后，单个 query 的原始 trace 是：

```text
x_chain    [5,50,14]  # 初始 x + 四次更新后的 x，末项是最终 clean action
z_endpoint [4,50,14]  # 四次 endpoint preview
timesteps  [4]
```

## 3. 从坐标方差到 action 方差，以及 chunk 概览的边界

从最后 `L` 个 endpoint previews 取尾部集合 `T_L`，先对每个 `(h,d)` 算总体方差：

\[
U_L(h,d)=\frac{1}{L}\sum_{i\in T_L}\left(z_i(h,d)-\bar z_L(h,d)\right)^2.
\]

再跨 active action dimensions 求和，得到 DVAC 的逐未来 action 信号：

\[
V_L(h)=\sum_{d=1}^{14}U_L(h,d),\qquad h=0,\ldots,49.
\]

对应张量层级是：

```text
U_L(h,d)       [B,50,14]  坐标级诊断
V_L(h)         [B,50]     主 action-level 信号
```

`U_L(h,d)` 能看出信号是否被 gripper 或单侧机械臂主导，但 `M=4` 意味着每个坐标方差只有 2–4 个
endpoint 样本，会比跨 14 维求和后的 `V_L(h)` 更抖。首轮保留它的可计算性，不直接把它当训练权重。

另一个重要边界是：50 个 `h` 由同一个 Transformer/action expert 联合生成，彼此相关。`V_L(h)` 是
endpoint 改口幅度，不是第 `h` 个动作失败的独立概率。

如果以后需要 episode 总时间线，可以离线派生 `V_L,total=Σ_hV_L(h)`。它会丢掉 crossing 在哪个 `h`
发生的信息，因此不保存、不作为首轮主分析、不进入 DVAC 执行长度判断；仅可作为快速概览。这里还要
区分 `H` 和 `C`：`V_L(h)` 在完整 `H=50` 上可算，但 `C=50` 决定这50个 action slots 都会被执行后才
重新观察，所以 `C` 对时间和视频解释仍然重要。

## 4. 为什么保存全部四步，再离线算 `L=2/3/4`

DVAC 论文的默认是绝对尾长 `L=5`，不是“最后某个百分比”；论文没有给出一个可直接迁移成比例的
统一规则。当前 π0 总共只有 `M=4`，不能硬套 `L=5`。

本次有效选择是：

| L | 使用 endpoint steps | 直觉 |
|---:|---|---|
| 2 | `t=.50,.25` | 最集中看临近结束时是否仍改口 |
| 3 | `t=.75,.50,.25` | 在末端稳定与样本数之间折中 |
| 4 | `t=1,.75,.50,.25` | 使用全部 preview；最早一步可能主导，不再是很窄的 tail |

`L=1` 的方差恒为零，没有信息。离线实现必须用分母 `L`，即 PyTorch
`var(..., correction=0)`/`unbiased=False`，不能误用默认的 `L-1` 样本方差。

因此“4、3、2 都看”是对的，但这些全部属于**推理后的离线计算**。采集代码只保存四个原始 `z_i`
及对应 times；首轮结束后再重算任何 `L`、聚合方式或阈值，不需要重跑模拟器。

## 5. 为什么只让 active 14D 进入主信号

π0 内部 action tensor 是 32D；RoboTwin/Aloha 实际只消费前 14D，其余是模型接口 padding。把 padding
维加入 `sum_d` 会让信号脱离真实环境 action。

主 DVAC 还应在 normalized model-action space 计算，而不是 decode 后直接相加。后者混合了不同关节与
夹爪的物理单位/变换，数值尺度不再可比。

首轮因此保存 active normalized 14D 的 `x/z/action`，manifest 同时记录 `D_model=32` 和
`D_active=14`，明确这是有语义依据的 slice，而不是不知道内部形状。

## 6. 首轮必须保留的信息语义

下述是信息合同，不是必须照抄的文件/字段实现。writer位置、rank shard格式、图像压缩格式和字段名可按
source-locked代码做窄适配，只要这些语义可以无歧义恢复。

### 6.1 `run_manifest.json`

至少记录：

- run ID、UTC/local start time、机器标识；
- RLinf/RoboTwin commit、checkpoint 路径与完整 revision；
- 完整 resolved config 与启动命令；
- task、renderer、seed-file 路径/hash；
- 实际 `H/C/M/D_model/D_active`、dtype、`flow_ode`；
- telemetry schema version 与 shard 清单。

### 6.2 `query_index_rankNN.csv`

每个环境的一次 policy query 一行，保留能回答“第几次评估、哪个 chunk、从环境第几步开始”的最小
坐标：

```text
query_uid, eval_epoch, episode_idx, query_idx, action_slot_start,
rollout_rank, batch_or_env_slot, reset_id_or_seed, H, C, M
```

当前 `primitive_step_start` 更准确地说是 `action_slot_start`：它计数 π0/RLinf 的 qpos waypoint slots，
不是 SAPIEN physics-step。实现可直接采用新名字；若保留旧名字，manifest必须解释。

字段名可以顺着现有 worker 元数据小幅适配，但语义不能丢。特别是 reset ID/seed 不应仅由 batch slot
猜测；若它目前没有传到 rollout worker，就补一条窄 metadata 通路。

### 6.3 `trace_rankNN.npz`

以 `query_uid` 顺序保存：

```text
x_chain (recommended) [N_query,5,50,14] float32
z_endpoint          [N_query,4,50,14] float32
timesteps           [4]               float32
final_model_action  [N_query,50,14]   float32
env_action          [N_query,C,14]    float32
```

其中 `z_endpoint/timesteps/final action/query key` 是必需；`x_chain` 是现有 sampler 已持有且成本很低的
推荐审计量。若 exact source 的 worker payload表明导出它会明显增加峰值，允许不存，并在manifest
写清即可。

每个 query 还需保存或稳定引用 policy 实际输入的：

```text
head image, left-wrist image, right-wrist image
robot state
RLinf MP4 relative path, tile index, pre-frame, post-frame
```

episode 完成后还要能回连 success、实际 primitive steps 和 termination reason；它可进入一个小型
episode index，或复用官方 eval 结果再由稳定 key join。具体落点在代码实施时跟随现有 runner 结构。

## 7. 数据量与 CSV 选择

每个 query 的 active float32：

```text
z_endpoint: 4×50×14×4 bytes = 11.2 KB
x_chain:    5×50×14×4 bytes = 14.0 KB
合计:                         25.2 KB
```

1000 个 query 约 25.2 MB；官方 128-env、每条走满四个 query 的 512-query 上界约 12.9 MB，压缩前也
很小。数据规模不是问题。

问题在于 raw scalar 若展开成长 CSV 会膨胀成数百万行，读写和 join 都更笨重。因此：

- CSV 只做一行一个 query/episode 的索引；
- NPZ 保存规则 dense tensor；
- 后续选定分析口径后，再导出 `horizon.csv` 或画图。

## 8. 视频、图像与任务阶段对齐

现有 RLinf MP4 不是每个 `h` 一帧。C50路线中，它通常只包含：reset初始帧、四个chunk结束帧，以及
auto-reset可能多出的spill frame；每个worker的多个环境会被tile成一张复合图，30 FPS只是播放参数。
RoboTwin native recorder在VectorEnv中关闭。

首轮因此保存 query 输入的三路图像，并建立：

```text
query_uid -> reset ID -> action_slot_start
          -> MP4 path/tile/pre-frame/post-frame
```

这支持 query-level phase 分析：例如根据当前输入图判断free-space/contact-rich，再看该query完整
`V_L(h)`曲线；也可看执行一个C50 chunk前后的变化。

它不支持把未来 `h` 精确对到 simulator frame。RoboTwin把50个qpos waypoints经TOPP变成可变数量的
control/physics steps；简单按视频每一/两帧抽样会制造错误对齐。精确 per-`h` phase capture属于第二阶段
simulator instrumentation，不在首轮最小采集合同内。

## 9. 首轮不做什么

- 不在线设 rolling threshold 或改变执行长度；
- 不先挑 100 个 state 取代官方评估；
- 不做 SFT/PPO/RLT 数字横向结论；
- 不在采集代码中决定哪一个 `L` 最好；
- 不先画大量图；
- 不把 uncertainty 当作 advantage 的正负号。

首轮采集完成后先报告数据事实，再进入可视化讨论。届时最先考虑的图仍可以是
`query_idx × h` 热图、episode primitive-step 时间线、代表性 `V(h)` 曲线和少量 `z_i(h)` 收敛图，
但它们不是当前采集合同的一部分。

## 10. 直接源码依据

- π0 noise、flow loop、chains、endpoint的AutoDL共同基线快照：
  `.dsrl-impl-worktree/rlinf/models/embodiment/openpi/openpi_action_model.py`；正式实现前从Git对象
  `6d0db56b...`重新核对，不把本地dirty staging当权威source。
- active Aloha action transform：
  `.rlt-impl-worktree/rlinf/models/embodiment/openpi/policies/aloha_policy.py`
- official adjust-bottle eval：
  `.rlt-impl-worktree/evaluations/robotwin/robotwin_adjust_bottle_openpi_eval.yaml`
- standalone eval worker 当前只发送 actions：
  `.rlt-impl-worktree/rlinf/workers/rollout/hf/huggingface_worker.py`
- 视觉帧收集与MP4：
  `.dsrl-impl-worktree/rlinf/envs/wrappers/record_video.py`、
  `.dsrl-impl-worktree/rlinf/envs/robotwin/robotwin_env.py`。

论文依据：[DVAC arXiv v1](https://arxiv.org/html/2606.03847v1)。官方 task-matched 权重范围可由
[RLinf RoboTwin Hugging Face collection](https://huggingface.co/collections/RLinf/robotwin)复核。
