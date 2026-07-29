# RLT Stage 2：参数审阅与 fresh/resume smoke 批准包（2026-07-29）

> 当前状态：Stage 1 正式 artifact 已验收；Stage 2 artifact 绑定、配置 compose、
> 静态/单元测试和 launch-script 自测已通过。
> 授权边界：**本文件不是已执行记录；fresh/resume smoke 尚未启动，等待用户明确批准。**
> 机器证据：
> [`evidence/stage2_pre_smoke_20260729/`](evidence/stage2_pre_smoke_20260729/)。

## 1. 结果先行

Stage 2 已停在一个可审批、不会误启动 formal pilot 的边界：

- Stage 1 endpoint/manifest/full-weights/stats 已闭合，不能靠改路径静默换 artifact；
- Stage 2 代码提交为
  `3b610cb4685a1d41c97da64df67ab86561697dfd`；
- `ruff`、`py_compile`、`git diff --check` 和 RLT 集中单测通过，
  `19 passed in 8.20s`；
- formal source 默认 `runner.max_steps=0`，不会进入 collection/training cycle，
  避免按 `max_epochs=1000` 意外开始长训；仍不把直接初始化 formal runner 当成批准命令；
- fresh/resume 的完整 resolved config、精确命令、输出目录、资源监控和停止条件已经固定；
- 计划 smoke run root 与 runtime evidence root 仍不存在。

Stage 1 现在能给出的结论是：token checkpoint 可严格加载、π0 数值未改、真实
`z_rl` 比 shuffled/zero 更能重建 sample-specific prefix。它不能证明控制成功率提升；
Stage 2 smoke 也只验证完整 RL 调用链和 resume，不把 20 primitive steps 的结果解释成学习效果。

## 2. “抄谁”的优先级

| 标记 | 来源 | 本项目使用方式 |
|---|---|---|
| P | [RLT 论文](https://arxiv.org/abs/2604.23073) / [项目页](https://www.pi.website/research/rlt) | 决定 token bottleneck、frozen feature、reference-conditioned actor、BC+Q、高 UTD、critic:actor 语义 |
| M | [RLinf ManiSkill RLT](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/rlt.html) | 决定可执行 MLP/twin-Q/target/replay/worker/optimizer 骨架和一组公开超参 |
| R | 已验证 RoboTwin π0 | 决定 ALOHA、三相机、H=50、14D canonical action/state、stats、decode、env 与 2-GPU/4-env 接口 |
| A | 本项目适配 | C=10、full-task route、gamma、UTD cadence、warm-up、bounded replay 和 resume/sync 合同 |
| G | 工程治理 | branch/worktree、fail-closed budget、hash、输出隔离、监控和停止条件 |

ManiSkill 是 SAPIEN 上的机器人仿真/训练框架，不是 RLT 算法。这里“继承 ManiSkill”
是继承 RLinf 的 RLT 实现骨架；不会把其 Panda 8D action、两相机、64 env、
peg geometry gate 或 motion-planning expert 搬进 RoboTwin。

## 3. Stage 2 数据流与调用流

```text
RoboTwin observation（三相机 + proprio14）
  -> frozen Stage 1 π0/RLT feature
       z_rl[2048] + π0 reference action[H=50,D=14]
  -> 截取 reference chunk[C=10,D=14]
  -> route
       train pre-ready: reference
       train ready: stochastic RLT actor
       eval: deterministic RLT actor mean
  -> 实际 canonical action[C=10,D=14] 写 transition
  -> 统一 output transform，只 decode 一次为 absolute qpos
  -> RoboTwin 执行最多 10 个 primitive steps
  -> reward[<=10] + termination/truncation + next observation
  -> compact replay
       curr/next {z_rl, proprio, ref_chunk}
       executed canonical action, reward chunk, terminal flags
  -> twin-Q TD update
  -> 每两个 critic updates 做一个 actor BC+Q update
  -> target EMA
  -> 每个 outer cycle 开始前 actor -> rollout full sync
  -> eval / DCP / per-rank RLT trainer state
```

`H=50` 是 π0 一次预测的未来长度；`C=10` 是实际执行和 TD 的 macro chunk；
`D=14` 是 12 个双臂 joint 加 2 个 gripper。相对旧 RoboTwin π0 的 C=50，
C=10 最多提高约五倍重规划频率；这是把 RLT 的 macro-control 接到 RoboTwin 的显式成本，
不是 ManiSkill 环境参数。

## 4. 正式 Stage 2 参数逐组审阅

### 4.1 系统、runner 与并发

| 参数 | formal 值 | 来源 | 解释 |
|---|---:|---|---|
| node/GPU | 1 node，GPU 0–1 | R/G | 当前服务器和已验证 π0/DSRL 拓扑 |
| placement | actor/env/rollout 均在 ranks 0–1 | R | 每 rank 2 个 train env；不复制环境或 venv |
| train/eval env | 4 / 4 | R | 小规模 RoboTwin 并行；不是 ManiSkill 64/256 |
| actor world size | 2 | R | 两 rank 同步梯度、各自 replay/state |
| weight sync | 每 outer cycle，interval=1 | M/G | fresh 和 resume 的第一轮 action 前都同步，避免 rollout 用随机/旧 actor |
| `max_epochs` | 1000 ceiling | M | 只作 runner 上界，不是批准预算 |
| `max_steps` | **0** | G | formal fail closed；pilot 必须显式 override |
| eval/save interval | 10 / 10 candidate | A | 只在正式 pilot 批准时生效；smoke 覆盖为 1/1 |
| logging | TensorBoard | M | 记录 train/eval、RLT schedule、loss、Q、gradient、timer |
| FSDP | `no_shard`，fp32 | M | MLP 很小，避免为它引入 shard 复杂度 |
| batch | global512 / micro128 | M | 两 rank 的 gradient accumulation=2；这是小 MLP batch，不是 4B π0 batch |

正式 pilot 的唯一尚未决定项是总 collection-cycle 预算。当前 source 不允许用
`max_epochs=1000` 代替明确决定。

### 4.2 环境、feature 与动作合同

| 参数 | formal 值 | 来源 | 解释 |
|---|---:|---|---|
| task/robot | `adjust_bottle` / `aloha-agilex` | R | 首版单任务 |
| images | 3，含 wrist camera | R | 与 Stage 1/base π0 相同 |
| prompt | `adjust the bottle` | R | task-specific fixed prompt |
| π0 horizon | H=50 | R | base checkpoint 原生 horizon |
| actor/ref chunk | C=10 | P/M/A | 每次执行 10 步后重观测；相对 C50 的主要适配 |
| action/proprio | D=14 / P=14 | R | output-transform 前 canonical 域 |
| feature | frozen `z_rl=2048` | P/M | 来自验收后的 Stage 1 endpoint |
| prefix | image-only `768×2048`，mask on | R/Stage 1 | 真实三相机 prefix；不是 ManiSkill 1024 |
| OpenPI denoise | flow ODE，4 steps | R | 继承当前 RoboTwin π0 inference |
| train max primitive | 200/cycle | R | 4 env 满长时 80 global macro rows |
| eval max primitive | 200 | R | fixed reset IDs、deterministic student |
| domain randomization | 全关 | R/A | 首版先隔离算法，不增加视觉分布变量 |
| reward | simulator sparse success，coef1 | R | 不接 reward model，不借用 DSRL reward |
| action decode | canonical 只 decode 一次 | R/G | replay/Q/BC 不混用 normalized/delta/absolute qpos |

Stage 1 manifest 已把 H/C/D、`z_rl`、prefix、adapter 和 stats SHA 写入合同。
Stage 2 worker 又把这些值与 resolved config 交叉验证；任一不一致都在训练前报错。

### 4.3 route 与 eval

| mode | learner ready | 控制器 | exploration | 写 replay |
|---|---:|---|---|---:|
| train | 否 | frozen π0 reference | 无 student exploration | 是 |
| train | 是 | RLT actor | fixed Gaussian std .002 | 是 |
| eval | 任意 | RLT actor mean | 无 | 否 |

ManiSkill 用 peg/hole geometry predicate 定义关键精细阶段，并可选 expert intervention。
RoboTwin `adjust_bottle` 首版没有可信 geometry gate、人类或 expert，因此公开设为
`C_t ≡ True`：student ready 后负责全任务。这样不伪造 gate，但任务比“只优化精细阶段”
更难；所以不能把它称为论文精确复刻。

### 4.4 critic、TD 与 target

| 参数 | formal 值 | 来源 | 解释 |
|---|---:|---|---|
| Q | twin Q | P/M | 两个 Q 减少 over-estimation |
| TD aggregate | `min(Q1,Q2)` | M | 计算 bootstrap target |
| actor Q | Q1 | M | actor objective 不对 min 的切换点反传 |
| `gamma` | .99 per primitive | A | chunk reward 内逐步折扣，bootstrap 用 `.99^reward_horizon`；比 ManiSkill `.96` 更适合 200-step RoboTwin |
| `tau` | .005 | M | target EMA |
| target cadence | every critic update | M | `target_update_freq=1` |
| target scope | all | M | 复制整个小 model；Q forward 实际只用 target Q |
| terminal | termination 不 bootstrap | M/R | 真实成功/失败终止截断 TD |
| truncation | pure time-limit bootstrap | A | 使用真实 final observation，不把时间上限伪装成失败 |
| entropy alpha | fixed 0 | M/P | fixed std 负责探索，不训练 temperature |

### 4.5 actor loss、探索与 optimizer

actor objective：

$$
\mathcal{L}_{actor}
=-w_Q Q_1(s,\pi(s))
+w_{BC}\operatorname{MSE}(\pi(s),a_{ref}).
$$

| 参数 | formal 值 | 来源 | 解释 |
|---|---:|---|---|
| warm-up BC/Q | 7 / .05 | M | critic 未成熟时强贴 reference |
| online BC/Q | 2.5 / .45 | M | 后期增强回报引导但保留 BC anchor |
| weight warm-up | 5,000 critic updates | A | 与 student-ready critic floor 对齐 |
| weight ramp | 再用 10,000 updates 线性过渡 | A | update 5k 开始，约 update15k 到 online 权重 |
| reference dropout | .5 | M | 半数训练样本遮 actor 输入的 reference；BC target 不遮 |
| fixed std | .002 | M/P | train mean 周围固定噪声；eval 取 mean；std 不学习 |
| critic:actor | 2:1 | P/A | 论文语义；ManiSkill 可执行 YAML 是 4:1 |
| actor/critic LR | 各 1e-4 constant | M | 两个独立 Adam optimizer |
| grad clip | 各 global norm 10 | M | 超过阈值才缩放；不是 loss clip |
| precision | fp32 | M/A | 小网络成本低，优先 Q-learning 稳定 |

BC/Q 数值、dropout、std、LR 和 clip 都直接继承 RLinf ManiSkill；5k/10k 时间轴与
ratio2 是项目选择。它们不是从 DSRL 或 Fast-WAM 搬来。

### 4.6 replay、UTD、warm-up 与 cap

当前同步 worker 的 desired update 数：

$$
N_{desired}
=N_{floor}
+\left\lfloor
\frac{N_{global\ online\ macro}}
{train\_every\_transitions}
\right\rfloor
\times update\_epoch.
$$

| 来源 | update epoch | trigger transitions | effective macro-UTD | critic:actor |
|---|---:|---:|---:|---:|
| 论文 | — | — | 5 | 2 |
| RLinf ManiSkill YAML | 5 | 5 | 1 | 4 |
| RoboTwin candidate | 5 | 1 | 5 | 2 |

所以当前 candidate 更接近论文的优化密度，但比 RLinf ManiSkill 可执行 YAML 更激进。
理由是 RoboTwin interaction/π0 query 昂贵，希望复用每条 transition；风险是低多样性
replay 上 Q 过拟合、clip 频繁和 wall time 变长。smoke 会验证执行合同，不会证明
UTD5 是最终最优超参。

| 参数 | formal 值 | 来源 | 精确含义 |
|---|---:|---|---|
| compact replay | on | M/A | 不存图片/prefix/logprob/value，只存 TD 所需字段 |
| cache/window | 15k/rank | A | 两 rank raw aggregate 约30k；是本地 cache/最近采样窗口，不改通用 hard capacity |
| min loader buffer | 1 | M/A | 只保护 dataloader；ready 由 RLT schedule 决定 |
| ready rows | 500/rank | A | 较慢 rank 也至少500；通常约1,000 global rows |
| critic floor | 5,000 | A | route 仅在实际完成5k critic updates 后交给 student |
| trigger | 每1 global online macro | P/A | steady UTD5 |
| per-cycle cap | 400 | M/A | 防止一次 collect 后阻塞做5k updates |
| replay auto-save | off | M | 每次 DCP 显式保存两 rank replay |

满长 formal cycle 是 `4 env × 200 / C10 = 80` global rows，UTD5 正好新增400 desired。
首次 ready 时只做400，留下约4,600 pending；之后 pending 通常保持而不是立即清零。
这不是丢 update：lifetime counters、warm-up anchor 与 `update_step` 在 resume 后重算
desired/pending。代价是 student 大约要到：

1. 约13个满长 reference cycles 才到 500 rows/rank；
2. 再约13个 cycles 才实际完成5k critic updates；
3. 因而约第26个满长 cycle 才开始 student control。

ManiSkill 的 10k rows/rank、30k critic floor、20k/50k actor schedule 不能原样照搬：
它是64 train env、不同 transition/action/task；但当前500/5k仍只是低预算 heuristic，
正式 pilot 必须以 smoke 的实际 throughput/稳定性再次确认。

### 4.7 resume 与 sync

每 rank checkpoint 只保存独立原始状态：

- `update_step`
- local lifetime transitions / episodes
- global warm-up transition / episode anchors
- replay 内容和 base SAC model/target/optimizers

`ready`、BC/Q ramp 和 pending update budget 均由这些原始状态重算，不在 checkpoint
重复保存第二套“真相”。completion manifest 必须 `complete=true`，rank set 为0/1，
world size、runner step、update step、文件 SHA 和 contract fingerprint 必须一致。

首版明确不做：

- RNG bitwise continuation；
- 跨 world-size resume；
- async RLT resume。

## 5. smoke 只改哪些参数

smoke 继承 formal 的两 GPU、4 train/eval env、frozen Stage 1 feature、H/C/D、MLP、
twin-Q、TD/target、batch512/128、fp32、LR/clip、BC/Q 数值、dropout、fixed std、
canonical decode、route、sync 和 resume contract。只缩短串行预算：

| 项 | formal candidate | smoke | 为什么缩 |
|---|---:|---:|---|
| runner cycles | fail-closed 0 | fresh 1；resume 总上限2 | 每个新进程只执行1个 cycle |
| primitive/cycle | 200 | 20 | 每 env 最多2个 C10 macro |
| ready rows | 500/rank | 2/rank | 两 env/rank 的 first chunk 即可 ready |
| critic floor | 5,000 | 8 | fresh 精确做8 critic updates |
| per-cycle cap | 400 | 20 | resume 精确覆盖20 critic updates |
| BC/Q warm-up/ramp | 5k/10k | 4/8 | 在两轮内覆盖 schedule 过渡逻辑 |
| replay cache/window | 15k | 64/rank | 仍大于 smoke 的实际 replay |
| eval/save interval | 10/10 | 1/1 | 每轮都走 eval 和 DCP |

global/micro batch **不缩**，仍是512/128和 accumulation2，因此 smoke 会真实覆盖正式
MLP optimizer batch；tiny replay 上会重复采样，只作执行合同，不作效果实验。

## 6. fresh 与 resume 的预期调用链

### Fresh

```text
randomly initialized MLP actor
-> cycle start full sync
-> 4 env × <=20 primitive，以 frozen π0 reference 控制
-> 每 rank replay >=2
-> 8 critic + 4 actor updates
-> target EMA
-> full sync
-> deterministic student eval
-> DCP global_step_1
```

fresh hard expectations：

- `critic_updates_run=8`、`actor_updates_run=4`、`update_step=8`；
- train `actor_switch_rate=0`，因为 collect 时尚未 ready；
- completion manifest `complete=true`、world size2、rank set0/1；
- 两 rank replay `total_samples>=2`，与 lifetime transition counter 相等；
- eval 明确是 deterministic student，不用 reference 替换。

### Resume

resume 是第二个新进程；只在 fresh exit0 和 DCP1 预检通过后启动：

```text
load DCP1 model/target/optim/replay/RLT raw state
-> collective contract + per-rank file SHA validation
-> global_step=1
-> cycle start full sync（在任何 action 前）
-> ready student collect
-> 20 critic + 10 actor updates
-> update_step 8 -> 28
-> deterministic eval
-> 独立 experiment name 写 DCP global_step_2
```

即使 episode 在第一 macro 后结束，只要每 rank 的两个 env 各写一行，新增 desired budget
也足以触发 cap20；所以 `update_step=28` 是 smoke hard gate。resume 前会保存 DCP1
逐文件 SHA，结束后必须证明 DCP1 未被覆盖。

## 7. 完整 resolved config 与 artifact

| 文件 | SHA-256 | 状态 |
|---|---|---|
| [`formal_bound_resolved.yaml`](evidence/stage2_pre_smoke_20260729/formal_bound_resolved.yaml) | `4cbb7c7c03457276723293845c14ddc3a2f960badabd38cc40b2c9c592baf59c` | formal fail-closed；不运行 |
| [`fresh_bound_resolved.yaml`](evidence/stage2_pre_smoke_20260729/fresh_bound_resolved.yaml) | `c45743c1c797a9010d9a0f0c36a41c4cbabf4fd8f69e39707cb501e7b3d5c229` | 待批准 |
| [`resume_bound_resolved.yaml`](evidence/stage2_pre_smoke_20260729/resume_bound_resolved.yaml) | `f91688d21c7d6180dacb169824210b415e49f1c7d26a27d2a917f562ab24c82a` | 待 fresh 通过 |
| [`stage1_binding_preflight.json`](evidence/stage2_pre_smoke_20260729/stage1_binding_preflight.json) | `b03c4ba6e6849152043e12a6287330179b756caeb35610b66f4eca1c9d2d229c` | 已通过 |
| [`resolved_contract_audit.json`](evidence/stage2_pre_smoke_20260729/resolved_contract_audit.json) | `a26d8db55d4ac306a618511ed971b325f219f0bb270d2bb26b638d540038469f` | 已通过 |

Stage 1 绑定：

```text
model:
  /root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/
  robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
manifest:
  /root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/
  artifact_acceptance_v2/stage1_artifact_manifest.json
manifest ID:
  robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1
manifest SHA:
  6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433
full-weights SHA:
  7dddc268733b978bf382cda77257371cf9de4155f60ec3094cc8ffcfd6d74bd0
stats SHA:
  649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a
```

## 8. 精确命令与输出目录

批准后先上传同一 evidence snapshot 内的 monitor 和 fresh wrapper，再运行：

```bash
bash /root/autodl-tmp/tmp/remote_rlt_20260729_start_stage2_smoke_fresh.sh
```

fresh 脚本：
[`remote_rlt_20260729_start_stage2_smoke_fresh.sh`](evidence/stage2_pre_smoke_20260729/scripts/remote_rlt_20260729_start_stage2_smoke_fresh.sh)，
SHA `6e0f1c7ce5497bd3d5a2bef539bbea5e3fc964a5d8259b16f472cf353d19e27a`。

只有 fresh 正常退出并通过 DCP1 preflight，才运行：

```bash
bash /root/autodl-tmp/tmp/remote_rlt_20260729_start_stage2_smoke_resume.sh
```

resume 脚本：
[`remote_rlt_20260729_start_stage2_smoke_resume.sh`](evidence/stage2_pre_smoke_20260729/scripts/remote_rlt_20260729_start_stage2_smoke_resume.sh)，
SHA `6494eaeb8cb2e6c1decee07d97798a0aba368ee15b0491faf3ecdfe6a9ff054c`。

输出固定为：

```text
run root:
  /root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1
fresh DCP:
  .../robotwin_adjust_bottle_rlt_stage2_smoke_fresh_v1/checkpoints/global_step_1
resume DCP:
  .../robotwin_adjust_bottle_rlt_stage2_smoke_resume_v1/checkpoints/global_step_2
runtime evidence:
  /root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1/
    fresh_runtime/
    resume_runtime/
```

两个 experiment name 分离，因此 resume 不会覆盖 DCP1。

## 9. 资源与时间预算

实际 CPU 构造给出：

- MLP total `2,162,202` params；model+target FP32 raw tensors 约16.5MiB；
- 含 model/target/grad/Adam moments 的粗上界约41.2MiB；
- compact replay `18,359 bytes/row`；
- smoke 64 rows/rank 的 raw tensor payload 约1.12MiB/rank；
- formal 15k rows/rank 约262.6MiB/rank。

因此资源主项是 frozen π0/RLT feature、RoboTwin env/Ray 与 file cache。参考同机历史
π0 两卡 smoke 约29–40GiB/card、27–29分钟；DSRL 两轮实测约34.8GiB/card。
本 smoke 的 actor update 只有8/20次，预计：

| 资源 | 预期/预算 |
|---|---|
| GPU | 约30–45GiB/card；保守不超过60GiB/card |
| non-reclaimable RAM | 预计低于180GiB；需看 anon/RSS，不拿 raw cgroup total 判断 |
| disk | 小 MLP DCP/replay/日志预计远低于1GiB；10GiB 为异常调查线 |
| wall time | fresh/resume 各约15–40分钟；每阶段 hard timeout 90分钟 |

资源 monitor 每2秒记录 host available、cgroup current/anon/file、memory events、`/dev/shm`、
磁盘、两卡显存/util、env/actor/rollout/Ray/driver RSS。raw cgroup total 中的大 file cache
通常可回收；停止依据是 anon/host available/memory events，而不是 RSS 之外再把
page cache 重复算成训练私有内存。

## 10. 立即停止条件

只停止本次 RLT smoke 的 driver/Ray，不切换或影响 DSRL worktree：

1. CUDA OOM、`memory.events` 的 `oom/oom_kill` 增加、rank death、NCCL/CUDA error；
2. actor/critic loss、Q、target、gradient、BC/Q weight 或 action 出现 NaN/Inf；
3. 任一 rank 未达到 replay ready，fresh 不是8 critic/4 actor updates；
4. replay action 不是14D canonical C10，或出现重复 decode/normalization mismatch；
5. eval 被 reference route 替代，而不是 deterministic student；
6. DCP completion 非 true、rank set/world size/state SHA/target/replay 不完整；
7. resume contract mismatch、`update_step`/lifetime/anchor/replay 回退，或首 rollout 前未 sync；
8. resume 没有新增 student transition、不是20 critic/10 actor、没有独立 DCP2；
9. GPU 持续超过70GiB/card、host available 低于200GiB，或 anon 持续超过300GiB且
   file cache 已不可回收；raw cgroup total 单独触顶不直接误判；
10. worker ready 后连续20分钟无 macro/update 进展，单阶段超过90分钟；
11. smoke 输出超过10GiB或磁盘 available 低于200GiB；
12. source/config/artifact SHA、branch cleanliness 或 upstream 状态与批准包不一致。

## 11. smoke 通过能证明什么

通过只允许得出：

- frozen Stage 1 feature -> reference/student route -> canonical action -> RoboTwin -> compact replay
  -> twin-Q/actor/target -> sync -> deterministic eval -> DCP 主链可执行；
- formal MLP batch512/128在两 rank 可用；
- fresh raw RLT state、replay、target/optimizer 可由第二个新进程严格恢复并继续；
- Stage 1 artifact、stats、action/prefix contract 在 Stage 2 没有漂移；
- 实测 GPU/RAM/disk/time 足以决定 pilot 的并发和预算。

不能得出：

- 20-step success rate 有统计意义；
- UTD5、500/5k warm-up 或 C10 已经最优；
- clean-50 token 一定提高控制；
- 与论文真机或 ManiSkill 400-demo 等规模复现。

## 12. smoke 后才决定的 formal pilot

formal source 已把总预算留空。若 smoke 通过，建议下一次只审批一个有解释的 bounded
pilot，而不是直接1000 cycles：

- 约30个满长 cycles：刚能覆盖约13-cycle replay warm-up、约13-cycle critic floor和
  少量 student cycles，只证明 phase transition；
- 约60个满长 cycles：才有更多 student data，并大致跨过5k warm-up加10k actor-weight
  ramp，更适合看初步控制趋势；
- eval/save 先保持每10 cycles；replay 15k/rank 在这两个预算内都不会成为 hard capacity。

最终选30还是60，应由本 smoke 的每 cycle wall time、feature/env RSS、Q/gradient稳定性和
student action偏离决定。本轮不提前启动或承诺 formal pilot。
