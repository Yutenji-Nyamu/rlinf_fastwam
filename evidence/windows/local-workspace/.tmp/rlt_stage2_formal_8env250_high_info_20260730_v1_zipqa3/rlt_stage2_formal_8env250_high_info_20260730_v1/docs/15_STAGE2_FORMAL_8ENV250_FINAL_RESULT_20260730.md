# Stage 2 RoboTwin 8-env × 250-cycle formal 最终结果

> 现场冻结时间：2026-07-30 23:03+08:00。本文记录已自然结束的
> `adjust_bottle` RLT × π0 × RoboTwin 单任务运行；ManiSkill 只是在实现和参数来源中
> 使用的参考，不是本次环境或结果口径。

## 1. 完成结论

- 运行于 `2026-07-30 11:37:25+08:00` 启动，
  `2026-07-30 22:23:41+08:00` 自然结束；
- 完成 `250/250` outer cycles，wall-clock `10h46m16s`，`exit_code=0`；
- driver、Ray 和训练进程均已退出，两卡现场显存回到约 5MiB；
- fatal、NaN、CUDA OOM、cgroup OOM 和 OOM-kill 均未发现；
- `global_step_250` completion marker、两个 rank trainer state 和十个按 25-cycle
  间隔保存的 checkpoint 均完整。

因此，这次运行是完整 endpoint，不是运行中快照，也不是从中途成功率挑出的 best
checkpoint。

## 2. 成功率与四阶段

阶段边界由本次 TensorBoard 实际 tag 重新推导，不使用启动前预测：

| 阶段 | 实测 cycles | train 成功率 | 含义 |
|---|---:|---:|---|
| P1 | 1–135 | 156/1080 = 14.44% | π0 reference 收集 replay |
| P2 | 136–154 | 22/152 = 14.47% | reference 控制环境，actor/critic 已开始更新 |
| P3 | 155–191 | 94/296 = 31.76% | student 接管，BC/Q 权重继续 ramp |
| P4 | 192–250 | 375/472 = 79.45% | student 在线，BC/Q 固定为最终权重 |

全程 train 为 `647/2000=32.35%`；它混合了前三个 warm-up/ramp 阶段，不能单独代表
最终 policy。更有信息量的是：

- student 控制阶段 cycle 155–250：`469/768=61.07%`；
- 最后 20 cycles：`134/160=83.75%`；
- 最后 10 cycles：`68/80=85.00%`；
- 最后 5 cycles：`36/40=90.00%`。

同一组独立 fixed 20-seed deterministic eval 为：

| Cycle | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | 225 | 250 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 成功 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 1/20 | 6/20 | 16/20 | 11/20 | **18/20** |

cycle 225 从 80% 回落到 55%，说明单个 20-seed 点仍有方差或策略波动；但 endpoint
恢复到 90%，同时最后 20 个 train cycles 为 83.75%，没有最终退化证据。可支持的结论是：
本次单任务、固定 20-seed 协议下 RLT student 明显优于 reference 阶段并在 endpoint
达到高成功率；不能据此声称跨任务、held-out seed 或论文真机泛化。

## 3. SAC 数值健康度

最终累计 critic/actor optimizer updates 为 `102,260/51,130`，pending debt 为 0。

| 指标 | Cycle 250 |
|---|---:|
| actor / critic loss | -0.08119 / 0.001286 |
| BC loss | 0.01280 |
| actor / critic grad norm | 2.821 / 0.170 |
| Q0 / Q1 / Q(data) | 0.2515 / 0.2383 / 0.2123 |
| BC / Q weight | 2.5 / 0.45 |
| weighted BC / weighted Q | 0.0320 / 0.1132 |

actor loss 为负是损失中含有 `-w_Q Q` 的预期结果，不是数值错误。两条 critic 的差约
0.0132；policy Q 比 replay-action Q 高约 12%–18%，需要作为轻度 Q 乐观偏差观察，
但 critic loss、critic grad 和 actor grad 都保持有限，actor grad 也远低于 clip 10。
结合全量 finite、pending=0、成功率上升和空 fatal 扫描，没有发现训练发散证据。

## 4. 资源结果与风险

| 资源 | 本次峰值/均值 |
|---|---:|
| GPU0 / GPU1 显存峰 | 19.37 / 19.51 GiB |
| GPU0 / GPU1 平均利用率 | 30.0% / 27.7% |
| matched worker RSS 峰 | 87.42 GiB |
| EnvWorker RSS 峰 | 62.05 GiB |
| cgroup anon 峰 | 82.43 GiB |
| cgroup file cache 峰 | 191.01 GiB |
| cgroup current 峰 | 240.00 GiB |
| cgroup OOM / OOM-kill | 0 / 0 |

两卡负载对称但没有算力顶满；主要瓶颈仍是 RoboTwin simulator、rollout 和
collect/train 交替。运行期间 worker RSS/anon 持续阶梯增长，且 cgroup 多次触顶，
所以启动前定义的 `sustained memory pressure / anon growth` 风险判据确实被触发。
这次没有因此崩溃；进程退出后 matched RSS 归零、EnvWorker RSS 近零、anon 降至约
0.3–0.6GiB，说明大部分 retained anonymous memory 隶属于 worker 并已释放。退出后仍约
156GiB 的 cgroup 占用主要是 file cache，不是存活训练进程 RSS。

结论是：当前 8-env 规模完整跑通，但下一次扩 env 或延长预算前应先定位 EnvWorker
retained-memory 增长；不能只凭显存余量继续放大并行。

## 5. 产物

- run root：
  `/root/autodl-tmp/experiments/rlt_stage2_formal_8env_250c_20260730_v1/
  robotwin_adjust_bottle_rlt_stage2_formal_8env_250c_v1`
- final checkpoint：
  `.../checkpoints/global_step_250`
- runtime evidence：
  `/root/autodl-tmp/experiment_exports/rlt_stage2_formal_8env_250c_20260730_v1/runtime`
- TensorBoard：
  `/root/autodl-tmp/experiments/rlt_stage2_formal_8env_250c_20260730_v1/tensorboard`

十个 checkpoint 为 cycle 25、50、75、100、125、150、175、200、225、250；总计约
5.2GiB，final checkpoint 约 876MiB。最终本地只读快照是
`.tmp/formal250_final_20260730_230104`，其中 event/resources/driver SHA-256 分别为：

- `4dea11b356ea75b6b9fcb7d58cebf2be2e3fb9ea2e0b1d71ad913f7b8f2b17e2`
- `b7f0862d8538b22f6f53211645e8a749fd9e6409d9aa124e6b2f01e0c4f9ded2`
- `e829d2ceb452967f3fe609a4db5d470c9a5a058e072b9ff54d103556f959792d`

最终统计 JSON 与 success、optimization、resources 三张静态图使用同一份 cycle 250
快照生成。交互图又分别以 736px 和 360px 系统 Chrome 实际渲染，均无 JS error、
水平溢出或阶段表缺行。

## 6. 当前停点

运行已经完成，不需要继续轮询，也不自动启动第二个 run、resume、删除 checkpoint 或
清理 cache。后续应先决定是做：

1. final checkpoint 的独立 deterministic/held-out seed 评估；或
2. EnvWorker retained-memory 定位；或
3. 单独批准跨任务/更大数据规模实验。

这些都属于新的实验或诊断授权，不在本次只读收尾中执行。
