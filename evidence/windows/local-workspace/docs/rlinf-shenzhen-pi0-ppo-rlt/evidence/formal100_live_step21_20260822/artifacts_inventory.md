# 深圳 π0 PPO formal-100：Step 21 只读产物清单

> 现场快照：2026-08-22 09:58 CST（服务器时间 `2026-08-22T01:58:34+00:00`）。  
> 本文只记录该时点已经落盘且可见的产物；训练仍在增长，不能把计数外推为终态。审计全程只读，未停止训练、未修改服务器文件，也未下载 checkpoint 或视频。

## 1. 运行现场

- run：`/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1`
- driver：PID `1375834`，alive；已运行 `09:24:46`。
- 最新完整训练步：`Global Step 21/100`；TensorBoard 用零基 step，故最新完整事件为 `step=20`。
- 审计结束时正在生成下一步 rollout，进度为 `2/4`。
- fatal 扫描：`0`（未命中 traceback、CUDA OOM、WorkerCrashed、RayActorError、SIGKILL、No space left）。

## 2. 总体产物

| 项目 | 快照值 |
|---|---:|
| run apparent size | `35G`（文件字节合计 `37,039,671,790`，约 `34.50 GiB`） |
| 文件数 | `372` |
| checkpoint | `global_step_10`、`global_step_20` |
| TensorBoard event | `1` 个，`60,854` bytes |
| `metrics.log` | `141,682` bytes |
| `driver.log` | `198,911` bytes（持续增长） |
| MP4 | `352` 个，`88,764,747` bytes（约 `84.65 MiB`） |
| `/data` 可用 | 约 `3.1 TiB`，使用率 `6%` |

关键小文件：

```text
resolved config
  .../resolved.yaml                         6,920 bytes
  .../resolved.yaml.sha256                    169 bytes
launch manifest
  .../launch_manifest.txt                     331 bytes
TensorBoard
  .../tensorboard/config.yaml                7,108 bytes
  .../tensorboard/events.out.tfevents.1787330043.admin.1375834.0
                                                60,854 bytes
metrics
  .../metrics.log                          141,682 bytes
```

## 3. Checkpoint 结构核验

两个 checkpoint 在本快照中的结构一致：

| checkpoint | apparent bytes | DCP metadata | DCP shards | full weights |
|---|---:|---:|---:|---:|
| `global_step_10` | `18,475,261,914`（约 `17.21 GiB`） | 1 | 4 | 1 |
| `global_step_20` | `18,475,261,914`（约 `17.21 GiB`） | 1 | 4 | 1 |

每个 checkpoint 的文件合同：

```text
actor/dcp_checkpoint/.metadata                         1,604,771 bytes
actor/dcp_checkpoint/__0_0.distcp                  2,601,381,166 bytes
actor/dcp_checkpoint/__1_0.distcp                  2,601,485,800 bytes
actor/dcp_checkpoint/__2_0.distcp                  2,601,491,833 bytes
actor/dcp_checkpoint/__3_0.distcp                  2,601,503,753 bytes
actor/model_state_dict/full_weights.pt             8,067,778,207 bytes
```

结论边界：这是非空文件、数量和大小的结构完整性核验；本轮没有做额外 hash 或 fresh-load，以免给正在运行的训练增加约 16 GiB checkpoint 顺序读取负担。

## 4. 视频

| split | 数量 | 字节 | 目录结构 |
|---|---:|---:|---|
| train | `344` | `87,709,588` | `seed_0..3` 各 `86` 个；最新编号 `85.mp4` |
| eval | `8` | `1,055,159` | `seed_0..3` 各 `2` 个；编号 `0.mp4`、`1.mp4` |

计数解释：21 个完整 outer step 对应 `21 × 4 rollout epochs × 4 sampled seed videos = 336` 个 train MP4；审计时下一步已完成 2/4 epochs，又增加 8 个，所以现场为 344。两次评估分别发生在 Global Step 10 和 20，每次四个 seed 视频，共 8 个 eval MP4。

## 5. TensorBoard 内容与最新高价值指标

- 唯一 event 文件共有 `21` 组训练标量（event step `0..20`，对应 Global Step `1..21`）。
- 两次 fixed-64 eval 位于 event step `9`、`19`，对应 Global Step 10、20。
- 所有已解析 scalar 都是 finite；没有 NaN/Inf。

| 指标 | Global Step 1 | Global Step 20 | Global Step 21 | 已见范围 |
|---|---:|---:|---:|---:|
| train success_once | 78.71% | 90.63% | 88.48% | 74.80%–90.63% |
| eval success_once | — | 96.88% | — | Step 10: 90.63%; Step 20: 96.88% |
| actor approx_kl | 0.0644 | 0.0125 | 0.0130 | 0.00427–0.0644 |
| actor clip_fraction | 0.0966 | 0.0571 | 0.0549 | 0.0374–0.0966 |
| actor grad_norm | 34.30 | 23.71 | 30.90 | 18.99–36.02 |
| actor ratio | 1.0007 | 1.0004 | 1.0017 | 0.9911–1.0093 |
| critic value_loss | 0.1193 | 0.0244 | 0.0291 | 0.0244–0.1193 |
| critic explained_variance | -0.240 | 0.397 | 0.397 | -0.240–0.397 |
| rollout returns_mean | 0.717 | 0.873 | 0.844 | 0.695–0.873 |

完整步耗时的 TensorBoard 中位数为 `1,567.8 s`（约 `26.13 min`）；普通步大多约 24–27 分钟，Step 10/20 因 eval/save 分别约 `1,807.0/1,949.0 s`。截至 Step 21 的经验 ETA 仍约 34 小时量级。

## 6. 资源快照

该时点处于 rollout epoch 之间，因此 GPU utilization 是瞬时低值，不能解释成训练空转；显存和进程归属显示四卡 workload 仍完整驻留。

| 物理 GPU | 显存 MiB | 总显存 MiB | 瞬时 GPU util | 功率 W | 温度 °C |
|---:|---:|---:|---:|---:|---:|
| 4 | 66,632 | 81,559 | 8% | 158.63 | 48 |
| 5 | 64,190 | 81,559 | 1% | 151.25 | 39 |
| 6 | 65,472 | 81,559 | 0% | 171.23 | 37 |
| 7 | 66,140 | 81,559 | 0% | 155.66 | 48 |

- driver 所在 cgroup current：`1,528,978,501,632` bytes，约 `1.424 TiB`；`memory.high/max=max`。
- cgroup memory events：`low/high/max/oom/oom_kill = 0/0/0/0/0`。
- 整机 RAM：约 `2.0 TiB`；现场 available `632,420,256 KiB`，约 `603.1 GiB`；swap `0/6 GiB` used。
- `ps` 的用户 RSS 求和约 `1.406 TiB`，其中 EnvWorker RSS 求和约 `1.318 TiB`；RSS 会重复计算共享映射，资源判断以 cgroup current 和 host available 为主。

## 7. 快照结论

截至 Global Step 21，运行链路、两次定期 eval、两次定期 checkpoint、TensorBoard 和视频产物均按配置持续落盘；训练标量全部 finite，fatal 与 cgroup OOM 计数为零。成功率总体较起点提高，critic value loss 下降且 explained variance 从负值升至约 0.397；actor KL、clip fraction、ratio 和 grad norm 均处于有界范围。当前没有证据表明训练异常。

不过这仍只是 21/100 的在线训练证据：两次 fixed-64 eval 的 90.63%→96.88% 是积极信号，但样本点只有两个，暂不能据此断言最终收益或单调收敛。
