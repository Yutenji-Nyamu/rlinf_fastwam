# AutoDL RLT/DSRL 并行与 AutoDL—深圳 GRPO 主存机制核对

日期：2026-08-23  
范围：只读复核本地保存的 resolved config、日志、资源 CSV、源码快照和既有报告；没有连接或修改服务器。

## 1. 结论先行

1. **AutoDL 的 RLT/DSRL 没有因为显存或主存不够而把训练 rollout 从并行改成串行。** RLT Stage 2
   的正式配置是在 4-env smoke 后通过独立 8-env 资源门，才采用 `8 env × 1 episode/cycle`；DSRL 正式训练是
   `4 env × 1 episode/cycle` 真并发。它们的顺序评估 wave 是为了凑固定评估样本数，不是训练资源退化。
2. **两卡首先是历史 world-size、batch/replay 合同和吞吐选择，不是单卡一定放不下。** RLT Stage 1
   是高利用率的 2-rank data parallel；RLT Stage 2 主要等 simulator；DSRL 约 85%–90% 墙钟花在大量
   SAC update，单卡会增加梯度累积，反而放慢主要瓶颈。
3. **用户记得的 AutoDL GRPO “锯齿式下降”是真实现象，而且旧 run 确有一项深圳 run 没开的完整
   train-env offload。** AutoDL resolved config 是 `env.train.enable_offload=true`：每个 outer-step
   rollout 结束后，EnvWorker 调 `RoboTwinEnv.offload()`，它关闭并清空 `VectorEnv.envs`；下一个 reset
   再重建子环境。深圳是 `env.train.enable_offload=false`，所以没有这次轮间全量重建。**这不是新版删了
   机制，而是两次配置不同**；current EnvWorker 仍保留同一 offload 分支。
4. **两边共有的细粒度清理也仍存在。** trajectory 发送后都有 `clear -> del -> gc.collect()`，
   `clear_cache_freq=1` 也会在 reset 时清 renderer/task cache。AutoDL CSV 中确有一次 cgroup/Env RSS
   同时下降约 11.5/12.7 GiB，但 Env RSS 的跨-step基线仍从约34.6 GiB增长到约138.9 GiB：完整
   offload 能制造较明显锯齿，却仍不能保证 native allocator/simulator 高水位全部退回启动值。
5. **深圳的更大压力来自 simulator 规模和生命周期同时放大。** AutoDL 的 eval 配置虽写了4个，
   `val_check_interval=-1` 使它根本不实例化；每个 EnvWorker 在 rollout 时只有8个 train env，且轮末
   offload。深圳每个 EnvWorker 同时建 `32 train + 16 eval = 48` 个 env，且 train/eval offload 都关。
   因而 rollout 时单 worker 活跃 env 是6倍、全局是 `192/16=12` 倍；最终约1.8 TiB 集中在四个
   EnvWorker，Ray 在整机95%阈值主动杀 worker。

## 2. RLT / DSRL：轮内并行是否被内存迫使收缩

| 阶段 | AutoDL 实际合同 | 资源与时间事实 | 判断 |
|---|---|---|---|
| RLT Stage 1 | 2 × A800；MB16/rank、GB32；2,000 optimizer steps | 28m54s；steady p50 0.777 s/step；26,447 MiB/card；overall util约83%；matched RSS 38.51 GiB、anon 39.53 GiB | offline data parallel 已较舒展；没有 simulator，也没有因内存串行化 |
| RLT Stage 2 | 2 × A800；8 train env × rollout_epoch1；250 cycles | 2,000 train episodes、34,851 macros；约10h46m；19.37/19.51 GiB/card；GPU util约30%；Env RSS峰62.05 GiB、anon峰82.43 GiB | 4-env smoke 后通过 8-env 门且**没有回退**；瓶颈主要是 batched simulator rollout，不是显存 |
| DSRL | 2 × A800；4 train env × rollout_epoch1；GB256/MB64；UTD20 | 198 cycles、792 train episodes、5,185 macros、94,260 SAC updates；常态31–32 GiB/card，DCP峰41.46/41.40 GiB；anon峰65.9 GiB | 4 env 是真并发；普通 learned cycle 约35 s rollout后做数百次 SAC update，更新占主要墙钟；不是内存迫使串行 |

细节上：

- RLT Stage 2 的 4-env 与 8-env 是**等总 episode 预算**的并发选择：`4 env × 500 cycles` 与
  `8 env × 250 cycles` 都是 2,000 episodes。历史运行先做 8-env 资源门，监控无 OOM 后才进入正式
  250 cycles，因此不能反写成“内存不够所以只敢串行”。运行后期确实发现 EnvWorker/anon 增长，
  这只支持“不再盲目扩更多 env”，不改变 8-env formal 本身是真并发这一事实。
- DSRL 每轮最多由4个并发episode产生4–40个 `N=20` macro transition，再严格做
  `20 × globally-new macros` 次更新。eval 的 `4 env × 3 waves = 12 episodes` 是评估样本数设计；
  train 没有用 3 个顺序 wave。增加 train env 会同步增加新 macro 和必须执行的 UTD20 更新，不能只按
  “显存还有余量”判断会更快。
- 所以深圳首条 smoke 保持两卡，是为了先复用已闭环的 rank、global batch、replay 分片和更新预算。
  单卡从容量上大概率可行，但不是最接近历史依据、也不是明显更快的首条配置。

## 3. AutoDL GRPO 与深圳 GRPO：规模不是同一档

| 项 | AutoDL 旧 GRPO | 深圳 current GRPO v2 |
|---|---:|---:|
| GPU / EnvWorker | 2 / 2 | 4 / 4 |
| train env × rollout epochs | 16 × 16 | 128 × 4 |
| train trajectories / outer step | 256 | 512 |
| train env / EnvWorker | 8 | 32 |
| eval env / EnvWorker | 配置为2，但eval关闭、未实例化 | 16，常驻；每10步使用一次 |
| rollout 时 active env / EnvWorker | 8 | 48 |
| train trajectories accumulated / EnvWorker / step | `8 × 16 = 128` | `32 × 4 = 128` |
| train-env轮末生命周期 | `enable_offload=true`；关闭后下轮重建 | `enable_offload=false`；跨轮保留 |
| actor global / micro batch | 512 / 32 | 2048 / 32 |
| host-memory policy | 240 GiB cgroup；运行期贴近上限 | cgroup high/max不限；Ray按整机95%阈值杀worker |
| 结果 | 到100/100并有`global_step_100`；OOM/OOM-kill=0 | 完整到52；Step53后四个EnvWorker被Ray杀 |

这张表把两个容易混淆的轴分开了：

- **临时 trajectory 轴**：每个 EnvWorker 每个 outer step 都是 128 条，规模相同；它能解释每轮
  `clear/gc` 附近的锯齿，但不能解释深圳每个 worker 数百 GiB 的长期基线。
- **simulator 轴**：深圳每 worker 同时存在48个env，AutoDL在train rollout时只有8个；全局是192对16，
  并且AutoDL train env在轮末关闭、深圳跨轮保留。
  深圳 Step36 时四个 EnvWorker PSS 已为 1,540.4 GiB、占 cgroup 93.55%，actor和rollout只占约3%。
  最终 Ray 日志中的四个 EnvWorker约为528.25/438.57/430.64/406.83 GiB，合计约1.8 TiB。

因此更有依据的解释是：深圳把大量顺序 rollout wave 换成了更高并发，同时把全局轨迹数从256翻到512；
这减少了顺序 wave，却把 simulator 常驻工作集显著放大。版本、视频和offload也有差异，不能用6倍做
精确线性预测，但直接PSS证据已经排除了“actor batch或普通file cache是主体”。

## 4. “动态清理 / 锯齿”到底是什么

### 4.1 AutoDL 独有于本次配置的轮末 train-env offload

AutoDL exact resolved config 的嵌套键是 `env.train.enable_offload=true`。旧 EnvWorker 在每次
`_run_interact_once()` 返回后遍历 train env 调 `offload()`；RoboTwin adapter 把它落实为
`VectorEnv.close(clear_cache=True)`，清空 `envs`，而下一次 `reset()` 发现列表为空便调用
`_init_envs()` 重建。深圳 exact resolved config 的相同嵌套键是 `false`，所以该 run 不走这条分支。

current EnvWorker 仍有同样的 `train_enable_offload` 判断和轮末调用。因此准确说法是：**深圳 run 没开
官方已有的 env offload 配置，不是新版 RLinf 丢了旧版动态清理代码。** 顶层
`env.enable_offload=true` 不控制这个分支；worker读取的是 `env.train.enable_offload`。

### 4.2 两边共有的细粒度清理

1. AutoDL resolved config 和深圳 resolved config 都是 `clear_cache_freq: 1`。保存的 RoboTwin
   `VectorEnv` 源码显示：每次 reset 增加计数，并在频率命中时调用
   `task.close_env(clear_cache=True)`。这会清 task/renderer cache，但不会重启 Ray EnvWorker。
2. 保存的旧 EnvWorker 在 trajectory 发给 actor 后执行
   `rollout_result.clear(); del trajectories; gc.collect()`；current `7d07` 只是把容器换成
   `EmbodiedTrajectoryBuilder`，仍执行同样的 `clear; del; gc.collect()`。

所以“旧 run 每轮会重建 train simulator”是对的；但它是**固定轮末 offload**，不是监视内存后才触发的
动态策略，也不会重启 Ray EnvWorker。没有证据显示旧 run 另有按内存阈值执行的 `ray.kill`、worker
restart 或周期性 Ray reset。

### 4.3 锯齿真实，但基线没有归零

AutoDL 原始资源 CSV 在 `2026-07-15 15:27:35 -> 15:27:37` 记录：

- cgroup：151,652 -> 139,831 MiB，下降 11,821 MiB；
- EnvWorker RSS：87,950.1 -> 74,971.9 MiB，下降 12,978.2 MiB。

这就是肉眼看到的锯齿。另一方面，用 metrics 的累计 elapsed 对齐同一资源 CSV：

| 完整 step附近 | cgroup | EnvWorker RSS |
|---:|---:|---:|
| 1 | 97,136 MiB | 34,626 MiB |
| 10 | 176,788 MiB | 112,920 MiB |
| 40 | 233,403 MiB | 135,893 MiB |
| 100 | 238,534 MiB | 138,921 MiB |

全程 EnvWorker RSS 峰为142,150.6 MiB，cgroup峰241,999/245,760 MiB。也就是说 AutoDL 在 step40后
大体形成了“高位平台 + 周期波动”，而不是每次清理回到启动基线。240 GiB cgroup自身还会对file cache
施加回收/限流，旧CSV没有anon/file分项，不能把每一次下降精确分摊给Python GC、renderer cache或内核回收。

深圳 current 仍有细粒度清理，却没有 AutoDL run 的完整 train-env offload；Step36从step1约
302.8 GiB增长到1,658.4 GiB，四个EnvWorker占93.55%，最终约1.8 TiB触发Ray阈值。
`gc.collect()`只能处理已经不可达的Python对象；renderer/native allocator、长期存活的48个env/worker
及其高水位不保证向OS返还。因此优先级有直接依据：先降低并发env，或评估重新启用官方
`env.train.enable_offload` 的速度/内存交换；不应先补重复GC或提高Ray阈值。

## 5. 可核对的本地证据

- RLT Stage 1/2规模和资源总表：
  `docs/rlinf-shenzhen-rlt-dsrl-port/05_PRE_SMOKE_DATA_RESOURCE_AND_GRPO_EXIT_DECISION_20260823.md:64-106`。
- RLT 8-env资源门“没有触发4-env回退”、formal资源：
  `docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md:730-748`；resume资源分项：
  `docs/rlinf-robotwin-pi0-rltoken/17_STAGE2_FORMAL_RESUME250_TO480_FINAL_RESULT_20260731.md:135-163`。
- DSRL并发/UTD设计：
  `docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md:274-299`；
  rollout/SAC时间：`docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_STATUS_REPORT_STEP15_20260728.md:88-143`；
  最终预算/资源：`docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_CLOSEOUT_REPORT_STEP198_20260729.md:20-32,130-145`。
- AutoDL GRPO exact config：
  `audits/20260717-084926-grpo-current/resolved-config.yaml:1-24,60-67,124-129,187-240`；
  两个EnvWorker现场：`audits/20260716-180543-grpo-current/remote-snapshot.txt:6-11`。
- 深圳 GRPO exact config：
  `docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/grpo_v2_final_step52_20260823/resolved.yaml:1-23,60-67,124-129,186-238`。
- AutoDL 100-step完成与资源：
  `audits/20260717-084926-grpo-current/metrics.log:3271-3284`；
  `audits/20260717-084926-grpo-current/peak.txt:1-30`；
  锯齿原始行 `audits/20260717-084926-grpo-current/resources.csv:3458-3464`；
  step对齐原始行 `:678-679,6843-6844,27788-27789,68290-68292`。
- 旧RoboTwin reset cache源码快照：
  `audits/20260719-robotwin-performance-analysis/source/robotwin_vector_env.py:110-111,169-175,409-459`；
  旧 EnvWorker 的eval开关、env分片、trajectory清理和轮末offload：
  `audits/20260718-1934-fastwam-diagnosis/env_worker.py:106-145,978-989,1247-1259`；
  RoboTwin adapter offload：
  `audits/20260719-robotwin-performance-analysis/source/rlinf_robotwin_env.py:241-259,410-412`；
  current source-lock的trajectory清理与仍保留的轮末offload：
  `.tmp/rlinf_7d07_source_20260823/rlinf/workers/env/env_worker.py:1026-1037,1360-1371`；
  current RoboTwin adapter仍保留同一offload入口：
  `.tmp/rlinf_7d07_source_20260823/rlinf/envs/robotwin/robotwin_env.py:410-412`。
- 深圳常驻env与锯齿机制源码审计：
  `docs/rlinf-shenzhen-pi0-ppo-rlt/11_FORMAL_CONFIG_AND_MEMORY_ANALYSIS.md:151-177`；
  GRPO Step36 PSS和增长：`docs/rlinf-shenzhen-pi0-ppo-rlt/20_GRPO_STEP36_LIVE_METRICS_AND_RESOURCE_REFRESH_20260823.md:61-98`；
  最终Ray kill：`docs/rlinf-shenzhen-pi0-ppo-rlt/23_GRPO_STEP52_EXIT255_FINAL_REFRESH_20260823.md:26-56`。

证据边界：AutoDL的resolved config、metrics和resource CSV是该run直接产物；旧源码来自同一时期保存的
本地审计快照，但当前目录未保存一个把该源码文件逐blob绑定到GRPO运行HEAD的manifest。因此“没有特殊
worker restart”应理解为**保存材料中未发现**，不是对所有未留存私有脚本的绝对否定；不影响配置、资源
曲线和current代码中已明确的主结论。
