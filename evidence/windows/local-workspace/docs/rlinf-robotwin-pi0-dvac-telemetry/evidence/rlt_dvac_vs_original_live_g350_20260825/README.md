# 原 RLT 与 RLT teacher-DVAC `[0,2]`：g357 只读对照

快照时间：2026-08-25 10:36:43 CST。动态状态以之后的服务器现场为准。

## 1. 当前运行

- 最新完整 cycle：`357/480`，训练仍在继续。
- g357 training rollout：`6/8=75%`；向后 5-cycle / 10-cycle 均值为 `90.0% / 83.75%`。
- g357 方法量：DVAC 已 apply；`weight mean/median/p05/p95=0.991/0.950/0.319/1.855`，
  `weight ESS=0.896`，top-20% weight mass=`0.289`。
- 现场资源：GPU0/1=`19.12/19.20 GiB`，cgroup RAM=`103.09 GiB`；
  `memory.events high/max/oom/oom_kill=0`，数据盘剩余 `724 GiB`。

本轮服务器操作均为只读；没有改变配置、进程或训练产物。

## 2. 曲线怎样读

[对照图](RLT_ORIGINAL_VS_DVAC_W0TO2_THROUGH_G357.png)包含三种独立口径：

1. raw training success：每个 cycle 恰好 8 条 on-policy episode，所以每点以 12.5 个百分点跳变；
2. trailing-5 / trailing-10：包含当前 cycle、只向后平均，不插值，也不混入 eval；
3. fixed20 eval：4 个并发 eval env × 每个 env 串行 5 个 fixed reset，共 20 条独立 episode。

截至共同的 g357：

| 口径 | 原 RLT | teacher-DVAC `[0,2]` | 差值 |
|---|---:|---:|---:|
| g357 raw | 100.0% | 75.0% | -25.0 pp |
| g357 trailing-5 | 100.0% | 90.0% | -10.0 pp |
| g357 trailing-10 | 93.75% | 83.75% | -10.0 pp |
| g1--357 training rollout 均值 | 50.07% | 39.60% | -10.47 pp |
| g301--357 training rollout 均值 | 93.64% | 87.06% | -6.58 pp |

固定评估最新点为 g350：原 RLT=`17/20=85%`，DVAC=`19/20=95%`。但最近三个 fixed20
点 g300/g325/g350 的合计恰好相同，都是 `53/60=88.33%`。截至 g350 的全部 14 个评估点则为：
原 RLT=`123/280=43.93%`，DVAC=`93/280=33.21%`；这个全程合计包含前期尚未学会的多轮零分。

当前最稳妥的读法是：DVAC 这一条的学习上升明显更晚；最近已经接近原 RLT，且 g350 单点更高，
但还没有形成全程稳定领先。应等待 g375--g480 的后续 fixed20 和最终点。

## 3. 训练中间评估合同

- 触发点：g25、g50、……、g475，再额外评估最终 g480；全程计划 20 次。
- 每次：4 个 eval env 并行，每个 env 跑 5 个 epoch，即 `4 × 5 = 20` 条 fixed-ID episode。
- 全程预算：`20 × 20 = 400` 条 eval episode。
- 本快照已经完成 g25--g350 共 14 次，即 280 条；下一次是 g375。
- `success_once`表示一条 episode 在最多 200 个 action slots 内是否曾经成功一次；不是只看终态。

## 4. 与历史原 RLT 的逐叶参数核对

逐行比较当前 live `runtime/resolved.yaml`、历史 fresh1--250 和历史 resume251--480；未发现意外差异。

| 分类 | 原 RLT 与当前 DVAC-RLT 的共同合同 |
|---|---|
| task / teacher | `adjust_bottle`；同 Stage1 `global_step_2000`；teacher `H50/M4/D14/flow_ode` |
| student | MLP action `C10 × 14`；同 `z_rl=2048` |
| GPU / ranks | 1 node、GPU0--1；2 actor ranks、2 rollout ranks、2 env ranks |
| train 并行/串行 | total 8 env，名义每 rank 4；rollout epoch=1，即每 cycle 并行收集 8 条 |
| eval 并行/串行 | total 4 env；rollout epoch=5，即并行 4、串行 5、每次共 20 条 |
| episode budget | 480 cycles、每条最多 200 slots；总训练 episode=`480×8=3840` |
| batch | global/micro=`512/128`；两 rank 时 gradient accumulation=2 |
| replay / update | per-rank readiness=10,000；cache/window=50,000/50,000；update_epoch=5；critic:actor=2:1；每 cycle 最多 1600 updates |
| actor schedule | update warmup/ramp=`20k/50k`；BC/Q从 `7/.05` 变到 `2.5/.45`；reference dropout=.5 |
| optimizer / Q | actor/critic lr=`1e-4` constant、clip=10；twin-Q、min target、gamma=.99、tau=.005 |
| eval / save | 两者都每 25 cycle；最终 g480 额外做一次 |
| video / offload | train/eval video=false；train nested offload=true、eval=false；clear_cache_freq=1 |

预期差异只有三类：

1. 当前新增 teacher-DVAC：L2/L3/L4、选 L3、取 student 前 C10、initial-replay global-z baseline、
   `z clip=±2`、`strength=.5 → w∈[0,2]`，只改 Q→student-action 的反向梯度；
2. run 名称、输出路径和 seed 文件路径改变，但 train/eval seed 文件内容 SHA 与历史相同；
3. 历史总 480 是 fresh1--250 后 strict resume251--480；当前按用户选择用一个 fresh 进程直接 1--480。

因此当前实验的“算法单变量”成立；进程在 g250 是否重启是运行协议差异，画图时已用灰色虚线标出。

## 5. 单卡是否能跑

容量上，`1 × A800-80GB`很可能能跑；但不能把当前 YAML 原样只隐藏一张卡：

- 当前是 `no_shard`，两张卡上的每个 rank 本来就持有完整模型，并非把一个模型切成两半；现场每卡约
  19 GiB，单卡合并后的估计峰值约 27--40 GiB，80 GiB 有明显余量；
- 要把 actor/env/rollout placement 从 `0-1`改成单 rank `0`；global/micro batch仍可保持512/128，
  gradient accumulation会从2自动变为4；train/eval总并发仍可保持8/4；
- 2-rank的warmup=10k和cache/window=50k都是 per-rank。若要保持与两卡相同的**全局**数据合同，
  单 rank应考虑改成 readiness=20k、cache/window=100k/100k；若不改也能训练，但进入更新会更早，
  replay覆盖范围也不同；
- 当前2-rank checkpoint不能直接切到1-rank resume，正式单卡应 fresh 启动，除非单独做状态转换；
- 主存不会因GPU减半而自动减半。保留8个sim env时估计仍需约90--115 GiB可用RAM；当前240 GiB
  cgroup足够，128 GiB属于较合适的下限；
- 预计单卡墙钟约为当前两卡的1.5--2.2倍，正式结论仍需一次单卡smoke。

所以结论是：显存和主存容量没有明显阻碍；真正要处理的是单rank拓扑、per-rank replay语义和不能直接续
当前checkpoint，而不是缩小任务并发或batch。

## 6. 本目录产物

- `RLT_ORIGINAL_VS_DVAC_W0TO2_THROUGH_G357.png/.svg`：同轴图；
- `training_and_eval_cycles.csv`：两条训练的raw、5步、10步、fixed20数据；
- `summary.json`：图中关键数字与评估合同；
- `raw/dvac_metrics.log`、`raw/dvac_resources.csv`、`raw/dvac_resolved.yaml`：本次只读快照输入。

正式启动合同见[33号文档](../../33_RLT_DVAC_FRESH480_FORMAL_LAUNCH_20260824.md)；历史原 RLT 完整结果见
[历史480结果](../../../rlinf-robotwin-pi0-rltoken/17_STAGE2_FORMAL_RESUME250_TO480_FINAL_RESULT_20260731.md)。
