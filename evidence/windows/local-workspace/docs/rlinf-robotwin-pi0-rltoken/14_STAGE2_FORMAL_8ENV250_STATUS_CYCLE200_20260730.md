# RoboTwin RLT Stage 2 formal250：cycle 200 验证节点

> 服务器现场：2026-07-30 20:23:28+08:00；TensorBoard完整到cycle 200，
> console已到cycle 202。仅做只读检查、证据下载和本地可视化，没有控制训练进程。

## 1. 当前结论

1. 训练已实测进入稳定student阶段。实际四段边界为：
   - P1 cycle 1–135：reference collect；
   - P2 cycle 136–154：reference控制、SAC更新；
   - P3 cycle 155–191：student控制、BC/Q ramp；
   - P4 cycle 192起：`actor_weight_ramp_progress=1`，稳定student。
2. student上线早期的成功率下降已经明显恢复：
   - P1：`156/1080=14.44%`；
   - P2：`22/152=14.47%`；
   - P3：`94/296=31.76%`；
   - P4 cycle 192–200：`55/72=76.39%`；
   - cycle 200时train rolling-10：`77.5%`。
3. 同一20-seed deterministic eval从cycle 150的`1/20=5%`，上升到
   cycle 175的`6/20=30%`和cycle 200的`16/20=80%`。这已经是明确的策略改进证据；
   仍应等待cycle 225/250确认能否维持，不能只凭一个80%点声称最终收敛。
4. 数值没有崩溃：
   - cycle 200 critic/actor累计updates为`75,215/37,608`；
   - actor/critic loss为`-0.0661/0.00149`；
   - actor/critic grad norm为`3.189/0.189`，均未触及clip 10；
   - Q0/Q1/Q(data)为`0.214/0.206/0.181`；
   - BC/Q权重已固定为`2.5/0.45`，pending update debt为0；
   - fatal扫描没有CUDA OOM、NCCL fatal、Ray death、NaN或Inf。
5. actor loss转负本身不是错误：当前目标包含负的Q收益项；随着Q增大，
   `weighted_q=0.0961`已经超过`weighted_bc=0.0300`。需要观察的是Q差距和critic
   loss是否继续扩大。目前twin-Q仍贴合，policy-Q比data-Q高约14%–18%，属于观察项，
   尚没有数值发散证据。

## 2. 资源与停止条件

cycle 200资源快照：

| 项目 | 当前/峰值 |
|---|---:|
| GPU0显存 | 16.12/19.37GiB |
| GPU1显存 | 16.26/19.51GiB |
| GPU0/1全程平均利用率 | 30.3%/27.5% |
| cgroup current | 239.16/240.00GiB |
| cgroup anonymous | 76.61/77.44GiB |
| matched process RSS | 81.59/82.41GiB |
| EnvWorker RSS | 56.37/57.18GiB |
| cgroup file cache | 160.17/191.01GiB |
| OOM/OOM-kill | 0/0 |

与cycle 162相比，anonymous和matched RSS分别继续增加约7.4GiB，同时file cache被回收。
因此预先批准的`sustained memory pressure/anon growth`停止定义仍然成立，并未因无OOM而
解除。本轮授权仅为只读检查，所以没有擅自停止；训练仍在继续。

## 3. 产物与进度

- 20:23现场driver PID `154857`仍存活，console为`202/250`；
- `global_step_200` completion marker完整，checkpoint约747MiB；
- checkpoint 25/50/75/100/125/150/175/200均存在，总量约3.5GiB；
- `/root/autodl-tmp`剩余约821GiB，无磁盘压力；
- 按cycle 200后的普通cycle约150秒、另计cycle225/250评估与保存，预计约
  22:25–22:40完成。

运行入口：

```text
/root/autodl-tmp/experiments/rlt_stage2_formal_8env_250c_20260730_v1
/root/autodl-tmp/experiment_exports/rlt_stage2_formal_8env_250c_20260730_v1/runtime
```

精确命令、下载中断、压缩只读下载修复、本地复算和可视化QA见实施账本A137。
