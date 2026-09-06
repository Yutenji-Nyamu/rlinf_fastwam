# AutoDL内存锯齿技术附录：配置、源码与系统机制

最后更新：2026-08-24  
阅读顺序：先看[34号3分钟说明](34_AUTODL_MEMORY_SAWTOOTH_AND_SHENZHEN_TRANSFER_20260824.md)；只有需要
核对数字、调用链或系统原理时再看本文。

## 1. 这次重新核对后最重要的结论

需要把两个问题分开：

- **为什么cycle边界会突然下降**：RLinf/RoboTwin offload提供局部释放；
- **为什么用量升到高位后仍被约束在边界附近，而没有先被Ray按整机95%杀掉**：run级cgroup
  `MemoryHigh/MemoryMax`是主要机制。

环境规模解释“为什么深圳占用更高”，但本轮不据此改变训练并发：

| 项 | AutoDL旧GRPO | 深圳GRPO v2 |
|---|---:|---:|
| EnvWorker数 | 2 | 4 |
| train env / worker | 8 | 32 |
| eval env / worker | 没有实例化 | 16，常驻 |
| rollout时常驻env / worker | 8 | 48 |
| `env.train.enable_offload` | `true` | `false` |
| `env.eval.enable_offload` | eval未运行 | `false` |
| `clear_cache_freq` | 1 | 1 |
| 系统边界 | 本次容器high约236 GiB、max 240 GiB | 没有同样的run级cgroup hard cap；Ray按整机95%杀worker |

所以深圳单worker常驻环境数是AutoDL的6倍，全局是`192 env vs 16 env`；而且AutoDL每轮关闭train env，
深圳跨轮保留。这是资源解释，不是建议降低env数。两边都有`clear_cache_freq=1`，因此它不能解释两边差异。
这里特指此前GRPO：深圳current RLT已经配置train offload并正常运行，证明同一机器能够走这条释放路径。

## 2. RLinf/RoboTwin到底释放了什么

current RLinf仍保留下面的分支：

```text
EnvWorker.interact()
  -> 本轮rollout完成
  -> 若env.train.enable_offload=true
  -> RoboTwinEnv.offload()
  -> VectorEnv.close(clear_cache=True)
  -> 关闭所有子环境并清空envs列表
  -> gc.collect()
  -> torch.cuda.empty_cache()
  -> 下次reset发现envs为空，再重建
```

源码位置：

- current EnvWorker轮末offload：
  `.tmp/rlinf_7d07_source_20260823/rlinf/workers/env/env_worker.py:1360-1371`；
- current eval结束后的独立offload分支：同文件`:1453-1456`；
- RoboTwin入口：`.tmp/rlinf_7d07_source_20260823/rlinf/envs/robotwin/robotwin_env.py:410-412`；
- VectorEnv关闭/清空：`audits/20260719-robotwin-performance-analysis/source/robotwin_vector_env.py:450-459`。

这不会卸载actor、critic、replay或Ray EnvWorker进程。它只销毁该worker内本轮不再保留的simulator env，
所以worker PID保持不变。

`gc.collect()`只处理已经不可达的Python对象；SAPIEN/native allocator不保证把全部历史高水位立即返还OS。
`torch.cuda.empty_cache()`处理PyTorch未使用的GPU缓存，不是主存清理的主体。

## 3. `clear_cache_freq=1`为什么不等于完整offload

`clear_cache_freq=1`发生在RoboTwin reset路径，用于关闭task/renderer cache。它仍然保留VectorEnv与子环境
生命周期；而`env.train.enable_offload=true`会在整轮结束后关闭并清空全部train env。

AutoDL和深圳当时都配置了`clear_cache_freq=1`，但深圳仍持续增长，说明仅重复GC或reset cache不足以替代
完整offload。

## 4. AutoDL的真实锯齿和高位平台

旧AutoDL GRPO资源CSV在2秒内记录过：

- cgroup：151,652 → 139,831 MiB，下降11,821 MiB；
- EnvWorker RSS：87,950 → 74,972 MiB，下降12,978 MiB；
- worker PID不变。

这次下降发生在236 GiB `memory.high`以下，且EnvWorker RSS同步下降、PID不变，因此主体可归到应用
offload/对象释放，而不是内核高水位回收。但它不是“每轮回到启动值”：

| 完整step附近 | cgroup | EnvWorker RSS |
|---:|---:|---:|
| 1 | 97,136 MiB | 34,626 MiB |
| 10 | 176,788 MiB | 112,920 MiB |
| 40 | 233,403 MiB | 135,893 MiB |
| 100 | 238,534 MiB | 138,921 MiB |

准确形态是“随训练上升，随后在cgroup高位边界形成平台，并伴随offload提供的周期下降”。offload不是能
消除所有native高水位的清零器；cgroup才是高位约束的主要组件。

## 5. Linux cgroup是高位约束主组件

本次AutoDL容器启动前快照为：

```text
memory.high = 253403070464 bytes ~= 236 GiB
memory.max  = 257698037760 bytes = 240 GiB
swap.max    = 0
```

这些是**本次容器现场值**，不是所有AutoDL实例的固定常数。

- 超过`memory.high`时，cgroup内的分配会受到回收压力和节流；
- `memory.max`是硬边界，碰到它时先回收，仍无法满足分配才进入cgroup OOM；
- `memory.events high/max`说明触发过边界压力；要结合`memory.stat`里的`anon/file`变化，才能判断主要回收
  了什么。

历史AutoDL run在`memory.events max`大量增加的同时保持`oom=0, oom_kill=0`并继续推进，说明分配反复触碰
边界后仍由回收/节流维持运行。与之相对，深圳此前没有同样的run级边界，先触发的是Ray整机95%杀worker。

代码、observer和启动脚本都没有写`memory.reclaim`、`drop_caches`或`malloc_trim`。当前容器也没有暴露
`memory.reclaim`文件。这支持“项目没有手动周期清理”；但仅凭容器视图不能排除宿主平台的全部管理动作。

官方语义见[Linux cgroup v2](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html)。AutoDL公开文档
说明实例有严格内存边界，但没有公开描述一个周期清理器：[AutoDL GPU实例说明](https://www.autodl.com/docs/gpu/)。

## 6. 为什么通常不损坏训练，但会影响速度

Linux按页能否重建和是否仍被引用处理，不理解业务上的“有用/没用”：

- clean file cache已有磁盘副本，可以丢弃并在以后重读；
- dirty cache通常需要先写回；
- 应用已经断开引用的simulator/Python对象可以由allocator释放；
- 仍被worker引用的anonymous/native内存不能直接删除；本次容器没有swap，不能靠换出解决长期存活堆。

所以不会随机删掉仍在计算的tensor，但可能产生环境重建、direct-reclaim停顿、page refault和I/O。历史DSRL
曾在没有OOM时出现memory PSI峰约7.12%，就是这种等待成本。内核内存类型见
[Linux内存概念](https://cdn.kernel.org/doc/html/latest/admin-guide/mm/concepts.html)。

## 7. Ray是另外两条路径

1. Ray object spilling只处理Plasma object store，可落盘后再恢复；不清理EnvWorker内部的Python、PyTorch、
   SAPIEN native heap。[Ray spilling](https://docs.ray.io/en/master/ray-core/objects/object-spilling.html)
2. Ray memory monitor在整机内存过高时会杀task/actor；深圳GRPO是在超过95%阈值后四个EnvWorker被杀，
   不是温和回收。[Ray OOM prevention](https://docs.ray.io/en/latest/ray-core/scheduling/ray-oom-prevention.html)

## 8. 深圳迁移：不改变训练并发合同

本轮冻结下列不变项：`total_num_envs`、每worker env数、rollout epoch、batch、评估预算和算法参数。

### 8.1 新增主要组件：run级cgroup

1. 只把本次训练process group放入独立cgroup/systemd scope，不影响其他用户和进程；
2. `MemoryHigh`放在Ray整机95%杀worker阈值之前，让内核先产生回收和节流；
3. `MemoryMax`略高于`MemoryHigh`，但也低于Ray阈值，并为系统/其他任务保留空间；
4. AutoDL的236/240 GiB只能说明约98.3%的high/max相对位置，不能直接当成深圳绝对值；深圳精确值应由
   启动前live总内存、后台占用和Ray阈值共同冻结。

cgroup不改变样本数、环境并发或优化公式；可能改变的是wall-clock，因为direct reclaim和节流会让分配变慢。

### 8.2 启用已有配套：train/eval offload

使用精确nested键：

```yaml
env:
  train:
    enable_offload: true
  eval:
    enable_offload: true
```

顶层`env.enable_offload`不能替代这两个键。offload不改变并发和trajectory预算，只在train rollout或fixed eval
完成后关闭环境，下次使用时重建；代价主要是重建时间。

不要部署周期性`drop_caches`，也不要提高Ray kill阈值。前者不释放仍存活的simulator heap并制造额外I/O，
后者只会推迟杀worker。[drop_caches说明](https://www.kernel.org/doc/html/latest/admin-guide/sysctl/vm.html#drop-caches)

## 9. 最小判别表

| 看到什么 | 更可能是谁在工作 |
|---|---|
| cycle边界`anon`与EnvWorker RSS/PSS下降，PID不变 | RLinf/RoboTwin offload |
| `file`下降，并伴随cgroup pressure/event | Linux cache reclaim |
| Plasma下降，日志出现Spilled/Restored和磁盘写 | Ray object spilling |
| PID消失，raylet报告memory pressure | Ray memory monitor杀worker |

## 10. 本地直接证据

- [AutoDL—深圳配置与资源完整核对](/C:/Users/86136/Documents/rl/docs/rlinf-shenzhen-rlt-dsrl-port/evidence/AUTODL_PARALLELISM_AND_GRPO_MEMORY_NOTE_20260823.md)
- [AutoDL旧GRPO resolved config](/C:/Users/86136/Documents/rl/audits/20260717-084926-grpo-current/resolved-config.yaml:1)
- [旧EnvWorker轮末offload](/C:/Users/86136/Documents/rl/audits/20260718-1934-fastwam-diagnosis/env_worker.py:1247)
- [RoboTwin VectorEnv关闭与清空](/C:/Users/86136/Documents/rl/audits/20260719-robotwin-performance-analysis/source/robotwin_vector_env.py:450)
- [深圳GRPO最终内存退出分析](/C:/Users/86136/Documents/rl/docs/rlinf-shenzhen-pi0-ppo-rlt/23_GRPO_STEP52_EXIT255_FINAL_REFRESH_20260823.md:26)

## 11. 当前fresh-480附带影响

历史RLT完整480是fresh250和resume230两个进程，cycle250自然做过进程级清空；当前teacher-DVAC run是单进程
fresh480，中点没有这次清空。replay容量受每rank 50,000限制，但native高水位仍要以当前monitor为准。
