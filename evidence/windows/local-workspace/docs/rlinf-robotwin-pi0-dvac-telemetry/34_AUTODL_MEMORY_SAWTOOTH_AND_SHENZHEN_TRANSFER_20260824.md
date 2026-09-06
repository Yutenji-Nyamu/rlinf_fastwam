# AutoDL内存锯齿：3分钟说明

最后更新：2026-08-24  
详细数字和源码：[35号技术附录](35_AUTODL_MEMORY_SAWTOOTH_TECHNICAL_APPENDIX_20260824.md)。

## 1. 一句话答案

如果问的是“为什么AutoDL主存到顶后还能在高位锯齿、没有像深圳一样被Ray杀”，**主要机制是run级
cgroup `MemoryHigh/MemoryMax`带来的回收与节流**。RLinf/RoboTwin轮末offload解释其中一部分cycle边界
下降，是配套释放机会，不是这次要替代cgroup的主方案。

## 2. 为什么两边不一样

| | AutoDL旧GRPO | 深圳GRPO v2 |
|---|---:|---:|
| train env / worker | 8 | 32 |
| eval env / worker | 不常驻 | 16，常驻 |
| `env.train.enable_offload` | `true` | `false` |
| `clear_cache_freq` | 1 | 1 |

深圳每worker常驻`32+16=48`个环境，是AutoDL的6倍；全局为192 vs 16。这个事实解释深圳为何占用更高，
**不构成本轮修改并发数的建议**。本轮保持env数、rollout、batch和评估预算不变。

`clear_cache_freq=1`两边都有，只清reset时的task/renderer cache。深圳当前RLT已经打开train offload并正常
运行，说明nested offload在这台机器可用；此前GRPO没有打开它。

AutoDL每轮的主路径是：

```text
rollout使主存上涨
→ cycle结束
→ RoboTwinEnv.offload()
→ VectorEnv.close(clear_cache=True)
→ 清空env，下轮reset再重建
```

它不删除模型、replay或仍被引用的tensor。worker PID不变，但重建环境会花时间。旧数据曾在2秒内看到
EnvWorker RSS下降约12.7 GiB；这是offload能制造局部锯齿的证据。长期RSS仍从约34.6升到138.9 GiB，
说明真正把整个run约束在高位边界内的仍是cgroup压力，而不是offload彻底清零。

## 3. Linux和Ray是什么角色

- 本次AutoDL容器现场为`memory.high≈236 GiB`、`memory.max=240 GiB`：Linux超过水位后回收/节流；历史
  多个run在`memory.events max`持续增加时仍保持`oom=0`并继续训练，这是cgroup主机制的直接运行证据；
- Linux可丢可重读的clean file cache，不能随意删除仍存活的simulator/native内存；
- Ray spilling只处理Plasma对象；Ray memory monitor则会杀worker；深圳上次是后者。

所以通常不影响训练数值，但可能造成重建、重读和短暂停顿。

## 4. 深圳先怎么做

本轮目标是新增系统资源约束，而不改变训练拓扑：

1. 把本次训练进程组放入独立cgroup；
2. `MemoryHigh`设在Ray整机95%杀worker阈值之前，让内核先施加回收/节流；
3. `MemoryMax`略高于`MemoryHigh`，但仍留在Ray阈值和其他进程所需内存之下；
4. 精确GiB值应根据深圳启动前的live总内存、后台占用和Ray阈值另行冻结，不直接照抄AutoDL的236/240。

同时启用RLinf已经存在的轮末释放：

```yaml
env:
  train:
    enable_offload: true
    task_config:
      clear_cache_freq: 1
  eval:
    enable_offload: true
```

当前深圳RLinf源码支持train/eval两条独立offload分支；必须设置上面两个nested键，顶层
`env.enable_offload`不控制这条路径。它会增加环境重建时间，但不改变env数量、trajectory预算、batch或
优化算法。

本轮明确不建议修改每worker env数、rollout epoch、batch或评估预算；也不使用周期性`drop_caches`或提高
Ray kill阈值。

## 5. 怎样快速判断

| 现象 | 谁在工作 |
|---|---|
| cycle边界EnvWorker RSS/PSS和`anon`下降，PID不变 | RLinf/RoboTwin offload |
| `file`下降并伴随cgroup pressure/event | Linux reclaim |
| 日志出现Spilled/Restored | Ray spilling |
| PID消失并有memory-pressure日志 | Ray杀worker |

给深圳机传达的核心是：**先用run级cgroup在Ray杀worker之前建立回收水位，同时打开已有的train/eval
offload；训练并发与样本合同保持不变。**

需要完整配置对照、源码调用链和系统原理时，再看
[35号技术附录](35_AUTODL_MEMORY_SAWTOOTH_TECHNICAL_APPENDIX_20260824.md)。
