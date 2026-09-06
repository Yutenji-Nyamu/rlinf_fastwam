# RLT论文设置、Git状态与单卡效率审计

日期：2026-08-30

## 1. 先说结论

1. 当前 RoboTwin RLT 在核心算法骨架上较接近论文：冻结VLA/RL token、`H=50`、student `C=10`、
   scalar twin-Q、actor的`-Q + BC`、50% reference dropout、UTD=5、critic:actor=`2:1`。
2. 实验协议不是论文复刻：当前是仿真full-task、同步`collect → update`、每个C10边界存一条replay row；
   论文是真机critical phase为主、采集与学习异步，并从primitive轨迹按stride 2构造重叠C10样本。
3. 当前“每轮8条”是`8 env × 1 episode`。论文没有runner cycle，也没有公开每批并行episode数；只报告
   每任务约400--1000个真机episode。因此不能说论文也是每轮8条或每轮1条。
4. Pure03/04源代码和配置已经推送到远端分支
   `codex/rlt-dvac-pure-reference-bc@f0aaf4b71669fad38d11ac85c90670386242c29d`；服务器工作树clean。
   训练checkpoint、完整日志和replay属于运行产物，没有推入Git；本地文档仓尚无远端。
5. 单条单卡run约比历史双卡慢`1.72×`，但两个单卡A/B能同时占两张卡。相较把两个双卡run顺序跑，
   总墙钟约从`40.4 h`降到`34.6 h`，实验吞吐提升约`16.7%`，单run GPU-hours减少约`14.3%`。
6. 显存确实宽松：每卡峰值约25.8/80 GiB；但两条并发任务的cgroup RAM峰值已到240 GiB上限，
   所以当前不能直接再叠加环境进程。GPU利用率呈“计算突发+等待”形态，不是稳定30%算力负载。

## 2. “样本”要先分三种

当前一次runner cycle中：

```text
8个并行env
→ 完成8个train episode
→ 每个episode最多200个action slots
→ student每次实际执行C10
→ 每个C10边界写1条macro replay row
→ 每条新增row安排UTD=5次critic更新，critic:actor=2:1
```

所以“每轮8个采样”准确指8个episode，不是8个primitive action，也不是固定8条replay row。

论文的数据口径不同：50 Hz primitive轨迹按stride 2滑动构造C10窗口，约每秒产生25个相互重叠的
RL样本；当前只在实际执行完一个C10后保存一个macro row。两边都写UTD=5，但分母不同，不能据此认定
“每单位机器人交互的更新强度完全一致”。

论文依据：[RLT正文与附录](https://arxiv.org/html/2604.23073)；公开复现说明：
[RLinf RLT文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/rlt.html)。

## 3. 论文与当前配置逐项对照

| 维度 | RLT论文 | 当前RoboTwin单卡RLT | 判断 |
|---|---|---|---|
| 场景 | 4个真机精细操作任务，主要训练critical phase | `adjust_bottle`仿真full-task | 任务适配 |
| 基础VLA | 私有π0.6 | RoboTwin任务SFT π0 | 模型适配 |
| Stage 1数据 | 每任务约1--10小时示范 | `clean-50`成功示范 | 当前更小 |
| Stage 1步数 | 2k--10k optimizer steps | 2k | 论文下界 |
| Stage 1更新对象 | 论文实验中VLA与RL token一起适配 | 已有task-SFT π0冻结，只训RL token | 算法允许，实验路径不同 |
| token重建decoder | autoregressive reconstruction | RLinf parallel reconstruction | 算子差异 |
| Stage 2冻结 | VLA与RL-token模块冻结 | 一致 | 对齐 |
| horizon / student chunk | `H=50 / C=10` | `H=50 / C=10` | 对齐 |
| action | 14D，student输出`10×14` | 一致 | 对齐 |
| 图像 | base/head + 双腕，共3路 | head + 双腕，共3路 | 高层对齐 |
| critic | twin-Q，整段C10输出一个scalar Q | 一致 | 对齐 |
| actor loss | `-Q + β·reference BC` | `-wQ·Q + wBC·MSE` | 高层对齐 |
| reference dropout | 50% | 50% | 对齐 |
| UTD / critic:actor | `5 / 2:1` | `5 / 2:1` | 数字对齐 |
| replay构造 | primitive stride 2，重叠C10窗口 | C10执行边界的macro row | 重要差异 |
| rollout与学习 | 异步 | 同步cycle：先collect再update | 重要差异 |
| reward | sparse task success | sparse task success | 高层对齐 |
| actor/BC schedule | 具体数值未公开 | `7/.05 → 2.5/.45`，20k warmup + 50k ramp | RLinf项目配方 |
| replay / batch | 容量、warmup和batch未公开 | 80k / 20k / global512 / micro256 | 项目运行参数 |
| 训练预算 | 400--1000真机episodes | 480 cycles × 8 = 3840仿真episodes | 数量更大，单episode成本不可直接比 |
| eval | 每agent 50 episodes，主要critical phase | 每25 cycles fixed20，full-task | 当前unique eval较小但更频繁 |
| intervention | 框架支持人类纠正 | 当前关闭 | 可选项不同 |
| primitive控制 | 50 Hz真机控制 | RoboTwin waypoint/TOPP执行 | 不能用同一Hz口径 |

因此当前最准确的称呼是“RLT的RoboTwin仿真移植”，而不是“RLT论文设置复刻”。

## 4. 当前相对RLinf公开示例的改动

RLinf的公开ManiSkill示例也不是论文原作者配置。当前RoboTwin单卡版本相对该示例主要是：

| 参数 | RLinf ManiSkill示例 | 当前单卡 |
|---|---:|---:|
| train env | 64 | 8 |
| eval | `256×1` | `4×5=20` |
| global/micro batch | `512/128` | `512/256` |
| `train_every_transitions` | 5 | 1 |
| 有效macro UTD | 1 | 5 |
| critic:actor | 4:1 | 2:1 |
| warmup replay | 10k/rank | 单rank 20k |
| replay窗口 | 50k | 80k |
| per-cycle update cap | 400 | 1600 |
| eval/save间隔 | 25/50 | 25/25 |

其中UTD5和2:1是有意贴近论文；其余主要是RoboTwin内存、单卡matched-width和480-cycle实验预算的适配。
当前完整resolved配置可看
[resolved.yaml](evidence/rlt_dvac_pure02_pure05_final480_high_info_20260829/pure02/resolved.yaml)。

## 5. Git到底推了什么

2026-08-30服务器现场：

```text
worktree: /root/autodl-tmp/RLinf_rlt_dvac_pure
branch:   codex/rlt-dvac-pure-reference-bc
HEAD:     f0aaf4b71669fad38d11ac85c90670386242c29d
status:   clean
remote branch HEAD: 同一个f0aaf4b7
```

最近相关提交：

```text
f0aaf4b7 config(rlt): add Pure03 and Pure04 formal variants
a2ae5cbe config(rlt): place strong pure DVAC on gpu1
ff432cbf config(rlt): add strong pure DVAC formal variant
cb88e9c5 feat(rlt): reweight successful reference BC with DVAC
```

结论：算法代码和Pure02--Pure05配置都已推送。Git不保存大checkpoint、replay、完整Ray日志和视频；它们继续留在
服务器运行目录。本地`C:\Users\86136\Documents\rl`目前是尚无commit/remote的文档工作区，所以这些Markdown、
轻量ZIP和图没有推到远端。

## 6. 单卡到底提高了什么效率

历史双卡单run：

```text
2 GPUs × 20.21 h ≈ 40.42 GPU-hours
wall time ≈ 20.21 h
```

当前matched单卡单run：

```text
1 GPU × 34.64 h ≈ 34.64 GPU-hours
wall time ≈ 34.64 h
```

因此：

- 单条run墙钟慢约`34.64 / 20.21 = 1.71--1.72×`；
- 单条run消耗的GPU-hours约少`14.3%`；
- 两个A/B若都用双卡只能顺序跑，约`40.42 h`；两个单卡run并行约`34.64 h`；
- 整体实验吞吐提升约`40.42 / 34.64 - 1 = 16.7%`。

所以你的判断是对的：资源利用改善了，但幅度没有达到“把双卡完全等价地压进单卡，因此A/B直接快一倍”。

历史数据和单/双卡拓扑见
[47号文档](47_RLT_DVAC_PURE_DUAL_FORMAL_LAUNCH_AND_C10_ANALYSIS_20260828.md)。

## 7. 为什么显存空很多，速度仍没有追平双卡

显存是“能放多少工作”，不等于“每秒能算多少”。当前global batch仍是512：

```text
历史双卡：GPU0算128×2，GPU1算128×2；两卡并行
当前单卡：GPU0算256×2；只有一张卡提供算力
```

把microbatch从128提高到256已经利用了空闲显存，并把单卡梯度累积从4轮恢复到2轮；但一张GPU仍不能凭借
更大batch自动获得两张GPU的总Tensor Core吞吐。

现场资源进一步说明瓶颈是混合的：

- 两卡显存峰值约`25.8/80 GiB`，容量有余量；
- GPU utilization P50约9%，P95=100%，说明运行在环境/同步/更新之间来回切换；
- 两任务cgroup RAM峰值约240 GiB，已经碰到容器上限；`oom=0, oom_kill=0`；
- 当前单卡稳态约`182--233 s rollout + 30--45 s update`，主要时间在rollout而非optimizer；
- fixed20评估step也明显更长，因此高频同步评估占一部分总墙钟。

## 8. 8个环境是串行的吗

不是。准确拓扑是：

```text
历史双卡：2个EnvWorker进程 × 每个4个env线程 = 8 slots
当前单卡：1个EnvWorker进程 × 内部8个env线程 = 8 slots
```

RoboTwin `VectorEnv`内部会用线程池并发提交8个环境step，所以不是逐个串行执行8条episode。不过
`setup_demo/reset`受一个VectorEnv级global lock保护：同一EnvWorker里的8次reset仍依次进行。损失的是：

- 第二个独立EnvWorker进程；
- 第二个rollout/model rank；
- 两条独立的`observe → policy → execute`流水线。

现场拆解也与此吻合：

| 段 | 历史双卡`2×4` | 当前单卡`1×8` |
|---|---:|---:|
| bootstrap/reset | 约29--32 s | 约59--61 s |
| env interaction | 约80--89 s | 约113--160 s |
| rollout总计 | 约119--125 s | 约182--233 s |

所以最明确的一块损失就是reset串行域从4个扩大到8个；其余来自单Python/SAPIEN进程的8线程争用和集中通信。
policy inference/通信在所查step里约7--11秒，不是第一瓶颈。

## 9. 单卡能否放多个EnvWorker

RLinf的placement层允许多个process rank共享同一物理资源；官方也提供env-decoupled模式，让env worker数与
rollout worker数不同：

- [Placement文档](https://rlinf.readthedocs.io/en/release-v0.2/rst_source/tutorials/user/placement.html)
- [Env-decoupled模式](https://rlinf.readthedocs.io/en/latest/rst_source/guides/env_decoupled_mode.html)

因此框架层面并非“不支持”。可以构造的候选是：

```text
单GPU
├─ actor rank ×1
├─ rollout rank ×1
└─ EnvWorker rank ×2，每个4 env
```

当前源码的env/actor分片计算允许`env world size=2, actor world size=1`，配置校验也只要求env数能够整除，
所以它首先是一个配置级候选，不需要改RLT loss。但本项目还没有验证同一GPU上两个SAPIEN进程、fixed eval与
offload的组合，因此仍应做一个短smoke和计时，而不是直接改formal。

RLinf也已有`AsyncRLTACFSDPPolicy`和async runner部件，但它不是等价开关：当前async SAC循环没有调用同步RLT的
20k replay warmup、30k post-collect update、pending update和1600 cap那套schedule，而且async env不允许当前的
offload。因此async版本需要迁移这些训练口径后才可比较。原论文确实采用异步采集与学习，长期看值得做，但优先级
低于同卡双EnvWorker。

当前两条任务并发时不适合直接加worker：主存已经到240 GiB。应在单任务、固定8 env总量下做短性能实验，避免
同时改变采样预算。

## 10. 效率优化优先级

1. **测试同卡`2 EnvWorker × 4 env`。** 总env仍为8、episode/update预算不变，直接缩小已确认的reset串行域。
   同时用RLinf官方trace记录几个cycle，分解env、rollout、actor/critic与同步等待：
   [Profiling文档](https://rlinf.readthedocs.io/en/latest/rst_source/guides/profile.html)。
2. **再测microbatch512。** 当前512/256需要2次累积；512/512只需1次。由于update只占约15--20%，即使
   update接近翻倍，整cycle预期也只改善约6--10%，所以它是次级优化。
3. **单任务测试`no-offload + overlap_env_bootstrap`。** RLinf能在actor training期间预取下一轮bootstrap，
   但官方说明该开关只在`env.enable_offload=false`时生效；当前为true，而且双任务RAM已经到240 GiB，不能直接
   放进当前pair。[Embodiment配置文档](https://rlinf.readthedocs.io/en/latest/rst_source/guides/embodiment_config.html)
4. **验证RLT async/env-decoupled路径。** 它有机会把collect和learner重叠，是结构上最大的提速点；同时需要记录
   policy staleness和replay增长节奏。RLinf系统论文报告其多阶段流水线在若干任务上有`1.1--2.13×`系统加速，
   但不是对本任务的保证：[RLinf系统论文](https://arxiv.org/abs/2509.15965)。
5. **最后看compile/CUDA graph。** 当前关闭；它主要优化update/推理算子，不解决reset锁和同步阶段空洞。
6. **把评估成本单列。** 降低频率或异步独立评估会缩短formal墙钟，但会改变checkpoint/eval协议，应作为运行协议
   选择，不与算法加速混称。

不建议直接把总env从8增加到16：那会同时改变每cycle数据量、replay增长、更新量和RAM，不再只回答系统效率问题。

## 11. 最终判断

目前最大的闲置并不是“还有55 GiB显存，所以把所有并行度翻倍即可”，而是同步阶段切换造成GPU时忙时闲，以及
单EnvWorker/单rollout rank对8个环境的集中服务。最值得做的是固定8 env和训练数学，先测试同卡双EnvWorker并
profile；microbatch512是次级优化；async RLT需补齐原schedule语义后再做。这个顺序比直接增加总env或只看显存
推断吞吐更可靠。
