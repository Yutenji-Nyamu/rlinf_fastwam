# RLT 单卡配对实验：语义、配置与启动

## 1. 结论

RLT 单卡在模型容量上没有问题。历史两卡使用`FSDP=no_shard`，每张卡本来就各持完整student、critic与
rollout π0；改成单卡不会把两份模型参数叠到一张卡。历史每卡约19 GiB，本轮单卡预计约27--40 GiB，
A800 80 GiB有余。

本轮建立新的单卡比较协议：GPU0跑原始RLT，GPU1跑RLT + teacher-DVAC `[0.5,1.5]`；两者都fresh
`1→480`。主要变化是并行执行方式和per-rank replay总量，不是模型或优化目标装不下。

## 2. 两卡改成单卡后怎样变化

| 项目 | 历史两卡 | 新单卡（每个实验） | 含义 |
|---|---:|---:|---|
| actor / rollout / EnvWorker | 各2个rank | 各1个rank | 少一个并行rank |
| train env | 每rank 4，全局8 | 单rank 8，全局8 | 每cycle交互预算不变 |
| eval env | 每rank 2，全局4 | 单rank 4，全局4 | 每次仍为`4×5=20`条 |
| global / micro batch | `512 / 128` | `512 / 128` | optimizer batch不变 |
| gradient accumulation | 2 | 4 | 单卡串行多累积两次 |
| warmup / replay window | 每rank `10k / 50k` | 每rank `10k / 50k` | 新单卡A/B彼此相同 |
| max cycle | 480 | 480 | 总训练预算不变 |

梯度累积由代码动态计算：

\[
\mathrm{accumulation}=\frac{512}{128\times\mathrm{world\ size}}.
\]

因此world size从2变1时自动从2变4，四个microbatch累积后才做一次optimizer step；有效global batch仍为
512。[动态计算位置](../../tmp/rlt_dvac_impl_source/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py)

Replay按actor rank持有。字面配置保持`10k/50k`意味着历史两卡合计约`20k/100k`，新单卡为
`10k/50k`。用户选择从新的单卡baseline开始，所以control与DVAC保持同一字面配置，不额外引入第二个参数
变化。两卡checkpoint记录`actor_world_size=2`并由代码校验，不能直接恢复到单卡；两条新实验因此都fresh，
以后单卡`1→1` resume仍受支持。

## 3. 配对配置合同

两条共同使用：

- RoboTwin `adjust_bottle`，同一Stage-1、π0 teacher、norm stats、student、Q/BC与seed 1234；
- train `8 env × 1 epoch`，eval `4 env × 5 epoch=20`；
- `max_steps=480`，每25 cycle评估并保存；
- global/micro batch `512/128`，单卡accumulation 4；
- `update_epoch=5`、critic:actor=`2:1`、每cycle最多1,600次update；
- warmup 10,000，replay cache/window 50,000；
- placement内均写逻辑GPU 0，由进程外`CUDA_VISIBLE_DEVICES`选择物理卡。

唯一方法差异：

| 字段 | Control | Teacher-DVAC |
|---|---|---|
| DVAC | off | apply |
| `selected_l / applied_horizon` | 不使用 | `3 / 10` |
| `z_clip / strength` | 不使用 | `2 / 0.25` |
| action梯度权重 | 恒为1 | `w=1+0.25 clip(z,-2,2)∈[0.5,1.5]` |

DVAC只调整student action经Q返回actor的逐`h`反向贡献；前向action、Q值、critic TD loss与BC分支仍走
原路径。

## 4. 两套Ray为何需要显式隔离

当前Ray 2.55.1的`address="auto"`会扫描整机GCS；只分开`RAY_TMPDIR`仍可能让第二条driver误连第一条。
两个隐式local runtime还会竞争默认dashboard-agent端口52365，并分别按整机申请CPU/Plasma。

本轮不改RLinf算法代码，而是为每个实验预启动一个external Ray head：

| 项目 | Control | DVAC |
|---|---|---|
| 物理GPU | 0 | 1 |
| GCS/系统端口 | 46001--46009 | 47001--47009 |
| worker端口 | 46100--46599 | 47100--47599 |
| Ray资源声明 | `GPU=1, CPU=18` | `GPU=1, CPU=18` |
| Plasma | 24 GiB | 24 GiB |
| temp / spill / run root | 独立 | 独立 |

driver连接具体GCS地址；启动探针会分别断言`GPU=1, CPU=18`。正常训练退出只清理自身Ray head PGID，
不调用会全机扫描的`ray stop`。资源监控每2秒只记录两卡、cgroup RAM/PSI/OOM事件与各进程组RSS，不设置
资源阈值，也不改变训练。

## 5. 实际执行与证据

- 服务器配置/source commit：`74c715515c94fd367aff274871bf9488e95ff6b3`，已推送
  `personal/codex/rlt-teacher-dvac-weighting`。
- Hydra真实compose与逐字段合同检查：`SINGLE_GPU_AB_CONTRACT_OK`；unexpected A/B mismatch为0。
- 旧teacher-DVAC `[0,2]`已自然完成`480/480`并`exit_code=0`；用户随后改为**暂不启动**新单卡A/B，
  两条新训练均未启动，GPU已空闲。单卡配置与提交仅保留为后续候选。
- 旧实验最终精简包：
  `exports/rlt_teacher_dvac_w0to2_formal_fresh480_high_info_20260825_v2.zip`（12,982,235 bytes，
  SHA256 `9c49aed8049c08e8cff093df3782ac5220143ecbb87f78720803aaf90872cf61`）。

完整逐指令、问题与窄修复见
[启动实施账](evidence/RLT_SINGLE_GPU_DUAL480_LAUNCH_LEDGER_20260825.md)。
