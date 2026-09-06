# RLT success-episode DVAC-BC 双单卡 smoke v9

日期：2026-08-25  
服务器：AutoDL，2×A800 80GB  
source：`codex/rlt-dvac-success-episode-bc@64f2779f266b7d7019895c7aee1ebc222312b7d3`

## 1. 终态

| 项目 | 单卡原始 RLT control | 单卡 success-episode DVAC-BC |
|---|---:|---:|
| 物理 GPU | 0 | 1 |
| driver | exit 0 | exit 0 |
| wall time | 14分22秒 | 14分17秒 |
| train episode | 8 | 8 |
| train success | 0/8 | 1/8 |
| fixed eval | 0/20 | 0/20 |
| replay transition | 160 | 151 |
| critic / actor updates | 8 / 4 | 8 / 4 |
| `global_step_1` | complete，world size 1，update step 8 | complete，world size 1，update step 8 |

这是机制 smoke，不比较效果；8 条 stochastic train 和 20 条 fixed eval 太少，而且两个 run 的 rollout 已经
因为随机行为不同而产生 160/151 条 transition。

## 2. 方法分支确实生效

方法版最终 TensorBoard 标量：

- `application_success_episode_bc=1`、`application_q_gradient=0`；
- baseline count/mean/std=`1510 / -4.899 / 0.563`，已经冻结并进入 apply；
- action 权重 mean=`1.000`，p05/median/p95=`0.741/1.006/1.264`；
- weight ESS=`0.975`，top-20% weight mass=`0.241`；
- down/up-weighted fraction=`48.69% / 51.31%`；
- success executed-target ratio=`7.78%`；成功权重均值=`1.000`、成功权重ESS=`0.982`；
- 4份 actor trace NPZ 已产生，本地保留最后一份代表样本。

本轮 `actor_switch_rate=0`，说明初始 warmup 仍全部执行 π0 reference route；因此成功轨迹里的 executed action
与 reference action 相同，`success_executed_ref_mse=0`。这不影响验证 target gate、per-h 加权、replay、
backward 和 checkpoint 链路；正式训练进入 student route 后，该距离才会变为更有信息量的机制指标。

## 3. 资源

- GPU0 / GPU1 峰值=`21,223 / 21,194 MiB`；两卡均到过100%利用率；
- cgroup RAM峰值=`83,896,451,072 bytes`，约78.1 GiB；
- `memory.high/max/oom/oom_kill`全程均为0；PSI采样为0；
- 结束后两卡均为0 MiB计算占用；driver、Ray head、raylet和训练worker均退出；
- 两个run各约46 MiB，轻量smoke没有空间压力。

## 4. v8暴露的问题与v9窄修

v8已证明namespace和GPU placement正确，但首批transition写replay时：control报`actor_switch` KeyError，
方法版报`episode_success` KeyError。原因是终态直接令`next_obs=curr_obs`，把当前动作训练元数据带进了固定
replay next-observation schema。

v9把所有 replay `next_obs`投影为模型真正使用的三项：`z_rl/proprio/ref_chunk`。DVAC、route和成功标签继续
保留在`curr_obs`；critic TD、终态mask与actor方法语义不变。服务器 format/lint/compile通过，目标单测
`10 passed`。

## 5. resolved 参数核对

- 历史成功双卡resolved → 单卡control smoke：18个叶子不同，全部属于单卡placement、smoke的一步/短
  warmup、fresh而非resume250、source/run路径与命名；其余叶子一致。
- 单卡control → 单卡方法版：13个DVAC方法字段、GPU0→1、独立输出路径/命名，以及OpenPI telemetry开关；
  未发现额外算法参数漂移。

正式480-cycle配置不会继承smoke的短warmup与1-step interval；它们只是为了让一次smoke走过真实update。

## 6. 本目录材料

```text
control/
  resolved.yaml
  exact_command.txt
  foreground.log
  events.out.tfevents
  replay_metadata.json
  rlt_trainer_state_complete.json

method/
  resolved.yaml
  exact_command.txt
  foreground.log
  events.out.tfevents
  update_00000006.npz
  replay_metadata.json
  rlt_trainer_state_complete.json

pair/
  paired_resources.csv
  final_summary.json
  launch_summary.txt
  ray_cluster_resources.json
  historical_two_gpu_vs_single_control_leaf_diff.json
  control_vs_success_bc_method_leaf_diff.json
```

服务器大checkpoint与全部replay正文未复制到Windows；本目录只保留可复核的轻量证据。
