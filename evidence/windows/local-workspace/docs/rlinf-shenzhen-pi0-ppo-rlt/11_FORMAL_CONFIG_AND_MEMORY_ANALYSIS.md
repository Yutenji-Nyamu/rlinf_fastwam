# 深圳 formal PPO：官方参数缩放、旧两卡对照与 EnvWorker 内存分解

> 口径：`SZ-H100` 现场只读检查；同相位内存分析已于 2026-08-22 14:59–15:02 CST 刷新到
> 完整 Step 33 / 下一步 `0/4`。该刷新推翻了早先“可能趋近平台”的暂定解释；动态运行状态以后仍需
> live 刷新。本轮没有改配置、停止进程、写服务器文件或触发新的训练。

## 1. 证据边界

- current official source：
  `/data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin@7d07a4212ee6858cc333e1d4fab7a37256d1f839`。
- official YAML：`examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml`，
  SHA256 `7ffd734f1e57cbd830fbafea392b58d0e150d88859967bf4f26bf0740acdc335`。
- formal resolved：
  `/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1/resolved.yaml`，
  SHA256 `48b4be79af300512d757ce47219ea2b27445155dbd6438be300459756b266926`。
- 旧两卡 GRPO 的 source/resolved 证据保存在 AutoDL 专题；旧 PPO 的精确私有 YAML 与启动命令保存在
  `pi0 + ppo_grpo.md` 历史材料。它们用于解释工程预算，不把旧动态状态冒充深圳现场。

## 2. official 8 卡到当前 4 卡究竟缩了什么

| 维度 | current official 8×GPU | 深圳 formal 4×GPU | 缩放 | 每个 rank 是否变化 |
|---|---:|---:|---:|---:|
| actor/env/rollout ranks | 8 | 4 | `1/2` | — |
| train env | 256 | 128 | `1/2` | **32 → 32，不变** |
| eval env | 128 | 64 | `1/2` | **16 → 16，不变** |
| train rollout epochs | 4 | 4 | 不变 | 不变 |
| eval rollout epochs | 1 | 1 | 不变 | 不变 |
| max primitive steps / episode | 200 | 200 | 不变 | 不变 |
| policy horizon `H` | 50 | 50 | 不变 | 不变 |
| executed chunk `C` | 50 | 50 | 不变 | 不变 |
| global batch | 2048 | 2048 | **不变** | — |
| micro batch / rank | 32 | 32 | 不变 | 不变 |
| update epochs | 2 | 2 | 不变 | — |
| max outer steps | unbounded `-1` | 100 | 只限定运行长度 | — |
| val/save interval | 10 / 10 | 10 / 10 | 不变 | — |

所以这不是“每卡并发减半”，而是把 official 的 rank 数和全局 env 数同时减半，保持完全相同的
per-rank 密度。当前每个 EnvWorker 同时常驻 `32 train + 16 eval = 48` 个 RoboTwin simulator。

路径替换、physical GPU `0-7 → 4-7` 和 `max_steps=100` 是部署/预算变化，不是 PPO objective 变化。
KL、clip、GAE、actor/value LR、FSDP、patch sync、模型、三相机与 14D action 语义均保持 official。

### 每个 outer step 的真实预算

`H=C=50` 且每 episode 200 primitive steps，因此每条 trajectory 最多有 `200/50=4` 个
action-query record。

| 预算 | official 8 卡 | 深圳 4 卡 | 深圳 / official | 每 rank（两者相同） |
|---|---:|---:|---:|---:|
| trajectories / outer step | `256×4=1024` | `128×4=512` | `1/2` | 128 |
| primitive simulator steps | 204,800 | 102,400 | `1/2` | 25,600 |
| action-query records | 4,096 | 2,048 | `1/2` | 512 |
| sample presentations (`records×update_epoch`) | 8,192 | 4,096 | `1/2` | 1,024 |
| distributed optimizer updates | `(4096/2048)×2=4` | `(2048/2048)×2=2` | `1/2` | — |
| gradient-accumulation microsteps / update / rank | `2048/(32×8)=8` | `2048/(32×4)=16` | `2×` | — |
| total microsteps / rank / outer step | `8×4=32` | `16×2=32` | 不变 | 32 |

这也解释了为什么“全局 batch 2048 没缩”仍然一致：全局 records 减半后，每个 outer step 从 4 次
distributed update 变成 2 次，但每个 rank 接收的总样本与累计 microsteps 不变。

## 3. 与 AutoDL 旧两卡 PPO / GRPO 对照

| 维度 | 旧 2 卡 PPO | 旧 2 卡 GRPO | 深圳 4 卡 PPO |
|---|---:|---:|---:|
| train env | 32（16/rank） | 16（8/rank） | 128（32/rank） |
| rollout epochs | 8 | 16 | 4 |
| trajectories / outer step | 256 | 256 | 512 |
| trajectories / rank | 128 | 128 | 128 |
| primitive steps / outer step | 51,200 | 51,200 | 102,400 |
| action-query records / outer step | 1,024 | 1,024 | 2,048 |
| global / micro batch | 512 / 32 | 512 / 32 | 2048 / 32 |
| update epochs | 2 | 2 | 2 |
| distributed updates / outer step | 4 | 4 | 2 |
| total actor microsteps / rank / outer step | 32 | 32 | 32 |
| group size / advantage | 1 / GAE | 8 / GRPO | 1 / GAE |
| train video | false | false | true（current official resolved） |
| inline eval | disabled | disabled | fixed-64 every 10 steps |

关键点不是“4 卡还比 2 卡慢”，而是：

1. 深圳每个 outer step 的全局 simulator 量是旧两卡的 **2 倍**；GPU 数也正好是 2 倍。
2. 每个 rank 仍承担 128 trajectories、25,600 primitive steps、512 records 和 32 个 actor
   microsteps，和旧两卡完全同量。因此理想预期是**相近 step wall time、约 2 倍全局吞吐**，不是
   step wall time 再减半。
3. 现场普通 step 中位数约 26.1 分钟；旧成功两卡 GRPO 的记录是约 24–26 分钟/step，实际上同档。
4. 当前 Step 21 的 `generate_rollouts≈1561.7s`，actor training 仅约 `23.2s`。瓶颈是
   RoboTwin reset/render/primitive interaction 与同步等待，不是 H100 上的 optimizer；换更快 GPU
   不会按算力比例缩短 simulator 时间。
5. 当前还沿 official 保存 train video，并每 10 步做 fixed-64 eval/checkpoint；旧私有两卡配置关闭
   train/eval video 且不做 inline eval，所以周期步会额外慢一些。

## 4. 内存：正常的常驻部分与异常增长信号

### 4.1 同相位跨 step 结果

| 时间（CST） | phase | cgroup | 4×EnvWorker 口径 | host MemAvailable |
|---|---|---:|---:|---:|
| 10:08:21 | 完整 Step 22，进入下一步前后 | 1.3966 TiB | RSS 1324.07 GiB | 597.00 GiB |
| 10:27:18 | Step 23 rollout 3/4 后 | 1.4375 TiB | RSS 1365.93 GiB | 555.08 GiB |
| 10:33:28 | Step 23 rollout 4/4 完成、update 前后 | 1.4516 TiB | RSS 1364.84 GiB | 540.61 GiB |
| 10:35:04 | **完整 Step 23，Step 24 已开始** | **1.4332 TiB** | **RSS 1362.02 GiB；PSS 1359.51 GiB** | **559.63 GiB** |
| 11:51:47–11:52:02 | **完整 Step 26，下一步 rollout 0/4** | **1,602,268,200,960 B（1.457254 TiB）** | **PSS 1,450,109,793 KiB（1382.93 GiB）** | **562,086,232 KiB（536.05 GiB）** |

outer-step 边界确实释放约 18.9 GiB cgroup 内存，说明 rollout/actor 临时 buffer 有正常锯齿。完整
Step 22 到完整 Step 23 时，基线曾净增 **37.50 GiB**，EnvWorker RSS 净增约 **37.95 GiB**。

后续同相位 Step 23→Step 26 的增速已明显放缓：Step 23 只保存了四位 TiB/两位 GiB 舍入值，
因此下列是**基于 rounded baseline 的近似差值，不是伪造的 exact delta**：

- cgroup：`1.457254 TiB - 1.4332 TiB` ≈ **+24.63 GiB / 3 steps = +8.21 GiB/step**；
- EnvWorker PSS：`1382.9325 - 1359.51` ≈ **+23.42 GiB / 3 = +7.81 GiB/step**；
- MemAvailable：`536.0472 - 559.63` ≈ **-23.58 GiB / 3 = -7.86 GiB/step**。

这与 Step 22→23 的单步约 38-GiB 跃升相比，更像 allocator 高水位在趋近平台；但仍是正增长，
且只有三个后续边界，还不能证明已稳定。

目前 `swap=0`，本次 cgroup `high/max/oom/oom_kill=0`，训练仍在运行；所以这是**增速明显放缓、
趋近平台但尚未证明稳定的黄灯**，不是已发生 OOM。不应用单个 step 的早期斜率线性外推最终 OOM 时间。

### 4.2 PSS / anonymous 分解

10:35 的四个 EnvWorker：

| worker | RSS | PSS | PSS anonymous | PSS file | private dirty |
|---|---:|---:|---:|---:|---:|
| 1383912 | 370.91 GiB | 370.29 GiB | 369.96 GiB | 0.310 GiB | 370.07 GiB |
| 1384178 | 301.02 GiB | 300.39 GiB | 300.06 GiB | 0.310 GiB | 300.17 GiB |
| 1384179 | 360.06 GiB | 359.43 GiB | 359.10 GiB | 0.311 GiB | 359.21 GiB |
| 1384181 | 330.03 GiB | 329.41 GiB | 329.08 GiB | 0.309 GiB | 329.19 GiB |
| **合计** | **1362.02 GiB** | **1359.51 GiB** | **1358.20 GiB** | **1.24 GiB** | **1358.64 GiB** |

11:52 的本次原始 `smaps_rollup` 摘要（worker 按脚本输出顺序）：

| worker | PSS (KiB) | PSS | PSS anonymous (KiB) | PSS anonymous |
|---|---:|---:|---:|---:|
| 1 | 388,449,268 | 370.454 GiB | 388,103,328 | 370.124 GiB |
| 2 | 334,949,846 | 319.433 GiB | 334,603,960 | 319.103 GiB |
| 3 | 376,291,730 | 358.860 GiB | 375,945,096 | 358.529 GiB |
| 4 | 350,418,949 | 334.186 GiB | 350,073,588 | 333.856 GiB |
| **合计** | **1,450,109,793** | **1382.932 GiB** | **1,448,725,972** | **1381.613 GiB** |

EnvWorker PSS 占 cgroup current 约 **92.6%**；每个 worker 的 PSS 与 RSS 几乎相等，而且几乎全部是
private anonymous/private dirty。结论：

- 不是把同一共享库映射重复计算四遍造成的 RSS 假高；
- 不是 Linux page cache 或 checkpoint file cache 主导；
- Ray/plasma、actor、rollout worker 不是 1.36-TiB 主体；
- 主体在 RoboTwin EnvWorker 自己的 Python/native heap，最可能落在 simulator/renderer/reset 与图像
  buffer 的生命周期。

### 4.3 source 生命周期与最小候选排序

1. **最主要：长期常驻 simulator + reset 后 native allocator/cache 不归还。**
   每个 EnvWorker 在 `init_worker()` 一次创建 32 train 与 16 eval env，两个 `VectorEnv` 和内部
   `SubEnv/task/thread pool` 随 worker 全程常驻。resolved 的 nested
   `env.train.enable_offload=false`、`env.eval.enable_offload=false` 才是 worker 实际读取的字段；
   official 顶层 `env.enable_offload=true` 不会销毁这两组 env。32+16 个常驻 simulator/rank 能解释
   很高的基础工作集。

   每次 reset 已按 resolved `clear_cache_freq=1` 执行 `task.close_env(clear_cache=True)`；RoboTwin
   会把 viewer/camera/robot/scene/renderer/engine 引用置空并调用 `sapien.render.clear_cache()`。
   早期 Step 22→23 同相位增长约 38 GiB，但 Step 23→26 已放缓到约 8.2 GiB/step。这与
   SAPIEN/Vulkan/C++/glibc allocator retention、fragmentation 或 native cache 逐渐接近高水位相容。仅凭
   smaps 不能再区分“可复用但不返还 OS 的高水位”与真正 leak，不能越界定性。

2. **次要且能解释锯齿：trajectory / observation / video 临时 buffer。**
   非 pipeline 路径会让 `self.rollout_results` 跨 4 个 rollout epoch 累积，发送给 actor 后才
   `clear + gc.collect()`；这与 Step 23 update 边界约 19 GiB 回落一致。

   `RecordVideo` 在每个 epoch 内缓存 tiled frames，`finish_rollout()` 后把 `render_images=[]`，等待 MP4
   writer 完成；future 列表也会 prune。它不像一个仍被 Python list 永久引用的直接 leak，但当前 official
   开启 train/eval video，反复大块 ndarray 分配可能加剧 native heap 高水位或碎片。需要受控 A/B 才能
   判断占比。

3. **较低优先级：actor/rollout offload、Ray shared memory、文件缓存。**
   它们合计远小于 EnvWorker，且 smaps/cgroup 的 file/shared 口径与 observed slope 不匹配；它们不能解释
   已观察到的 private-anon 基线增长。

## 5. 11:52 运行快照与当前判断

- wrapper/driver 与 Ray 仍 alive；日志已有完整 Step 26，已进入下一步 `rollout 0/4`；
  fatal traceback/OOM/worker-crash 命中数为 **0**。
- Step 25 TensorBoard：`success=0.896484375`、`KL=0.015977446`、`clip_fraction=0.06515204`、
  `grad_norm=31.20695`、`critic_explained_variance=0.42353374`、`value_loss=0.02522251`；
  `generate_rollouts=1506.533s`、`actor_training=23.232s`、`step=1531.659s`。数值为有限值，形态与前述正常运行一致。
- GPU 4–7 显存依次为 **59,564 / 60,206 / 59,438 / 59,118 MiB**；GPU 0–3 本次无 compute 进程。
- `global_step_10` 与 `global_step_20` checkpoint 各 **18,475,261,914 B**。
- 数值训练链正常；内存链的准确描述是“**巨大但尚可运行的 EnvWorker private-anon 常驻集；
  增速明显放缓、趋平台，但未证明稳定**”，维持**黄灯**。
- 下一次仍只需取同相位完整 step 边界，不要混入 rollout 中途点拟合斜率。如果后续边界继续放缓并
  摆动，可判为 allocator 平台；如果重新恢复数十 GiB/step 的连续增长，再讨论利用现有 checkpoint
  做窄的 video-off / env-lifecycle 对照。当前没有停止或改配置授权，本轮不实施。

### 5.1 Fast-WAM 闭环后的 12:55 单点刷新

- driver/Ray 仍 alive；已有完整 Step 28，正在下一 step 的 train rollout `2/4`；`fatal_match_count=0`。
- Step 27：`success_once=0.884765625`、`KL=0.01246691`、`clip_fraction=0.05894747`、
  `grad_norm=30.98494`、`critic explained variance=0.42312`、`step=1558.5s`；数值仍为有限值。
- GPU 4–7 为 `63,738 / 63,006 / 65,178 / 64,404 MiB`；Fast-WAM 退出后 GPU 0–3 无 compute 进程。
- cgroup 为 `1,658,459,029,504 B`（`1.50836 TiB`），4 个 EnvWorker PSS 合计
  `1,506,723,187 KiB`（`1436.92 GiB`），`MemAvailable=505,476,936 KiB`（`482.06 GiB`）；
  `high/max/oom/oom_kill=0`。该点位于 rollout `2/4`，与 11:52 的 step 边界不同相，只说明内存仍高但未触发
  cgroup 压力/OOM，**不用它重算跨-step 斜率**。黄灯判断不变。

### 5.2 15:00 同相位刷新：增长未平台化，进入需决策的高风险状态

本节证据见
[`evidence/08_SERVER_PPO_LIVE_REFRESH_20260822.md`](evidence/08_SERVER_PPO_LIVE_REFRESH_20260822.md)，
并**取代**上文基于 Step 23→26 三步所作的“可能趋近平台”暂定解释。

- driver/Ray 仍 alive；console 已完整输出 Step 33/100，随后进入下一步 train rollout `0/4`；
  fatal traceback/OOM/worker-crash 命中数为 0。
- Step 33：train success `87.50%`、KL `0.013`、clip fraction `0.044`、grad norm `34.882`、
  critic explained variance `0.420`、value loss `0.031`；step `1542.6s`，其中 rollout `1517.3s`、
  actor training `22.929s`。训练数值仍有限且没有发散形态。
- Step 10/20/30 的 fixed-64 inline eval 为 `58/64`、`62/64`、`58/64`；不是单调提升，也没有崩坏。
- 同为“完整 step、下一步 `0/4`”相位，Step 26→33 的 7-step 精确差值为：cgroup
  `+198.09 GiB`（`+28.30 GiB/step`）、四个 EnvWorker PSS `+191.87 GiB`
  （`+27.41 GiB/step`）、host MemAvailable `-191.07 GiB`（`-27.30 GiB/step`）。
- 当前 cgroup `1.651 TiB`，四个 EnvWorker PSS 合计 `1.538 TiB`，host MemAvailable 约
  `345 GiB`，swap 为 0，`high/max/oom/oom_kill=0`。增长主体仍是 EnvWorker private anonymous memory。

因此当前准确判断是：**训练链正常，但内存增长没有平台化，继续无人值守到 Step 100 风险过高。**
最近 7 步斜率不能精确预报 OOM step，也不能保证未来保持线性；它只足以排除“已稳定平台”这一解释。
已有完整 Step 30 checkpoint。是否立即停止、观察到下一个自然 Step 40 checkpoint 后停止，或承担风险继续，
需要用户明确决定；本轮只读审计没有控制进程。

### 5.3 18:10 Step 40同相位复核：自然停止点已经到达

- `global_step_40`与fixed-64 eval均完整；eval=`62/64`，Step40 train success=`92.58%`，KL/clip/grad/
  critic EV均finite，fatal与cgroup OOM events为0。
- Step40进入下一步`0/4`时，cgroup=`1,920,436,883,456 B≈1.747 TiB`，host
  `MemAvailable=266,727,452 KiB≈254.4 GiB`。
- 相对Step36同为next`0/4`，四步内cgroup约`+84.8 GiB`（`+21.2 GiB/step`），MemAvailable约
  `-75.8 GiB`（`-18.95 GiB/step`）。增长没有平台化。
- 18:26的rollout中段只读点已到Step41 `2/4`，MemAvailable进一步降至约`227.9 GiB`，四卡约
  `67.8–68.7 GiB/卡`，仍无OOM；该中段点不用于重新拟合同相位斜率，只说明风险仍在继续。

因此停止建议已经从“等到Step40再判断”收敛为：**使用完整Step40 checkpoint/eval，现在停止exact owned
formal driver，不再等Step60。** 停止仍是进程控制，需要用户明确回复；本次文档更新没有发送信号。
完整Step1–40曲线与原始轻量证据见
[`evidence/09_PPO_CURVES_LIVE_20260822.md`](evidence/09_PPO_CURVES_LIVE_20260822.md)。

## 6. 本轮只读命令账

以下 command-file 均通过固定 host key 的 Paramiko 密码通道，以 `chenyiteng` 执行；密码只进入当前
进程提示，没有写入文件。命令只读 `/proc`、cgroup、source/config/log，没有服务器写入：

| command-file | SHA256 | 用途 |
|---|---|---|
| `local_scripts/remote_commands/shenzhen_rlinf_formal_config_memory_deepread_20260822.sh` | `bc540ff3db495c5ebfbeece351abe9a3c00b6fc85d4a48ee735803d75a776d6c` | official/resolved/manifest、进程树、cgroup 与首轮 smaps |
| `local_scripts/remote_commands/shenzhen_rlinf_memory_source_inspect_20260822.sh` | `15ada7f412f078dfd5b7cc45d4e519f0abbb6dbb7f3c91c40b409b4bd88e959` | RecordVideo 与 RoboTwin cache 引用定位 |
| `local_scripts/remote_commands/shenzhen_rlinf_memory_source_details_20260822.sh` | `5a77219ecff521549e2fd42dbff650fcf84f5501fe762f19dce17b9a8bb42e4a` | flush/reset/VectorEnv 生命周期精确源码 |
| `local_scripts/remote_commands/shenzhen_rlinf_step23_boundary_memory_20260822.sh` | `73fb091cc933e0e24897f5d3a204f4740c6f186e116a4ebb36582a01fef7eb16` | Step 23 rollout 与完整 metric 边界的 cgroup/PSS 快照 |
| `local_scripts/remote_commands/shenzhen_rlinf_close_env_source_20260822.sh` | `5be7838da3a7234f21becb3e81a754a647570aa37e02596ebf9896b61f6ededd` | `close_env()`、Sapien clear-cache 与 offload 调用链 |
| `local_scripts/remote_commands/shenzhen_rlinf_formal100_post_download_single_snapshot_20260822.sh` | `f02b530bd4f26739167993785f6a02323fd6077c11c2fbf97fabc08892b2f14a` | 在 Step 26 边界与 Fast-WAM 闭环后 Step 28/next-rollout 2/4 各执行一次单点只读快照 |
