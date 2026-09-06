# Idea2：DVAC 信号进入 π0-GRPO 训练的最小增量路线

最后更新：2026-08-22  
当前状态：**v1 `[0.8,1.2]`训练实现与2-step smoke已完成，100-step formal按用户授权主动停止于
完整g54；v2 R-only `[0.5,1.2]`已实现/前测并完成真实2-step smoke，获批的fresh-SFT 100-step
formal已启动，当前尚无完整Global Step。**
权威 Idea2 父提交为 AutoDL `61996e15...`，训练实现使用独立 child branch/worktree。

本文件是训练修改的唯一设计真值。推理 telemetry 合同见
[01_SIGNAL_AND_DATA_CONTRACT.md](01_SIGNAL_AND_DATA_CONTRACT.md)，64-query 首轮事实见
[04_FIRST_DATA_ANALYSIS.md](04_FIRST_DATA_ANALYSIS.md)，实施与 smoke 逐指令记录见
[TRAINING_IMPLEMENTATION_AND_SMOKE_LEDGER.md](evidence/TRAINING_IMPLEMENTATION_AND_SMOKE_LEDGER.md)。

## 1. 首版方法：一句话和一张图

一句话：**在不改变原 GRPO 的 chunk reward、advantage、joint ratio 和 joint clipping 的前提下，
用训练 rollout 里的 DVAC `V_L3(h)` 让高不确定 future action 对反向更“大声”一点，低不确定
action 更“小声”一点。**

```text
π0 train rollout (flow_sde, 原有4次velocity forward)
  x_i, v_i
    └─> z_i = x_i - t_i v_i
         └─> V_L2/L3/L4 [query, h=0..49]
              └─> 最近5个已完runner step的global log-V均值/标准差
                   └─> w[h] in [0.8, 1.2]

actor replay
  new_logprob [query, 50, 14]
    └─> 前向数值不变，反向时第h位贡献×w[h]
         └─> 原chunk sum -> 原joint ratio/clip -> 原GRPO advantage
```

第一个 completed runner step 的DVAC权重为`w=1`并收集统计，原GRPO本身照常训练；第二步开始真正
使用非均匀权重。这样一个
2-step smoke 就能同时走过 warmup 和 apply 两条路径。

## 2. 从名词到梯度：这次到底改了什么

| 名词 | 在这个项目里的意思 |
|---|---|
| action slot / `h` | 一次生成的50个未来动作中的第几个，`h=0..49` |
| denoise step / `i` | 从噪声 action tensor 到 clean action 的第几次 flow 更新，当前`i=0..3` |
| log-prob | 模型对实际采样 action 给出的对数概率密度；它的导数把更新传回模型 |
| advantage | GRPO 根据同组 rollout 结果决定这条样本该强化还是抑制 |
| gradient | 当前 loss 要把每个参数往哪个方向推、推多少 |
| PPO ratio | 新旧 policy 对同一 chunk 的相对概率 |
| PPO clip | 防止一次 policy update 过大的分支选择 |
| global grad clip | backward 完后把整个模型梯度范数限到`1.0`；它不能单独识别`h` |

原训练的一次 query 先做：

\[
\ell_{chunk}=\sum_{h,d}\ell_{h,d},\qquad
r=\exp(\ell_{chunk}-\ell_{chunk}^{old}).
\]

因此到 scalar loss 阶段时，50个`h`已经混在一起。直接把 YAML 改成`action_level`会把joint ratio/
clipping也改成50套，不再是小改。首版改成：

\[
\widetilde\ell_{h,d}
=\operatorname{sg}(\ell_{h,d})
+w_h\left(\ell_{h,d}-\operatorname{sg}(\ell_{h,d})\right).
\]

`sg`可理解为“这份数值只看，不往回传梯度”。这个表达式的前向数值始终等于原`ell`，但反向时
第`h`位变成`w_h`倍。例如：

- `w=1.2`：该 future action 的 likelihood 梯度贡献增加20%；
- `w=0.8`：贡献减少20%；
- `w=1`：与旧训练一样。

这里的20%是进入整模型global grad clip之前、该`h`的likelihood贡献变化，不等于最终每个模型参数
一定移动20%；不同`h`的梯度方向还会相加或抵消，最后整体范数仍按旧配置裁到`1.0`。

advantage 仍决定方向：高V+正advantage会更强地学，高V+负advantage会更强地压制。
DVAC 说“哪里多用一点力”，GRPO outcome 说“往哪个方向用力”。

原 joint PPO clip 也保留：如果整个chunk进入了不给policy gradient的clipped分支，那50个`h`都不会
被这个首版“解锁”。这是保持原PPO语义所必然带来的结果。

## 3. `log(V)`、均值、标准差和权重

首版冻结公式：

\[
y=\log(V_{L3}+\epsilon),\qquad
z=\frac{y-\mu}{\max(\sigma,\epsilon_\sigma)},
\]

\[
s=\operatorname{clip}(z,-2,2),\qquad
w=1+0.1s.
\]

通俗解释：

1. `V`的大小往往跨多个数量级；取`log`会把“乘法差距”变成更容易比较的“加法差距”。
2. `mu`是最近5个已完成runner step中、进入trajectory的4个action-query（不含额外terminal/bootstrap
   forward）之全部`query×h`的`log V`均值；`sigma`是其标准差。这里不按loss mask筛query；真正的
   weight/advantage汇总仍按loss mask统计。
3. `z=1`表示比近期常态高一个标准差，`z=-1`则低一个标准差。
4. 把`z`限到`[-2,2]`，避免极少数outlier决定更新。
5. 乘`0.1`后，权重范围是`[0.8,1.2]`，即最多减/增20%。

`epsilon=1e-12`使`V=0`时仍能取log，`epsilon_sigma=1e-6`使历史分布暂时几乎无方差时仍能计算。
当前step不进自己的baseline：先用旧统计生成权重，训练完成后才把该step推入recent-5队列。

首版**有意保留future-h位置效应**。也就是说，越远的action普遍更不确定，它得到更高权重不一定
是错的；我们先实际训一次再看。落盘时同时保留`h`，离线仍会把 raw `log V`和扣除每个`h`中位数的
residual并排分析，但不让这个分析选择阻塞首次训练。

## 4. 与 Beyond 80/20 的关系

高层思想是一致的：

| Beyond 80/20 | 本项目 |
|---|---|
| 一条文本里的 token 位置 | action chunk 里的 future index `h` |
| next-token entropy | endpoint variance `V_L3(h)` |
| 内部信号定位不确定/分叉位置 | 用去噪“改口”程度定位动作不稳定位置 |
| 重分配 policy-gradient credit | 按`h`重分配GRPO likelihood gradient |

但低层实现不相同。[Beyond 80/20](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a797c2d2e0c1fdabf4d1ab8cd0b465c6-Abstract-Conference.html)
是文本token的top-20%硬mask，未选token基本不给policy gradient，并按选中token数重新归一化。
我们的首版是连续`0.8..1.2`权重，没有屏蔽80%动作，且保留原joint chunk clipping。
如果未来改成top-20% action使`w∈{0,1}`并归一，才更像它的原操作。

近期相关工作也表明不确定性到梯度没有唯一公式：[A3PO](https://aclanthology.org/2026.acl-long.134/)
结合outcome正负和confidence；[STEER](https://aclanthology.org/2026.acl-long.1436/)用连续降权而非hard binary；
[DPPO](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c0749c39aaff9e9e4c91f7118bf21b1e-Abstract-Conference.html)
在diffusion policy里也对denoise transition分配credit，但它的粒度是`i`而非我们的future `h`。
因此连续小幅调整是当前比较干净的第一次实现，不需要再等一个“唯一正确”的公式。

## 5. 训练信号与推理信号的差别

评估 telemetry 是`flow_ode`；历史GRPO rollout是`flow_sde`，每个 batched policy forward 选一个
denoise transition注入SDE，并将`denoise_inds`随trajectory传给actor。因此：

- 训练期的`V_L(h)`必须当场从那条真实train chain计算，不能由actor事后只看clean action重建；
- 训练SDE与评估ODE的绝对数值不直接混为一个分布；
- `denoise_ind`仍全量记录作为诊断轴；首版online baseline按用户冻结的简单路线，跨`denoise_ind`与所有`h`全局池化。

这不会多跑模型forward。那4次velocity本来就要计算，新分支只多做detach、方差和紧凑tensor传输。

## 6. 历史 GRPO 基线与参数血缘

工程锚点是2026-07-15启动、2026-07-17自然完成的100-step `adjust_bottle` π0-GRPO：

- source：`/root/autodl-tmp/RLinf@6d0db56bf26f972cd27fa29535f5eb939e80e5bf`；
- [resolved config](../../audits/20260717-084926-grpo-current/resolved-config.yaml)；
- [启动命令](../../audits/20260717-084926-grpo-current/command.txt)；
- [100-step结构化结果](../../audits/20260717-084926-grpo-current/analysis.json)。

| 类别 | 正式训练原样保持的值 |
|---|---|
| task/model | `adjust_bottle`，同一task-matched π0 SFT |
| action/flow | `H=C=50`、active `D=14`、`M=4`、`flow_sde` |
| reward/advantage | chunk reward、GRPO、group size 8 |
| PPO | chunk logprob、update epoch 2、clip±0.2、KL/entropy coeff 0 |
| rollout | 16 parallel env×16 rollout epoch = 256 trajectories/runner step |
| batch | global 512、micro 32 |
| optimizer | lr `5.6e-6`、Adam `0.9/0.95`、wd `0.01`、global clip_grad `1.0` |
| system | 2×A800、FSDP full-shard、train-env offload、CPU patch sync |
| budget | 100 runner steps、save every 10、fresh SFT |

新增仅是：独立source/output/name、DVAC train mode、`V[h]`传输/记录、recent-step统计、per-h
straight-through挂点，以及默认关闭的细录像。不切`action_level`，不改reward、advantage、ratio/
clip、LR、FSDP或checkpoint逻辑。

历史run到100/100，step100当步on-policy训练rollout success为98.4375%，平均每step约1450秒，
整段约40小时20分。它证明旧工程路径能跑通，但98.4375%不是held-out fixed-ID eval。当前目标是
先跑修改版，不让严格off/apply多重run阻塞实现；以后要下强因果结论时再加同commit对照和多seed。

历史100步的`actor/grad_norm`（FSDP在global clip前返回的整体范数）中位数25.429、均值26.087、
范围8.236–42.256，而配置`clip_grad=1.0`；所以每步整体梯度随后都会被统一缩短。历史
`actor/clip_fraction`均值0.152、范围0.057–0.271，指的是PPO ratio分支中被clip的比例，不是global
gradient clipping比例。首版per-h权重的作用是先改变不同h梯度的相对组成与方向；最后即使整体再缩到
范数1，这个相对方向变化仍可保留。

## 7. 记录什么，不记录什么

每个真实rollout query紧凑保留：

- 身份：`runner_step, rollout_epoch, env_rank, env_slot, reset_id, query_idx, action_slot_start`；
- 信号：`V_L2/L3/L4[h]`、`denoise_inds`、按动作维求和的old log-prob诊断；
- 结果：reward、done、loss mask、advantage及其正负；
- 权重：`history_steps/count/mean/std`、raw logV可由V恢复、`clipped_z[h]`、`w[h]`、warmup/apply标记；
- runner-step汇总：weight分位数、两端clip比例、正/负advantage权重均值和原actor grad/clip/KL/loss指标。

记录为每actor rank独立的compressed NPZ、CSV、manifest和recent-stats state。同一query的weight在两次
actor replay中固定，当前step只统计一次，不因`update_epoch=2`重复入池。

不保存全模型逐action gradient，不为每个`h`单独backward，不在trajectory中长期累积全量raw
`z[4,50,14]`。三套`V[h]`体积很小；对历史100-step上限102,400个进入actor/loss的action queries，单套`[50]`
float32约20.5 MB，三套约61.5 MB，外加元数据仍可控。

## 8. 更细录像的高层和底层设计

高层目标是写一个窄的自定义旁路，复用RoboTwin已有camera API和ffmpeg编码方式，但不直接调用
official direct-evaluator recorder，也不把RLinf的一次C50执行拆成50次
`take_action()`。拆分会改变原本“整段压缩+TOPP+执行”的轨迹，不只是录像。

首版在已有control loop中旁路做：

1. 不重跑path compression、TOPP、observation或policy，不消耗policy RNG；
2. 根据左/右臂已有的同步progress映射到`h=0..49`近似bin，仅在新bin、success或chunk终点取head frame；
3. 每个C50最多约50帧，每episode最多200帧；同时写`query/control/physics/progress/h/success`映射CSV；
4. 训练默认只确定性选`worker0 + slot0 + 第一条episode`，一次不间断driver启动最多产生一支录像；
5. 用H.264、CRF32和160×120记录，画质只要足以辨别moving/contact/success，实际体积由smoke实测。

历史完整训练共25,600 trajectories（4次进入actor/loss的policy query/trajectory，共102,400 queries，
另有25,600次不进入loss的terminal/bootstrap forward）。若全录，将变成25,600支episode和最多
5.12M帧；所以必须抽样。当前数据盘剩余824G，但设计不依靠“空间还多”来
无界写盘。

细录像可以把一个query内的动作、phase和首次success对齐得更清楚，但不会把DVAC时间序列从4个
query点变成200个点；新的`V`仍只在policy query时生成。

## 9. 已完成的实现分解

1. 复用`openpi_action_model.py`已有`return_dvac_telemetry`endpoint旁路，不改去噪/action更新顺序。
2. `huggingface_worker.py`仅在train `mode=apply`时请求endpoint，当场计算float32
   `V_L2/L3/L4[B,H]`，不把raw endpoint带过worker通道。
3. `env_worker.py`给每个query加rollout epoch/env slot/reset/query/action-slot元数据，随同一trajectory对齐。
4. 新`dvac_train_weighting.py`实现公式、recent-step状态、straight-through helper和rank-local writer。
5. `fsdp_actor_worker.py`在shuffle前为query冻结weight，在`[B,H,D]`求和前挂载，一个runner step结束后更新recent stats。
6. RoboTwin `BaseTask`旁路新control trace writer，VectorEnv/RLinf只注入worker/slot身份。
7. 新建一份从历史成功GRPO YAML复制的专用smoke config，不改旧配置。

实施使用独立worktree：

- RLinf：`/root/autodl-tmp/RLinf_idea2_dvac_train`，branch `codex/idea2-dvac-train-weighting`；
- RoboTwin/wamppo：`/root/autodl-tmp/idea2_dvac_train_wamppo`，branch `codex/idea2-dvac-control-trace`。

## 10. 少量高信息量前测和 smoke

服务器前测已覆盖：

1. 新/旧 YAML compose，`mode=off`不请求endpoint且不进actor新分支；
2. `z→V_L2/L3/L4`、warmup/recent5滑窗、global mean/std、精确clip/边界和finite检查；
3. straight-through的forward恒等、all-one旧梯度和非均匀per-h梯度；
4. query/V/metadata经trajectory stack/cat/shuffle后仍对齐；
5. control progress/bin、抽样边界和一个小编码探针。

smoke使用两卡、16 parallel env保持历史已跑通的并行结构，只把串行rollout epoch从16减到8；
每runner step有128 trajectories/最多512个进入actor/loss的action queries，2 steps合计256 trajectories/
最多1,024个actor/loss queries，另有256次terminal/bootstrap forward，
`update_epoch=2`下预计共4个optimizer updates。这比直接2个完整历史step更紧凑，又真正覆盖
step1 warmup→step2 apply。

资源observer每2秒记录GPU、host/cgroup RAM、memory events、`/dev/shm`、主要worker RSS和磁盘；
**它没有阈值、告警动作、timeout、signal或exit-code联动**，只随driver自然结束而结束。

真实smoke已自然完成：step1全部`w=1`且普通GRPO照常更新，step2得到`[0.8,1.2]`非均匀权重；
两卡显存峰值约29.08/28.80 GiB，cgroup峰值82.03 GiB，memory events全0。完整结果见
[06_TRAINING_IMPLEMENTATION_AND_SMOKE_RESULT.md](06_TRAINING_IMPLEMENTATION_AND_SMOKE_RESULT.md)。

## 11. 训练时长怎样选

历史实测是平均约1450秒/runner step：

- 30 steps：约12.1小时，适合先看信号、权重、grad/clip和success走势的pilot；
- 50 steps：约20.1小时；
- 100 steps：约40.3小时，与历史工程预算最直接对齐，但并非“跑12小时”。

因此目前更合理的讨论候选是：**一次不间断30-step pilot**，或直接100 steps与旧curve同预算。
当前v1尚未把recent-5 DVAC统计装入DCP；若从30-step checkpoint另起进程续跑，会使用新的artifact目录并
重新warmup一个runner step，因此不是方法状态的无缝续训。正式值要等smoke的真实wall time、资源和
checkpoint体积后再与用户冻结。smoke实测28分28秒、run约9.7 GiB，但它每step只有历史正式串行预算的
一半，正式耗时仍以历史1450秒/step估算。历史100-step cgroup峰值已达`241999/245760 MiB`，所以正式
并发保持16最可比；资源继续只观察，不设自动干预。

## 12. 阶段和授权边界

| 阶段 | 运行 | 目的 |
|---|---|---|
| T0 | 实现+窄前测（完成） | default-off、shape、公式、对齐和编码正常 |
| T1 | 2-step apply smoke（完成） | step1 warmup和step2非均匀权重均真实走通 |
| T2 | 分析smoke产物（完成） | train-SDE、weight、grad/clip、资源、视频均已核验 |
| T3 | 用户与Codex冻结正式100-step packet（完成） | 用户选择直接100 steps |
| T4 | v1正式修改版（主动停止于54/100） | 保持历史主体参数；g50 DCP和g1–54轻量closeout保留 |
| T5 | 同fixed-ID eval；严格off/apply对照为后续可选 | 区分训练rollout success与控制性能 |
| T6 | v2 R-only fresh-SFT 100-step formal（已启动） | 与历史成功GRPO对齐100 runner steps；只替换per-h residual与`[0.5,1.2]`权重分支 |

用户随后授权在完整g54后停止v1、整理轻量closeout并运行v2两步smoke；这些动作已执行。2026-08-22又
明确授权上述唯一v2 R-only fresh-SFT 100-step formal启动并自然运行。该授权不扩展为停止无关进程、
改变参数、覆盖旧run或删除旧产物；资源观察器仍只读记录，不是训练控制器。

## 13. R-only v2：per-h robust residual 与偏重降权

### 13.1 冻结公式

v1使用全部`h`共享的global mean/std，因此会保留chunk后部天然更高的future-position趋势。v2只把
“同一个`h`上，这次比recent history异常多少”送入权重：

\[
y_{q,h}=\log(V_{L3}(q,h)+10^{-12})
\]

对此前最多5个已经完成的runner step，按每个`h`分别计算：

\[
b_h=\operatorname{median}_q y_{q,h},\qquad
s_h=\max\left(1.4826\operatorname{median}_q|y_{q,h}-b_h|,10^{-6}\right)
\]

\[
R_{q,h}=\frac{y_{q,h}-b_h}{s_h},\qquad u_{q,h}=\operatorname{clip}(R_{q,h},-2,2)
\]

再使用用户冻结的偏重降权映射：

\[
w_{q,h}=\begin{cases}
1+0.25u_{q,h}, & u_{q,h}<0,\\
1+0.10u_{q,h}, & u_{q,h}\ge0.
\end{cases}
\]

所以`u=-2,-1,0,1,2`精确对应`w=0.5,0.75,1.0,1.1,1.2`。它没有per-query mean-one：低于
同位置历史中心的action可以明显降权，高于中心的action最多增权20%。

### 13.2 与成功GRPO保持不变的部分

- step1仍以`w=1`照常完成普通GRPO更新并建立history；current step训练成功后才进入history；
- 信号仍为训练flow-SDE链现成endpoint得到的`V_L2/L3/L4`，online weight选L3，不增加模型forward；
- recent统计仍使用进入trajectory的全部4个action-query×h，不含terminal/bootstrap forward；
- per-h straight-through挂点不变：只改变`new_logprob[B,H,D]`各h的反向倍率；
- chunk reward/logprob、GRPO advantage、joint PPO ratio/clip、global grad clip、任务、SFT、rollout、
  batch、优化器、FSDP、offload与control trace均不变。

### 13.3 实现落点

唯一新RLinf child从v1正式训练提交`145fa810f1d8baee23012922b81e496661d61cf5`派生：

```text
/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight
branch: codex/idea2-dvac-residual-downweight
```

参数`signal_mode`在同一套代码内选择旧`global_zscore`或新`per_h_robust_residual`，不为两种公式维护
两套训练框架。新模式每step把两个actor rank的`[query,h]` log-V做一次小型`all_gather`，让两边使用
完全相同的recent-5 per-h history；只把本rank的weight带入原trajectory shuffle/replay。修改集中在：

```text
rlinf/algorithms/dvac_train_weighting.py
rlinf/workers/actor/fsdp_actor_worker.py
tests/unit_tests/test_dvac_train_weighting.py
examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_dvac_r_only_downweight_2step_smoke.yaml
```

writer schema v2保存raw `V_L2/L3/L4`、实际weight、50维position center/MAD/scale与clipped residual，
因此后续可以离线复算并比较position-only、R-only和v1 raw信号。

### 13.4 前测结果与当前停点

- 服务器`py_compile`与`git diff --check`通过；
- train weighting + telemetry合计`11 passed, 3 warnings`；warnings均来自既有依赖；
- 2-process probe验证两actor rank合并后得到相同per-h center/MAD/scale；
- 新config和default-off config均完整resolve；新旧2-step resolved除run路径和DVAC块外无差异；
- 真实2-step smoke于`2026-08-21T23:06:31+08:00`启动、23:35自然完成：2×A800、16 env、
  rollout epoch8、max steps2；其余任务/SFT/G8/B512/mb32/update2/flow-SDE/PPO/FSDP参数保持旧成功smoke；
- driver/observer均rc0。step1全`w=1`并生成两rank一致的512-query逐h history；step2实际weight的
  p05/median/p95约`0.66/1.02/1.20`，rank-local mean约`0.975/0.981`，两rank均同时出现小于和大于1的weight；
- step2 grad/ratio/KL均finite，双rank`rollout_step0001.npz`、`global_step_2`与抽样control trace均落盘，
  memory events全0。

v1服务器checkpoint与旧产物均未删除。正式配置作为一个单文件提交加入同一child；当前authority为
`/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight@3061872e30cfb496eb296354d30274d35b66576e`，
branch已推送、跟踪且worktree clean。v2 formal于`2026-08-22T00:13:16+08:00`启动：

```text
run     /root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
runtime /root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
PID     wrapper/driver/observer = 198255/198259/198260
```

截至00:18:35，它仍在第一个runner step的rollout epoch `1/16`，尚无完整Global Step。该状态只表明
已进入真实rollout调用链，不是step1完成或formal结果；精确进程、资源和claim后检见实施账L022–L023。
