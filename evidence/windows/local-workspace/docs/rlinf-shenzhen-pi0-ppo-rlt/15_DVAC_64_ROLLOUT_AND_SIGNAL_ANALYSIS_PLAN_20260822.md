# SZ π0 / Fast-WAM：64-rollout 扩展与 DVAC 分解分析执行草案

更新时间：2026-08-22  
状态：**SOURCE-RESOLVED DESIGN；本文件没有启动 GPU、下载、修改服务器源码或创建 run。**  
父计划：[`13_DVAC_SIGNAL_PI0_FASTWAM_OBSERVATION_PLAN.md`](13_DVAC_SIGNAL_PI0_FASTWAM_OBSERVATION_PLAN.md)。

## 0. 结论

建议把下一批数据分成两条互不混算的主集：

1. **π0：新跑一份 official fixed-64 `adjust_bottle`**。使用 4 个 rollout/env rank、每 rank 16 env，
   保持已经跑通的 official fixed-64 并发语义；现有 fixed-16 作为工程 P1/cross-check，不和新 fixed-64
   拼成一个有重复风险的“80条主集”。
2. **Fast-WAM：总计 4 tasks × 16 = 64**。复用已经完成的 `adjust_bottle` 16条，再各跑
   `move_stapler_pad`、`turn_switch`、`pick_diverse_bottles` 16条。三项新增任务都在锁定 vendor 的
   official task list 内、step limit 都是400；Fast-WAM论文 clean 成功率分别是77%、61%、80%，比
   `adjust_bottle` 的100%更有机会自然出现成功和失败，又避免首批就选 step limit 900/1500 的长任务。

调度上，Fast-WAM 不用 Ray，可在 live RAM 允许时与 GPU4--7 的 GRPO 并行，单独使用 GPU3。
π0 evaluator 会启动同一 Linux 用户下的 Ray；当前 GRPO 已有一套 Ray，因此**即使 GPU0--3 空闲，π0
fixed-64 也不能直接并发启动**。首版不人为维护第二个隔离 Ray runtime：等 GRPO 自然结束/经授权停止且
Ray 完全退出后，再用 physical GPU0--3 跑 π0 fixed-64。

首轮分析不在线改变 chunk，不用 DVAC 选样本或 phase 标签。CPU 离线统一计算：

```text
raw:       V_L(q,h), y(q,h)=ln(V_L+1e-12)
two-way:   Position b_h, raw residual r=y-b, standardized residual R=r/s_h
four-way:  y = mu + P_h + S_raw(q) + I_raw(q,h)
            R = S_std(q) + I_std(q,h)
```

这样既保留用户想看的“位置 + 剩余”两通道，也避免把 raw-log 单位和 MAD 标准化单位错误相加。

## 1. 现有数据合同与规模

### 1.1 π0 fixed-16（已完成）

权威运行：

```text
/data/chenyiteng/results/dvac-observation/
  pi0-adjust_bottle-p1-16ep-800baf80-v1
```

已核终态：16 episodes、12 success / 4 failure、64 queries、1份rank聚合NPZ、192张三相机PNG、
1个query-boundary tiled MP4；source=`800baf80...`，H/C/M/D=`50/50/4/14`。

NPZ schema：

| 数组 | 每query shape | float32裸字节/query | 含义 |
|---|---:|---:|---|
| `x_chain` | `[5,50,14]` | 14,000 | 4次Euler更新的5个action latent状态 |
| `z_endpoint` | `[4,50,14]` | 11,200 | 已按 `z=x-t*v` 保存的4个clean endpoint预览 |
| `final_model_action` | `[50,14]` | 2,800 | 最终normalized active-14D action chunk |
| `env_action` | `[50,14]` | 2,800 | 交给环境的action chunk |
| `robot_state` | `[14]` | 56 | query时14D state |
| `timesteps` | `[4]`/shard | 16/shard | `[1,.75,.5,.25]` |

不计NPZ header/压缩时，共约30,856 bytes/query；现有64-query tensor裸量约1.88 MiB。磁盘主体是PNG，
不是denoising tensor。

join文件：

- `query_index_rollout_rank*.csv`：`episode_idx/query_idx/action_slot_start/reset_id/success_before`、
  三路query图、tiled-video worker/index/pre/post frame等；
- `episode_index_env_rank*.csv`：真实reset ID、success、return、action slots和终止原因；
- `manifest_rollout_rank*.json`、`run_manifest.json`：source/config/shard合同。

### 1.2 Fast-WAM sequential-16（已完成）

权威payload：

```text
/data/chenyiteng/results/dvac-observation/
  fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2
```

official result：

```text
/data/chenyiteng/projects/fastwam-standalone/worktrees/
  fastwam-dvac-observe-7faa711/evaluate_results/robotwin/
  robotwin_uncond_3cam_384/fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2
```

已核终态：16 episodes、16 success、80 queries、80个per-query NPZ、240张三相机PNG、16个完整MP4；
source=`c63dc9b5...`，H/C/M/D=`32/24/10/14`，且 resolved
`skip_get_obs_within_replan=false`，所以视频是逐executed-action fresh observation，而非旧的低FPS重复帧。

每个NPZ schema：

| 数组 | shape | float32裸字节/query | 含义 |
|---|---:|---:|---|
| `x_chain` | `[11,32,14]` | 19,712 | 10次Euler更新的11个action latent状态 |
| `v_chain` | `[10,32,14]` | 17,920 | action flow velocity |
| `x_next` | `[10,32,14]` | 17,920 | 必须逐元素等于 `x_chain[1:]` |
| `timesteps` | `[10]` | 40 | scheduler action timestep |
| `deltas` | `[10]` | 40 | 实际Euler delta |
| `final_model_action` | `[32,14]`等价元素数 | 1,792 | normalized final action |
| `env_action` | `[32,14]` | 1,792 | denormalized action chunk |
| `robot_state` | `[14]` | 56 | query时14D state |

不计NPZ header/压缩时约59,272 bytes/query；现有80-query tensor裸量约4.52 MiB。`z_endpoint`当前不重复
落盘，分析时由 `x_chain[:-1] - (timesteps/1000)*v_chain` 重建。

`queries.csv`已经给出query action-slot `[start,end)`、planned/executed length、success before/after、
三路query图和视频帧 `[start,end)`；`episodes.csv`给出outcome/总action slots/query数/first-success。
当前writer的`source_seed`字段为空是已知接口边界；accepted seed应从同一official eval log的每episode
`current seed:`行按完成顺序离线join，不为此改policy/runtime。

### 1.3 现有分析器能复用什么

[`local_scripts/analyze_dvac_first_collection.py`](../../local_scripts/analyze_dvac_first_collection.py) 已实现：

- π0 rank shard与episode/query join；
- `L=2/3/4` population variance、`V_total`、action-dimension贡献；
- query timeline、horizon heatmap、endpoint convergence、success boundary等10张图；
- source hash与schema检查。

它**不能原样用于本批**：它硬编码π0 `M/H/D=4/50/14`、16 episodes、每episode 4 queries、旧CSV名字；
它不读Fast-WAM per-query NPZ，也没有 Position/Residual/S/I。实施时保留旧脚本作为结果oracle，新建一个
统一CPU分析入口，复用其已验证公式/画图思想，不在旧证据脚本上堆大量task-specific条件。

## 2. Fast-WAM 多任务能力的source依据

这一结论不是由checkpoint文件名猜出来的：

1. 锁定 [Fast-WAM README@7faa711](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/README.md)
   只发布一份 `robotwin_uncond_3cam_384.pt` + 配套stats，并用
   `run_robotwin_manager.py`评估RoboTwin benchmark；因此它是released RoboTwin benchmark model，
   不是`adjust_bottle`单任务权重。
2. 锁定 [run_robotwin_manager.py@7faa711](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/experiments/robotwin/run_robotwin_manager.py)
   在 `EVALUATION.task_name`为空时，从vendor `_eval_step_limit.yml`读全任务；否则可只跑指定任务。
3. 锁定 [eval_robotwin_single.py@7faa711](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/experiments/robotwin/eval_robotwin_single.py)
   把任意指定`EVALUATION.task_name`传给vendored RoboTwin，README/模块docstring还直接以
   `click_alarmclock`示例。
4. 锁定 vendor [RoboTwin `_eval_step_limit.yml`@bf44be51](https://github.com/RoboTwin-Platform/RoboTwin/blob/bf44be51cf5717a5595ce59447f2cf5263d2aa95/task_config/_eval_step_limit.yml)
   包含50个official任务；深圳已准备的是该vendor lock对应的完整assets copy与exact task_config tree，
   不是只含`adjust_bottle`的资产子集。

边界：这证明released checkpoint/evaluator按官方设计支持多任务；不保证本次16-seed块复现论文百分比。
每个任务仍须以真实exit、video、episode CSV和outcome报告本机结果。

## 3. 推荐64-rollout数据集

### 3.1 π0：新 fixed-64 主集

| 项 | resolved草案 |
|---|---|
| task/policy | `adjust_bottle` / pinned π0 SFT `92684e50...` |
| source | telemetry head `800baf80...`；RoboTwin `0008ae68...` |
| resources | **GRPO/Ray完全退出后** physical GPU0--3，placement `{env, rollout:0-3}` |
| episodes | 64 fixed reset episodes，4 env ranks ×16 |
| seed合同 | base seed `0`；`eval_seeds.json` SHA `194164f7...`；沿official `partition_success_seeds` |
| horizon | H=C=50，M=4，D=14，max episode=200 |
| upper budget | 12,800 environment action slots；256 policy queries；1,024 denoise steps |
| expected wall/resource | 历史同机non-telemetry fixed64为5m58s、约11GiB/卡、host used约113GiB；本次加PNG/NPZ，预算10--15min |
| disk reservation | 2GiB足够；全部写`/data`，实际完成后记exact bytes |
| output | `/data/chenyiteng/results/dvac-observation/pi0-adjust_bottle-fixed64-800baf80-v1` |

现有fixed-16是单rank seed partition，新fixed-64是4-rank official partition；两者可能有reset ID重合。
所以主报告只用新fixed-64，旧P1只做schema/分布复核。启动packet应从已有official fixed64
`budget-and-seeds.json`复制并固化64个精确reset IDs，而不是在聊天中手抄或运行后选seed。

### 3.2 Fast-WAM：4 tasks ×16总集

| task | 数据量 | source step limit | 选择理由 | paper clean/rand参考 |
|---|---:|---:|---|---:|
| `adjust_bottle` | 已有16 | 400 | 共同任务、bimanual/contact；现有数据全成功 | 100/100 |
| `move_stapler_pad` | 新16 | 400 | 单臂grasp→transport→aligned place；预期自然混合outcome | 77/64 |
| `turn_switch` | 新16 | 400 | articulated/contact manipulation；低成功率候选 | 61/59 |
| `pick_diverse_bottles` | 新16 | 400 | 双臂并行grasp/lift/place与20种bottle实例 | 80/85 |

表中成功率只用于预注册任务组合，来自
[Fast-WAM paper Appendix A.1 Table 3](https://arxiv.org/html/2603.16666#A1.T3)；不是本机预期值或验收阈值。

任务源码分别锁在vendor：

- [`move_stapler_pad.py`](https://github.com/RoboTwin-Platform/RoboTwin/blob/bf44be51cf5717a5595ce59447f2cf5263d2aa95/envs/move_stapler_pad.py)
- [`turn_switch.py`](https://github.com/RoboTwin-Platform/RoboTwin/blob/bf44be51cf5717a5595ce59447f2cf5263d2aa95/envs/turn_switch.py)
- [`pick_diverse_bottles.py`](https://github.com/RoboTwin-Platform/RoboTwin/blob/bf44be51cf5717a5595ce59447f2cf5263d2aa95/envs/pick_diverse_bottles.py)

三项新增任务都保持official single-evaluator语义：

```text
checkpoint/stats = same locked release
task_config       = demo_clean
instruction_type  = unseen
seed              = 42
candidate seed    = 4,300,000 起
accepted block    = 每任务按official expert check取得的前16个feasible seeds
H/C/M/D           = 32/24/10/14
sigma_shift       = 5.0
skip_get_obs_within_replan = false
GPU               = physical 3，一次只跑一个task
```

accepted seed列表不是outcome筛选：它是官方evaluator在policy rollout前的expert-feasibility过滤。每个任务
完成后从log提取16个`current seed`并写入derived episode表；不因看到policy成功/失败而补挑seed。

每个新增task的最坏预算均为：6,400 action slots、272 queries、2,720 denoise steps；0训练、0 optimizer、
0 checkpoint。现有adjust_bottle 16条约20分钟；考虑失败episode走满400步，新增任务各预计25--60分钟，
三项总计约1.5--3小时。每任务保守预留1GiB、3小时owned timeout；三个独立run id，例如：

```text
fastwam-move_stapler_pad-p2-16ep-c63dc9b5-v1
fastwam-turn_switch-p2-16ep-c63dc9b5-v1
fastwam-pick_diverse_bottles-p2-16ep-c63dc9b5-v1
```

不把`open_microwave`或`hanging_mug`塞进首64：它们很有分析价值且paper成功率更低，但step limit分别为
1500/900，会把首批时间上界放大。首64看完后，可把它们作为long-horizon P3，而不是悄悄替换已注册task。

## 4. 启动与停止合同

### 4.1 共同启动条件

- 精确source/checkpoint/stats/task_config/seed文件锁仍一致；worktree只有Fast-WAM三个已知symlink；
- 新run的metadata、payload、official-result目录都不存在；不复用失败v1目录；
- 目标GPU没有他人compute process，`/data`有上述预留空间；
- live检查GRPO/Ray/host RAM：Fast-WAM可与GRPO共存，但不停止或修改GRPO；π0必须等当前Ray退出；
- 每个run先以同一overrides保存resolved config，再启动official入口。

### 4.2 运行中停止条件

只约束本run owned process：official child非零、traceback、CUDA OOM/illegal instruction、writer/schema
确定性错误或hard timeout。成功率低、DVAC大/小、某一phase曲线不好看都不是停止条件，也不按outcome重跑。

### 4.3 完成条件

- driver自然exit0；episode rows精确等于目标16/64；query rows和NPZ一一join；所有数组finite、shape/schedule正确；
- episode outcome、accepted/reset seed、query action range、query images、video path全部可join；
- Fast-WAM每task有16个完整MP4；π0有rank/tile metadata与combined MP4，不能要求它伪造per-action视频；
- 记录actual wall、GPU peak、host RAM、exact bytes和成功/失败数；不把16/64外推成总体成功率。

## 5. 离线信号计算：两通道与四项同时给出

对每个 `(policy, checkpoint, task, L)` 独立计算；不同模型/任务不共享位置基线。位置基线只用
`baseline_eligible = (success_before == false)` 的query：Fast-WAM成功后立即停，本来没有post-success
query；π0 fixed-C50会保留已成功后的后续query，这些行继续派生R/S/I并单独展示，但不反过来污染
pre-success位置基线。

### 5.1 raw endpoint variance

```text
z_i(q,h,d) = x_i(q,h,d) - tau_i * v_i(q,h,d)

V_L(q,h) = sum_d Var_population(z_tail(L)(q,h,d))
y(q,h)   = ln(V_L(q,h) + 1e-12)
```

π0直接读`z_endpoint`并报L=2/3/4；Fast-WAM先由`timestep/1000`重建z，再报L=3/5。
跨模型主面板固定`L_common=3`。已有旧脚本的`log10`可继续用于旧图复核；新表用自然对数并在manifest写明，
不要把两种raw y数值直接拼表。

### 5.2 Position + Residual

```text
b_h    = median_{q in baseline_eligible} y(q,h)
mad_h  = median_{q in baseline_eligible} |y(q,h)-b_h|
s_h    = max(1.4826*mad_h, 1e-6)
r(q,h) = y(q,h)-b_h
R(q,h) = r(q,h)/s_h
```

- `b_h`：该task/model的第h格通常水平，即Position-only；
- `r`：仍保留raw-log单位的位置剩余；
- `R`：除以该位置自身典型波动后的可比较异常度。`R=+2`表示“同一h位置内高约2个robust scales”，
  不是成功概率或安全阈值。

### 5.3 四项：同时保存raw-unit与standardized版本

为了满足严格加法记账：

```text
mu             = mean_h b_h
P_h            = b_h - mu
S_raw(q)       = mean_h r(q,h)
I_raw(q,h)     = r(q,h) - S_raw(q)

y(q,h) = mu + P_h + S_raw(q) + I_raw(q,h)
```

为了比较不同h自身波动：

```text
S_std(q)       = mean_h R(q,h)
I_std(q,h)     = R(q,h) - S_std(q)

R(q,h) = S_std(q) + I_std(q,h)
```

不要写成`y=b+S_std+I_std`，因为两边单位不同。最终`query_horizon.csv`至少包含：

```text
policy, checkpoint, task, episode, query, h, L,
V, y_ln, b_position, mad_scale, r_raw, R_std,
mu, P_h, S_raw, I_raw, S_std, I_std,
baseline_eligible, success_before, executed_flag, action_slot,
phase_coarse, phase_task, outcome
```

## 6. 视频与任务阶段对齐

### 6.1 Fast-WAM：可做真实executed-action时间轴

当前及新增run都锁`skip_get_obs_within_replan=false`。writer的query区间为：

```text
video pre-action frames = [video_frame_start, video_frame_end_exclusive)
action slot a           -> frame a
h within query          = a - query_start_action_slot
terminal success frame  = terminal_success_video_frame（若存在）
```

只有`h < executed_length <=24`能对齐真实动作帧；`h=24..31`是model-only未执行tail，只能放在I heatmap，
不能画成task phase。成功早停query同样只对齐实际executed部分。

代表视频不按DVAC挑“最好看”的。每task预注册：

1. outcome内action-slot长度最接近中位数的一个success；
2. 同规则的一个failure（若存在）；
3. tie按accepted seed/episode index最小者。

在打开DVAC图之前，先根据视频独立标注：

- `phase_coarse = MOVING / OPERATING / TRANSITION / UNKNOWN`；
- `phase_task = approach / grasp-contact / transport-or-rotate / place-release / verify`。

输出一张story panel：抽取query-start与关键action帧，上方是视频帧，下方同一x轴画`S_std(q)`阶梯线和
executed-action的`R/I_std(q,h)`；背景色只来自独立phase CSV，success boundary单独画线。

### 6.2 π0：只作query/chunk级对齐

π0 tiled MP4只有query-boundary帧，metadata虽能定位tile与pre/post query frame，却不能把H=50解释成50个
视频动作帧。主图使用三路query PNG：

- `S_std(q)`对`action_slot_start=0/50/100/150`；
- 每个query旁放同一时刻三路图与`I_std(q,h)` heatmap；
- success-before/episode outcome来自CSV，不从文件名猜；
- 不在本64批次引入optional control trace或近似h↔physics frame映射。

## 7. 最小图表与统计口径

每个policy/task先单独报告，再做standardized cross-model panel：

1. `b_h ± s_h` Position profile；
2. 同一query的`y`、`r`、`R`三联heatmap；
3. `S_raw/S_std` episode timeline + 独立phase/success边界；
4. `I_std(q,h)` heatmap + 预注册代表frame strip；
5. success/failure、MOVING/OPERATING的episode-level聚合与bootstrap CI；
6. π0 L2/3/4、Fast-WAM L3/5 sensitivity；
7. 跨模型只比较R/S_std/I_std的分布、rank与phase/outcome association，raw V/y不横比。

统计单位以episode为主：先在episode内聚合，再bootstrap episode；不能把64×H个cell当独立样本。
某task某outcome少于5条时只作描述，不补挑seed，也不做显著性结论。

## 8. 实施批次

实现只需一个CPU入口和两种reader，候选CLI合同为：

```bash
python local_scripts/analyze_sz_dvac_rollouts.py \
  --run 'policy=pi0,task=adjust_bottle,payload=/data/.../pi0-fixed64' \
  --run 'policy=fastwam,task=adjust_bottle,payload=/data/.../fastwam-adjust,official=/data/.../adjust-result' \
  --run 'policy=fastwam,task=move_stapler_pad,payload=/data/.../fastwam-stapler,official=/data/.../stapler-result' \
  --phase-labels /data/.../phase_labels_v1.csv \
  --output /data/.../analysis/sz-dvac-64-v1
```

第一次可省略`--phase-labels`先生成待标注的代表episode/frame清单；phase CSV补齐后，在新的analysis输出
目录生成最终图，不覆盖首次derived表或source payload。无需给每个task另写分析脚本。

1. **现在可做的CPU工作**：为现有π0-16和Fast-WAM-16实现统一reader/derived表，先生成Position/R/S/I；
   这不需要重跑模型。
2. **GRPO运行期间**：若live GPU3/RAM/disk允许，按顺序运行三项Fast-WAM task×16；每task自然完成后再启下一个。
3. **GRPO Ray退出后**：GPU0--3运行π0 fixed64 telemetry。
4. 全部source run只读冻结；derived analysis写新目录，例如
   `/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-64-v1`，不回写payload。
5. 先交付CSV/summary和8张左右高信息图，再决定long-horizon task、更多seed或π0 control trace；不预设下一批。

## 9. 仍需live补齐但不改变设计的字段

本文件编写时按主协调要求没有新增SSH连接；以下只在GitHub设备登录流程结束后串行读一次现场：

- 两个现有P1目录及扩展名分组的exact bytes；
- Fast-WAM三个新增task所需asset模型目录在当前vendor copy中的存在性；
- GRPO当时Ray/PID、GPU3与host RAM，决定Fast-WAM何时启动；
- 新run启动前目标路径absence与实际accepted seed列表的post-run join。

这些是执行前动态字段，不改变上面的official多任务能力、task选择、数学分解或视频对齐合同。

## 10. 2026-08-22 23:35 CST：首版真实离线分析已完成

- 真实 source 已按只读合同合并分析：π0 `adjust_bottle` fixed64（64 episodes，42 success / 22 failure）
  与 Fast-WAM `adjust_bottle` P1（16 episodes，16 success / 0 failure）。两者按 physical run/cohort 保持
  两个独立 Position/Residual 基线，没有把早期 π0 P1-16 混入 fixed64。
- 唯一新 output 为
  `/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-pi0-fixed64-fastwam-adjust-p1-v1`；默认 plots 与
  storyboards 一次 exit 0，35 files / `48,494,941` bytes。随后创建的完整tar.gz为`9,111,207` bytes，
  已下载并解压到本地`evidence/dvac-analysis-p1-fixed64-20260822`；远端source/output/archive均保留。
- 主统计只使用 294 条 pre-success query；42 条 π0 post-success query 只在显式
  `*_all_queries.csv` 伴随表保留。Fast-WAM 1,852 条真实 executed-action/frame 行均为 `h<24`，16 条
  terminal-success action/frame 边界已保留。
- 两通道与四项恒等式最大误差不超过 `1.78e-15`；19 张 PNG 全部可解码，其中含 15 张统计图、
  3 张代表 query strip 与 1 张真实 Fast-WAM storyboard。
- 本轮没有独立 phase annotation 与 official seed-map，因此 phase 汇总为固定 schema 的空表，属于计划中
  “先生成待标注首版”的预期状态。下一步应先看首版图和代表视频，独立填写 phase CSV，再写全新 analysis
  目录，不覆盖本输出。
- 精确命令、SHA、schema、counts、identity 与大小见
  [`evidence/15_DVAC_OFFLINE_ANALYSIS_LEDGER_20260822.md`](evidence/15_DVAC_OFFLINE_ANALYSIS_LEDGER_20260822.md)。
- 首批实际Position、outcome、S/I与视频帧结果见
  [`16_DVAC_FIRST_REAL_RESULT_20260822.md`](16_DVAC_FIRST_REAL_RESULT_20260822.md)。π0中failure的episode均值
  `S_std`比success高`0.436`（success-minus-failure 95% bootstrap CI=`[-0.630,-0.242]`），而
  `abs(I_std)`差异接近0且区间跨0；这是观察性关联，不是训练收益或因果结论。
