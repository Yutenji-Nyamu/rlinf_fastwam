# current RLinf π0 × RoboTwin GRPO-DVAC formal-100 启动

## 1. 结论

2026-08-24 18:59 CST，formal 最初在物理 GPU `2,3,6,7` 启动；GPU `4,5` 当时保留给仍在运行的 RLT，二者没有混卡。

RLT自然完成并释放4/5后，用户要求把GRPO-DVAC迁到连续的物理GPU `4,5,6,7`。旧run在完整Step4后按owned PGID停止并原样保留；它尚无Step10 checkpoint，因此新run按完全相同算法、模型和预算fresh启动，不伪称resume。

2026-08-24 20:11 CST，新run在物理GPU `4,5,6,7`启动。首次健康确认：wrapper存活、exit pending、无fatal，已进入Step1第`3/8`个真实rollout epoch；四张卡均有目标进程。后续不持续盯守。

2026-08-24 21:11 CST再次只读刷新：新run完整到Step4/100并进入Step5 rollout `4/8`，无fatal；但发现旧run的wrapper/driver停止后，physical `2,3,6,7`上的旧Ray actors仍由shared raylet托管。它们与新run在GPU6/7重叠，因此“旧run已完全释放”这一先前判断不成立。当前尚有约1.4 TiB available RAM、没有OOM；本次状态查看未控制任何进程，后续应按旧run精确namespace清理残留，不能重启全局Ray或误杀新run。

## 2. resolved 合同

| 项目 | 正式值 |
|---|---:|
| source | `codex/sz-current-pi0-dvac-grpo@66c863bc5a45e90cb5161b30af54355b1104c810` |
| physical GPU | 当前 `4,5,6,7`，4 actor ranks；旧Step1–4 run为`2,3,6,7` |
| outer steps | 100 |
| train env × rollout epochs | `32 × 8` |
| trajectories / groups / max chunk records per step | `256 / 32 / 1024` |
| group size | 8 |
| actor global / micro batch | `512 / 32` |
| update epochs / max optimizer calls | `2 / 4` |
| π0 | exact SFT；`H=C=50`；Flow-SDE；LR `5.6e-6`；PPO clip `0.2` |
| DVAC | global-z；`L=3`；recent-5；warm-up 1 step；weight `[0,2]` |
| eval / checkpoint | fixed-64 every 10 / every 10 |
| video | disabled |
| normal / hard stop | Step 100 / 129,600 seconds |

`32×8/B512` 保留 current official/AutoDL 成功 DVAC 的每步 `256` trajectories 与 `B512` 全局预算，同时避免把深圳旧 `128` 个常驻 train env 的主存风险带入本次 formal。

## 3. 路径与精确入口

- 当前 packet：`/data/chenyiteng/results/rlinf-shenzhen/grpo/packets/dvac-global-z-formal100-4gpu32x8-g8-phys4567-v3`
- 当前 run：`/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-4gpu32x8-g8-phys4567-v3`
- 当前启动：`bash /data/chenyiteng/results/rlinf-shenzhen/grpo/packets/dvac-global-z-formal100-4gpu32x8-g8-phys4567-v3/launch.sh`
- 已停止的旧run：`/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-4gpu32x8-g8-phys2367-v2`；终点Step4，`runtime/stopped_by_user.txt`已记录迁卡原因。
- 本地生成/启动脚本：
  - `local_scripts/remote_commands/shenzhen_prepare_current_dvac_grpo_formal100_phys2367_20260824.sh`
  - `local_scripts/remote_commands/shenzhen_launch_current_dvac_grpo_formal100_phys2367_20260824.sh`

首个 packet 曾因非连续 GPU 列表中的逗号未加引号而在 compose 前退出；未创建 actor、未占卡。v2 只修正该启动参数表达，算法配置不变。

迁卡时首次从旧packet做文本派生，因`physical_gpus`在JSON中跨行而未命中，脚本在新run启动前主动退出；随后只把packet的GPU字段和路径做结构化改写并重新核对resolved/command/launcher。训练配置本身未变。

## 4. 与深圳此前成功GRPO v2的精确参数关系

两份resolved YAML逐叶比较：旧227个leaf、新237个leaf，共24处不同。除路径、命名、显式卡号和关闭视频外，真正有训练意义的差异只有两类：DVAC新增，以及三项预算缩放。

| 项目 | 深圳成功GRPO v2 | 当前GRPO-DVAC | 判断 |
|---|---:|---:|---|
| model / SFT / H=C / Flow-SDE | 相同 | 相同 | 完全继承 |
| GRPO、G8、reward/logprob、clip、LR | 相同 | 相同 | 完全继承 |
| micro batch / update epochs | `32 / 2` | `32 / 2` | 相同 |
| train env × rollout epochs | `128 × 4` | `32 × 8` | 每步trajectory由512减到256 |
| actor global batch | `2048` | `512` | 改为一半样本下更小batch |
| 最大chunk records / optimizer calls | `2048 / 2` | `1024 / 4` | 优化调用节奏不同 |
| eval / save / outer steps | fixed64 / 10 / 100 | fixed64 / 10 / 100 | 相同 |
| seed文件内容 | train/eval | train/eval | 两侧逐字相同 |
| DVAC | 无 | global-z、L3、recent5、warm-up1、`[0,2]` | 本方法唯一算法增量 |
| 视频 | 开 | 关 | 仅减少I/O，不改训练数学 |

因此它确实是“小代码增量 + 大部分核心参数不变”，但并不是深圳成功GRPO v2的严格单变量DVAC对照：每步样本、global batch和optimizer-call节奏同时变化。当前配置更接近旧AutoDL/current DVAC的`256 trajectories/B512`预算，并回避深圳`128`常驻env导致的约1.8 TiB主存问题。若要做严格方法对照，最干净的是以后补一个同样`32×8/B512`但`DVAC mode=off`的baseline，而不是直接把当前曲线与旧`128×4/B2048`归因为DVAC效果。

## 5. 预算事故、清理与严格匹配重启

上述判断暴露了实验设计事故：正式比较目标是深圳 GRPO v2，但启动时误用了本地 config-only commit
`554c6dc8...` 的 `32×8/B512` 默认值，而没有逐叶继承目标 run 的 `resolved.yaml`。规避主存风险不是
未经确认改变对照预算的充分理由。

错误配置 v3 在完整 Step 7 后按用户授权停止。先前迁卡遗留的 old job `1b000000/RLinf_1` 与错误配置
job `1c000000/RLinf` 均以精确 namespace 的 `ray.kill(no_restart=True)` 清理；shared Ray 未重启，
liwenbo 的进程未触碰。前者释放约159 GiB GPU memory与398 GiB GPU进程PSS；两次清理后目标 actors均为0。

2026-08-24 22:02 CST，fresh v4 启动到 physical GPU4--7：

- run：`/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v4`；
- `128 train env × 4 rollout epochs = 512 trajectories`，G8/64 groups，最多2,048 chunk records；
- GB2,048/MB32/update2，最多2 optimizer calls；fixed64/save10/100 outer steps；
- 与深圳 GRPO v2 resolved 逐叶比较只剩19项差异：10项DVAC字段，以及卡号等价写法、逐字相同seed的
  worktree路径、run隔离路径和命名；unexpected difference为0；训练预算及所有非DVAC数学字段相同；
- train/eval video也恢复为baseline的`true`，没有把I/O变化混入比较；
- 启动后已完成Step 1的首个真实rollout epoch（`1/4`，约5分55秒），GPU4--7约27--29 GiB/card、
  host available约1.8 TiB、wrapper存活、fatal=0；按用户要求不持续盯守。

source-locked official `7d07a421...` 没有精确 RoboTwin π0 GRPO YAML；它的官方RoboTwin GRPO示例是其他
VLA（典型`128×8/B1024`），official OpenPI GRPO示例是LIBERO（典型`64×8/B2048`）。因此二者都不能
替代本实验的实际深圳 GRPO v2 resolved 对照。

## 6. 2026-08-25 Step29图表与Step30最新现场

- 10:05 CST：完整Step30并进入Step31 rollout。wrapper存活、exit pending、fatal=0；
  `global_step_10/20/30`均存在，Step30 checkpoint为18 GiB并含4份DVAC sidecar。
- Step29 train success=`92.58%`，五步均值=`93.13%`；同预算GRPO v2同期五步均值=`92.70%`。
  Step1--29均值DVAC与baseline只差`-0.15 pp`，当前不能据单次run声称方法提升或退化。
- Step29 KL/clip/grad=`0.013/0.062/14.235`；DVAC weight mean/std/ESS=
  `1.0288/0.4436/0.8432`，upper/lower clip=`3.860%/0.292%`。机制持续工作且优化标量有限。
- fixed64：Step10=`57/64`，Step20=`59/64`；matched GRPO v2同期为`57/64`、`60/64`。
- 最新Step30 train=`90.23%`（512 trajectories），fixed64=`63/64=98.44%`；三张图的统一截点仍是完整Step29，
  避免把Step30后续现场状态混进已经下载的原始输入。
- GPU4--7当前约62--68 GiB/card、历史峰值74.90 GiB；GPU0--3空闲，其他用户没有GPU任务。
- 主存是唯一明显风险：available RAM从启动约1.984 TiB降到约0.4 TiB，最近3小时平滑变化约
  `-103 GiB`。PSI为0且没有持续swap I/O，当前没有抖动；但若下降继续，可能在100步前再次接近Ray
  95%阈值。该趋势与本次严格继承的`128 train + 64 eval`常驻环境合同一致，不归因于MiB级DVAC状态。
- 完成Step29时v4 available约450 GiB，旧matched GRPO v2同口径约464 GiB；两条内存轨迹高度接近。
  Step31 rollout期间瞬时值已到约293 GiB，因此更实际的预期是再次在Step50附近逼近阈值，而不是自然跑满100；
  这是风险判断，不是已经发生的退出。
- 三张图与小型原始输入：
  `evidence/dvac-grpo-v4-live-step29-20260825/`。本轮没有打ZIP或下载大产物。

## 7. 2026-08-25 Step31重绘、AutoDL复核与Step33现场

- 新快照统一以完整Step31为输入，并把matched GRPO baseline完整画到其真实终点Step52；Step32--52明确标为baseline-only，不冒充成配对区间。
- fixed64图使用真实图例：紫色方块为DVAC、蓝色三角为matched GRPO；Step10/20/30分别为`57/59/63`与`57/60/62`，累计均为`179/192`。Step30单点DVAC高`1/64`，但不能据此声称稳定提升。
- Step1--31 train mean差`-0.50 pp`，最近5步差`-1.25 pp`。Step31 weight std/ESS=`0.4382/0.8425`，证明DVAC不是全1/no-op；优化标量有限。
- old AutoDL与current深圳逐公式/统计域/recent5/ST挂点/rank reduce复核未发现语义错误；深圳与matched baseline逐叶仍为严格单变量DVAC对照。AutoDL g1--49训练曲线的累计/末5/末10优势为`+2.081/+2.734/+2.930 pp`，但旧run没有fixed held-out eval、同seed同代码mode-off对照或重复seed。
- AutoDL为`2×A800/256 trajectories/B512/约4 optimizer calls`，深圳为`4×H100/512/B2048/约2 calls`；跨机器同一个global step不是同一样本或更新预算。完整判断见[专题27](27_AUTODL_VS_SHENZHEN_DVAC_GRPO_EFFECT_AND_PARITY_20260825.md)。
- 11:10 CST现场完整Step33并进入Step34 rollout `1/4`；wrapper alive、exit pending、fatal=0。GPU4--7约59--61 GiB/card，GPU0--3空闲；host available约318 GiB，memory PSI很低但非零。继续运行的主要风险仍为常驻环境主存，不是DVAC tensor。
- 新图和小型原始输入位于`evidence/dvac-grpo-v4-live-step31-20260825/`；未生成ZIP、未下载checkpoint或视频。

## 8. 2026-08-25：被误杀后严格resume到100

v4在完整Step33后、Step34 rollout `2/4`期间被外部`ray.kill`终止；driver随后因actor死亡与Gloo连接断开返回255。最后可恢复点为完整`global_step_30`，因此Step31--33的已完成计算不会伪装成可恢复状态。

用户授权保持一切不变并固定physical GPU4--7。新run
`dvac-global-z-formal100-grpo-matched-4gpu128x4-b2048-phys4567-v5-resume30`从Step30严格恢复：

- target仍为绝对Step100，实际执行Step31--100，共70步；
- `128×4/512 trajectories/G8/2,048 records/GB2,048/MB32/update2`不变；
- global-z L3/recent5/warm-up1/`[0,2]`不变，4份rank sidecar严格恢复；
- fixed64/save10/video均不变；
- resolved逐叶只差`runner.resume_dir`与6个run-scoped输出路径，unexpected=0。

12:46 CST已明确载入Step30并进入Step31 rollout；wrapper alive、fatal=0，GPU4--7约21 GiB/card，host available约1.9 TiB。旧v4原样保留，新run独立落盘；没有清理或重启shared Ray，也没有触碰其他用户进程。

13:07 CST再次逐叶现场复核：v4/v5均为237个resolved leaf，仅`resume_dir`和6个run-scoped输出路径不同；模型、`128×4`采样、G8、GB2048/MB32/update2、global-z L3/recent5/`[0,2]`、fixed64/eval10/save10/video全部相同。续跑已到Step31 rollout `3/4`，wrapper alive、fatal=0。

## 9. 2026-08-25 16:54 CST：续跑完整Step40

- v5已完整到Step40并进入Step41 rollout；wrapper alive、exit pending、fatal=0，`global_step_40`完整目录已落盘。
- Step40 train=`507/512=99.02%`；fixed64=`63/64`，matched原GRPO同点也是`63/64`。四次fixed累计两侧均为`242/256`。
- 配对Step1--40训练成功率均值DVAC相对原GRPO为`-0.42 pp`，最近5步为`-0.98 pp`；当前仍没有稳定收益证据。
- Step40 KL/clip/grad=`0.01397/0.03385/6.406`；weight mean/std/ESS=`1.0049/0.4258/0.8478`，上下z裁剪=`3.04%/0.30%`。数值有限且DVAC不是no-op。
- v5最新host available约`971 GiB`，单卡显存最新最大约`65.1 GiB`、本段峰值约`67.0 GiB`；离Ray约100 GiB边界仍远，但主存继续下降，仍是后续主要风险。
- 三张手机可读PNG、自包含交互HTML、拼接CSV与小型原始输入位于
  [`evidence/dvac-grpo-v5-live-step40-20260825/`](evidence/dvac-grpo-v5-live-step40-20260825/README.md)。主曲线只拼接v4 Step1--30与v5 Step31--40，未混入被误杀后丢弃的v4 Step31--33；未下载checkpoint或视频，未打ZIP。

## 10. 2026-08-25：结束 `[0,2]`，fresh启动连续 `[0,5]`

- 用户授权结束当前 `[0,2]` 并启动严格matched的 `[0,5]`。17:28 CST精确停止发生在完整Step41后、Step42 rollout `2/4`期间；仅终止owned PGID、Ray job `20000000`及namespace `RLinf`的21个named actors，shared Ray和其他用户进程未动。
- `[0,2]` 最终完整Step41：train=`94.92%`；Step1--41相对matched GRPO的均值/最近5步差=`-0.39/-0.66 pp`；fixed10/20/30/40累计两侧均为`242/256`。轻量终态位于[`evidence/dvac-grpo-w0to2-final-step41-20260825/`](evidence/dvac-grpo-w0to2-final-step41-20260825/README.md)。
- current global-z原映射是`w=1+strength*clip(z,-2,2)`；把strength直接改大既会被非负权重guard拒绝，也不会得到`[0,5]`。因此复用旧R-only已有依据的分段线性映射：`z=-2/0/+2 -> w=0/1/5`。生产增量3 files，另补1个focused test file；7个目标测试全部通过。
- commit=`0e28ac6f09f821ea12e7d54eba7118ce0000ca86`，branch=`codex/sz-current-pi0-dvac-grpo-w0to5`，已push到personal remote。
- 新formal run：`/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-w0to5-formal100-grpo-matched-4gpu128x4-b2048-eval5-phys4567-v1`，fresh SFT、physical GPU4--7、100步。
- 相对 `[0,2]` 只改变`weight_min/max=0/5`、用户批准的`val_check_interval=5`及必要run路径/命名；save仍为10。相对原GRPO和`[0,2]`两份resolved逐叶检查，未批准的模型、seed、采样、batch、optimizer和训练数学差异均为0。
- 训练预算仍为`128 env × 4 rollout epochs=512 trajectories/step`、G8/64 groups、最多2,048 records、GB2,048/MB32/update2；global-z L3/recent5/warmup1/z-clip2、fixed64和video均不变。
- 18:01 CST wrapper存活、fatal=0、21个named actors齐全，已进入Step1 rollout `0/4`；GPU4--7约10.9--11.4 GiB/card、host available约1.9 TiB。按用户要求确认正常启动后不持续盯守。
- 18:27 CST只读刷新：完整Step1并进入Step2 rollout `0/4`；Step1 success/KL/clip/grad=`73.24%/.045/.130/22.799`。Step1是预期warm-up，DVAC weight mean/ESS=`1/1`；非均匀`[0,5]`从有history的Step2开始。GPU4--7约31--32 GiB/card、host available约1.6 TiB，fatal=0。
