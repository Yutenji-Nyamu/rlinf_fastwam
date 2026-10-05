# RynnValue WMRL 实施与实验记录

2026-10-05。用户已授权实现、实验、少量并行smoke和通过后正式训练；Rynn/WMRL最高优先，RLT只补空闲。本页记录实际执行，前两份计划文档保留调研时状态。

## 实验合同

| 项 | 本轮设置 |
|---|---|
| 机器/卡 | 深圳3，物理4–7；4/5策略＋分时Rynn，6/7 OpenDW B16 |
| 任务/方法 | adjust_bottle，π0.5＋OpenDW＋Rynn Success＋GRPO |
| 新奖励 | 每C32末判定；首次Yes为1并done，No为0；旧RM仅诊断 |
| unknown | 缺失/截断128→256 token补一次；仍unknown则整G8无效，actor算优势前屏蔽 |
| 输入/模型 | 主图256×320，K8均匀历史含首末；HF8738c5e4，BF16，原隔离attention和官方Analysis提示 |
| 正式预算 | N64/G8/R8，512条/轮，C32/max384；global2048/micro8/U2；CP70续至总200轮，save/eval每10轮 |
| 原生评估 | 原固定32种子、三图/C32/max384、物理仿真check_success |
| 选并行 | RM少量B4/B8/B16，先2条B1参考；按实际吞吐及共卡显存余量选档，WM B16保持 |
| 接线smoke | N64/G8/R1/max384，一轮；global768/micro8/U2，无额外原生评估；smoke从SFT，正式独立续CP70 |

global768对应64×12个动作块，让smoke每U只有一个全局更新批；正式global2048不变。两份checkpoint路径分开，smoke权重不接正式。CP70已有原生17/32记录；本实验是改变裁判后的接续，不称从SFT开始的公平方法对比。

资源切换复用现有精确owner/adoption：保存的旧CP70保留；旧WM正常清理，RLT保持已暂停，接新owner管理四服务、smoke和正式；真正失败才由原借还链释放并恢复RLT。共享Ray、其他用户、深圳1/2保持。

## 路径与命令

```text
S=/data/chenyiteng/projects/opendw-robotwin-smoke-20261003
旧运行：S/runs/formal-b16-v1
新源码：S/rlinf-rynn-v1                 branch codex/rynnvalue-wmrl-20261005
当前控制：S/rynn-control-v2
当前输出：S/runs/rynn-success-v2
v1失败记录：S/rynn-control-v1、S/runs/rynn-success-v1

python S/rynn-control-v1/code/rynn_handoff.py \
  --ready-manifest S/rynn-control-v1/prepared/ready.json \
  --resume-receipt S/rynn-control-v1/prepared/resume-receipt.json \
  --plan-template S/rynn-control-v1/prepared/plan-template.json
```

上列handoff是v1历史命令，不能重放。当前v2正常重借入口为`python S/rynn-control-v2/code/rynn_formal_owner_v2.py --plan S/rynn-control-v2/prepared/plan.json owner`，由一次性`launch_v2_remote.py`启动。以上S为阅读缩写，实际argv使用完整绝对路径及原RLinf Python；以唯一启动回执判断是否启动。

停止条件：身份/卡位错误、CUDA/OOM或服务异常、卸载失败、输出身份错位、基本控制样本明显误判、smoke无有效学习信号时不进入正式。新RM或WM无法解释的输出先保留证据，不改阈值制造通过。

奖励链复核：C32奖励求和保留终局1；done先清未来累计、再加本步奖励，不吞掉首次成功。原组过滤`[0.1,0.9]`在0/1回报下保留1～7/8成功的混合组，新增effective指标在过滤之后计算；数量为每actor rank局部值，全局须求和。这是过滤组，尚无动态补采样。

已知边界：现有actor对整轮零有效组仍会调用optimizer step；恢复的Adam动量/weight decay可能移动参数，不能把这种更新当学习。smoke已明确要求有效组与梯度均非零；正式若出现持续零有效组，需单独处理跳过/停止，当前未把旧actor改成新优化器流程。

## 准备证据

- 16:29实时核SZ3原owner394537/start707417273、driver3159103/start707579155，0–3只4MiB基线；旧实验完成CP70并进入下一轮。
- 独立Git worktree已建立，复制原运行4处差异；旧repo和既有dirty保持。新源码仅叠加Rynn接口、history、显式unknown组mask与资源阶段控制。
- CP70两份rank checkpoint在CPU加载到`model/optimizers/lr_schedulers/fsdp_version/rng`；full_weights归档完整、随后已有采样日志，未另做外部checkpoint调用。
- 32条基本控制样本：前16份clean50演示，各取初始静止帧和完整专家片段。用于粗语义核验，**不是原生策略或生成视频成功率测试集**；生成视频判定与学习信号由完整接线smoke检查。
- 最终服务器CPU27项通过：6环境/组mask，14服务parser/batch/卸载，7owner合同。已修复Rynn控制请求必须为空body的差异，并加入真实有效组与非零梯度的正式启动门槛。

## 实际交接

16:47:55精确交接启动；16:49:54回执确认旧WM完成清理、资源转交成功，RLT保持暂停。新owner为`3480426/start716121382`，交接入口的唯一回执位于`rynn-control-v1/launch-receipt.json`，不可重放。16:52正在CPU加载WM，GPU smoke尚未开始；此时0–3无新占用。

17:01首条B1调用发现dtype兼容错误：BF16隐藏状态进入仍为FP32的附加value head。显存峰值allocated约19.87GiB，未OOM，也尚未产出任何Success结果。固定官方`inference.py:189`及旧数值服务均额外做整模型`.to(device,dtype=BF16)`；新服务遗漏此步骤，已在本地补CPU端整模转换及混合dtype回归测试，准备部署独立v2。

17:03:47 v1正式终态为failed：四服务精确清理完成，`recovery_error=null`，原借还链已派发RLT恢复；首轮恢复尚未验收，不能称RLT训练已完成。v1代码、日志和失败回执保留，未启动策略smoke/正式训练。后续从实际返回的RLT身份重新借卡，不重放旧handoff。

17:12修复版服务器CPU29项通过（6环境、15服务、8 v2 owner），包含BF16/FP32错误重现与官方转换后的修复。17:12:59正常重借计划核验通过，RLT恢复断点为225/225/150/150；17:13:25启动v2 owner `1753971/start716262918`，计划SHA `416890b7d50080b9b9be8da6fc0a8d033be1466242a368393273a290d5769d82`。v1目录保留，运行repo及CP70不变。GPU结果仍需新回执。

## v2 GPU结果与当前诊断

17:26–17:30，dtype问题已修复，真实B1/B4成功执行，无OOM、无unknown。B1两条冷调用50.24秒；B4共32条27.64秒，约1.158片段/秒，allocated峰值19.76GiB、reserved峰值21.37GiB。请求按精确prompt长度分桶，实际批次为4/2混合，不能称全部满B4。

质量结果：16份初始画面全部No；16份完整clean50专家轨迹也全部No。输出正常EOS、25–37 token，未截断，不能归因于解析失败。原始Match初始13/16为Yes、专家12/16为Yes，故也不只是任务不匹配。此时停止质量门槛检查，尚未测B8/B16，也未进入N64策略smoke或正式训练。

17:30:27 v2终态failed，四服务全部清理，`recovery_error=null`，RLT已派发恢复；这仍不等于恢复首轮已验收。v1/v2代码、回执和CP70均保留。

已看专家画面：中后段瓶子竖直举起，最终部分出画。正准备仅借GPU4做有限诊断：可见前缀、重复可见末态、K64、官方视频及官方metadata；同时单独测B4/B8/B16吞吐，不把语义No当基础设施失败，也不放宽正式质量门槛。GPU5–7维持RLT。诊断路径`S/rynn-diagnosis-v1`，与v2冻结代码分开。分析见[语义诊断](rynnvalue_semantic_diagnosis_20261005.md)。

17:48:34单卡诊断启动，owner1178576/start716473794；5项owner CPU检查及17案实际processor预检通过（无CUDA）。GPU结果：17次有效生成全部No；包括K64完整专家、可见前缀/重复可见画面、简化任务文字、官方视频K8/K64及官方metadata。3案空metadata按固定processor要求明确拒绝，再独立记录generic对照，没有把输入错误记成No。此结果排除“仅末帧出画/仅K8太少”作为充分解释，但尚不能排除预先缩小图像的影响。

| K8实际batch | 16条总耗时 | 片段/秒 | allocated峰值GiB | reserved峰值GiB |
|---|---:|---:|---:|---:|
| B4 | 7.104秒 | 2.252 | 19.75 | 20.43 |
| B8 | 4.056秒 | 3.945 | 21.62 | 22.58 |
| B16 | 2.989秒 | 5.354 | 25.36 | 26.88 |

上表均预热后、同prompt长度的两个固定样本重复，仅测吞吐，不称准确率。B16比B4快约2.38倍，无OOM/unknown，后续K8在线候选选B16；还未与π0.5同时驻留验证。512条仅终局评分按单卡理想速度约96秒、双卡约48秒；每C32判一次时最多6144条评分，双卡约9.6分钟，另有预处理/长度分桶/通信等开销，不能把48秒当整轮RM时间。

17:53:08单卡诊断工程completed、child0、清理完成、GPU4已派发RLT恢复，5–7原driver身份仍live。工程completed不代表奖励质量通过。GPU4当前归还父链为`rynn-diagnosis-v1/prepared/rynn-diagnosis-v1-gpu4`，5–7仍来自原v2归还链，后续不能把四卡当同一旧owner直接重放。

18:09:52最后四案原尺寸对照已启动，owner3351488/start716601632，路径`S/rynn-diagnosis-v2`，plan SHA `a714f803424062df8e0272ac2c31437605fffb134d5bd8bcde59915faaf9331f`。仅GPU4，6项owner检查和4案实际processor预检通过。Rynn环境缺cv2的首个CPU预检失败记录保留；用既有RLinf环境预导出原始像素解决，未安装或改环境。实际专家原图320×240，处理后仍每帧80视觉token；官方视频1280×720按原入口等比到640×360，每帧220视觉token。复用Git clean的固定官方数值forward后generate，只有4个最终K8输入，不跑冗余前缀曲线或绘图。

18:12最后四案完成推理，仍全部Success No：官方原视频按640×360输入（2027 tokens）、旧320×256输入（907 tokens），两段专家原320×240输入（930/926 tokens）。所有输出正常EOS，官方数值forward后generate与直接generate的结论一致。专家末时刻数值头原值约0.3404/0.4986；未核定它们的成功阈值或奖励尺度，不能据此把No改成Yes，也不能把数值输出当已通过的替代奖励。

**实施结论：K8 Success batch工程可用，B16为吞吐候选；Success奖励质量未通过，完整策略smoke与正式均未启动。** 16个专家对照及额外前缀/K64/原尺寸/官方流程未提供可用Yes；没有证据支持只调并行、token数或输入尺寸即可开训。当前冻结CP70；后续建议先离线验证数值头对进展/失败的区分，或校准Success。采用连续数值奖励将改变奖励及GRPO组过滤定义，需单独记录实验方案，不静默替换。

18:13:27原生四案工程completed，child退出0，同步offload及进程清理完成，recovery_error=null，GPU4已派发RLT恢复；5–7原driver身份仍live。18:14只读GPU快照0–3无计算/图形进程。GPU4后续权威借还路径为`rynn-diagnosis-v2/prepared/rynn-diagnosis-v2-gpu4`，5–7保持WMRL v2父链；此次未等候RLT恢复首轮。

## 发布范围

通过独立发布分支`codex/rynnvalue-wmrl-release-20261005`保存接口、修复、诊断脚本、专题和三份轻量回执；不改运行checkout的HEAD/index，不上传权重、原始视频或完整环境。实际commit与远端同SHA核验记在发布事务`S/publication/rynn-success-release-20261005-v1/published.json`。本机对应目录为`docs/world-model/publication_rynn_20261005`；[源码清单](../publication_rynn_20261005/source_manifest.json)列明运行绑定与历史失败版本边界。发布本身不代表训练通过。
