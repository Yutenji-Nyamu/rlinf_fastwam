# 成功在线BC＋DVAC：实施、验收与现场简报

本轮授权为独立实现、简要测试与单卡smoke；未授权新DVAC正式长训。方法唯一设计仍为[设计§11](../01_DVAC_DESIGN.md#11-已授权的独立实现与gpu7验收入口)，逐操作证据在[实施账本](DVAC_IMPLEMENTATION_LEDGER_20260905.md)。

保存状态后续变更（09-05 20:49）：用户批准清理后，BC v7和本次DVAC smoke各Step1的大权重已精确删除，Step2保留，日志/小状态不变；新BC v8两代未动。[删除清单与独立复查](../../server-admin/CHENYITENG_CHECKPOINT_PRUNE_LEDGER_20260905.md#最终结果与保留边界)。下方“两代保存通过”是当时验收事实，不代表清理后两代大权重仍都存在。

## 1. 用户问题：接点与依据

**旧RLT不是FM。** 旧RLT在actor目标内对逐动作位置的动作回归BC平方误差乘权重，然后与−Q合并；DVAC不加到reward、critic或−Q。本次在相同“监督项”位置加权，但监督项换为π0原生FM。项目源码证据为[切入点调研§5](BC_DVAC_PIPELINE_ENTRY_RELATED_WORK_20260905.md#5-我们旧rlt到底接在哪里)。

**AttenA+最接近的是粒度与插入位置，不是信号来源。** [作者锁定π0源码](https://github.com/DaojiePENG/openpi/blob/fb3954ba7c0a2d0852b7f15c4445bcdb2ec7bee8/src/openpi/models/pi0.py#L223)在仍保留动作位置H的监督误差上加权。本次沿这个位置接DVAC endpoint方差信号，不引入AttenA+的动作幅度先验，也不引入新的attention模块。GRPO/RLT只借各自已核对的信号/标定与均值1监督组织，不搬旧训练目标。

**“FM误差加权”就是“loss加权”。** RA-BC、RynnValue-IQL也在监督loss归约前加权，通常粒度为整query；本项目需要50个动作位置各有一个权重，必须在H平均之前接。不是声称文献中某一具体实现数量“最多”；已核对的同模型、同粒度代码使该接点足够有依据。其他入口如数据重采样、模型条件、噪声—动作配对确实存在，但会改更多管线。完整比较见[作者代码表](BC_DVAC_PIPELINE_ENTRY_RELATED_WORK_20260905.md#2-直接对监督项加权实际代码与粒度)。

方法链：原采集 → 原成功过滤 → 原累计池均匀有放回抽样 → **FM[B,50,14] × detached w[B,50,1]** → 原mask/平均/反传/Adam。

每个未来14维动作向量一个权重，仍是action-level；不先压成chunk标量。不修改动作标签、FM训练噪声/时间采样、梯度累积、LR、优化器或部署动作。

## 2. 实现范围与测试

- 独立树`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi0-online-bc-dvac`；从原BC `385d4e75`建立，源码`736b1416`已推送`personal/codex/sz-pi0-online-bc-dvac`。原BC树未改。
- 新`online_bc_dvac.py`136行；原五个文件+115/-12，仅连接同次推理信号、collector统计、入池固定权重、loss可选权重、sidecar保存恢复。新增集中测试307行、薄配置组18行及说明。不是另造采集/训练/FSDP框架。
- 同次M4取末L3 endpoint，记录V[50]；无额外forward/RNG抽样。以过去5轮新query的log统计标定，高V相对多学，α0.25、z裁剪±2，chunk均值1/理论范围[0,2]；首轮权重1，入池后固定。失败只贡献小统计，不把失败RGB写入成功池。
- 服务器22/22测试通过，含原BC11项及新DVAC11项；[原始结果](DVAC_TEST_CONFIG_RETEST_20260905.txt)。真实RLinf配置校验与逐叶差异检查通过；[完整resolved](DVAC_SMOKE_RESOLVED_20260905.yaml)、[差异](DVAC_FORMAL_BASE_TO_SMOKE_DIFF_20260905.json)。CPU测试中模型/策略部分使用受控替身，不冒称原生模型完整恢复。
- GPU7容量smoke保留train32×1、micro32/global1024/U10/M4、eval16×2；仅总轮数2、eval/save每轮。64次训练尝试、20次Adam、20480个chunk呈现、64次固定评估、两代checkpoint。完整命令/预算/停止条件见[合同](GPU7_DVAC_SMOKE_CONTRACT_20260905.md)。

## 3. 单卡smoke验收

**验收通过。** 18:53:14—19:31:36，38分22秒，exit0；完整2轮/64次采集/20次Adam/64固定评估/两代checkpoint。GPU7已释放，无所查错误。原始[验收JSON](DVAC_SMOKE_VERIFICATION_20260905.json)与[现场](DVAC_SMOKE_STATUS_20260905.json)。

| 完整轮 | 采集成功/32 | fixed/32 | 平均FM loss | 累计成功episode / query |
|---|---:|---:|---:|---:|
| 1 | 24 | 24 | 0.022524 | 24 / 72 |
| 2 | 21 | 25 | 0.015727 | 45 / 135 |

实际新动作位置5200→5350，失败只进入小统计；第一轮72个成功query全部w=1，第二轮63个新成功query使用前轮统计，w均值1/std0.15960/范围0.60236—1.78022。CPU读回保存的sidecar与replay，逐记录重建权重完全一致，旧72条记录的V/w未变；replay RNG恢复抽样一致。两代native shard各10390434238字节、full各8065002471字节，learner更新计数10/20，replay与dvac.pt均在且已读取。

资源每5秒采样峰79576MiB＝77.71GiB/79.65GiB；Env FD最高1003、RSS约74.96GiB；主机RAM available最低1203.51GiB（包含Sidney及后段GPU6并行，不是DVAC独占内存）。显存余量仍窄；不能宣称修好了原第6轮OOM。

这是完整真实采集/更新/同步/评估/保存及新CPU状态读回验收，**没有生产worker全模型/优化器重启恢复测试**；两轮数字不用于判断DVAC学习提升。

## 4. 原BC故障与服务器边界

原GPU6 BC正式不是仍在运行：17:44:40已退出255，完整5轮，train25/24/28/22/25 /32；首次fixed5为27/32，FM loss0.02162→0.01041，累计124成功episode。第6轮在rollout VLM prefix/Gemma MLP申请816MiB时CUDA OOM，仅余469MiB；Env53.11GiB、Actor14.85GiB、Rollout10.68GiB。不是本轮DVAC改动引起，本轮代码在它退出后才独立创建。未到save10，所以无正式checkpoint；成功archive保留。[首错与源锁](DVAC_PREFLIGHT_20260905.json)。

用户后续明确改原BC为8×4并要求GPU6并行；源码a8764944仅改三个eval叶值及测试，11tests/同32固定种子核对通过。GPU6已于20:05:53正常完成两轮，20:14实际验收通过：[唯一8×4接续账本](BC_EVAL8_RESTART_LEDGER_20260905.md)、[验收JSON](BC_EVAL8_SMOKE_VERIFICATION_20260905.json)。GPU7本次没有热改配置，仍是16×2；这两项smoke因此不构成严格方法效果比较。用户又明确**先完成smoke、暂不放正式**；不自动重启原BC/new DVAC formal，不切/home，也不删旧产物。

| GPU6原BC eval8×4，完整轮 | 采集成功/32 | fixed/32 | FM loss | 累计成功episode / query |
|---|---:|---:|---:|---:|
| 1 | 26 | 24 | 0.023685 | 26 / 78 |
| 2 | 23 | 25 | 0.016830 | 49 / 147 |

完整64训练尝试/20Adam/64固定评估/两代checkpoint；模型文件、learner.update_step10/20和replay已核验，原BC池不含DVAC权重。每5秒采样GPU峰69.52GiB（原v7 eval16×2为77.46GiB），Env FD峰882、RSS62.50GiB；主机available最低1203.40GiB/PSI0。无所查错误，GPU6释放。该改动增加了短测显存余量，但仍不是100轮或完整worker恢复验证。GPU6配置/验收轻量证据`2467d997`已push，原树clean；训练算法不变。

训练曲线与服务器图已按20:17现场刷新并完成PNG可读性检查：[交互图](bc-dvac-status-20260905/index.html)、[成功率](bc-dvac-status-20260905/success.png)、[优化](bc-dvac-status-20260905/optimization.png)、[资源](bc-dvac-status-20260905/resources.png)。图中BC指已退出的原正式5轮；新smoke的两轮单列上表，不混作连续训练。不同任务分栏，不据同一step直接排名。Sidney保持另一窗口管理，本轮只读；无其他用户/共享Ray/依赖/模型变更。

## 5. 最终服务器只读快照与存储边界

09-05 20:17:06—20:17:13 CST，原始[现场JSON](BC_DVAC_SERVER_REFRESH_20260905_LATEST.json)。Sidney完整118/200，第119轮采集已结束、更新阶段；train118=169/256=66.02%，MA10=66.09%；fixed115=21/32=65.625%，最佳保存点70为26/32。checkpoint110双native rank和full逐文件在，原wrapper602620/driver602627继续，无所查fatal/OOM/Traceback等；本轮不做其恢复测试、不改其参数或工作树。

GPU4/5采样74.51/74.92GiB，GPU6/7各11/10MiB且无compute进程，GPU1/2/3亦无compute；GPU0其他用户约9.63GiB，仅计整体占用、不再定向调查。RAM available1350.66GiB，CPU load3.59/128核、近期97%idle，memory/I/O PSI0、无即时swap-in/out。SSH/代理服务active、无failed systemd unit；GPU不可纠正ECC与row-remap failure均0，GPU1有2条可纠正SRAM计数，不能从单次读数判断是否新增；本轮未做管理员SMART/内核日志审计。

`/data`可用370.04GiB/已用89%，`/home`可用1246.15GiB/已用47%；inode不紧张。19:24的422GiB可用是两项smoke末几代保存前快照，不再称当前。未来新formal的存储位置/保留策略仍需用户明确，目前只停在smoke验收。

用户最新存储问题已在[独立清理稿§5](../../server-admin/SZ_CHEN_STORAGE_CLEANUP_DISCUSSION_20260905.md#5-追问348gib是什么上次删过后为什么还有)逐项回答：历史已删146大文件1051.64GiB；当前旧GRPO/PPO残留81文件347.73GiB；加两个探针33.40GiB和v7/DVAC第1代34.38GiB，可讨论第一批合计415.51GiB。**未删除/移动任何文件，未改变磁盘保留比例。**这批只是初筛，不是执行allowlist或新的删除授权。
