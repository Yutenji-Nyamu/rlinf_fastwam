# 深圳3 WM修复后进展与验收

2026-10-01，r6从原固定SFT重开，沿原1000轮/save40，不重跑smoke。源码修复与依据分别见[修复专题](WAN_GOAL_REPAIR_20261001.md)、[官方与社区线索](WAN_GOAL_REPAIR_SOURCES_20261001.md)。

## 15:53：10轮、机制与阶段资源核查

w159退出0，owner活、RUNNING_WM，完成step0–9十个runner轮，七轮有正mask/非零有限GRPO梯度，三轮全组过滤；第11轮采集4/8。最新step9 grad1.8499382、mask7.1972653%、有限loss0.00009743094和优势[-0.7245674,2.4748666]，本轮1000.206秒；monitor诊断及近期primary错误为空，尚无正式CP。

用户要求记录并行配置与每卡负载。w158 15:44更新阶段每卡48.38–48.79GiB且GPU利用率均100%；w157 15:39采集阶段62.1–62.5GiB；w160 15:55采集阶段60.9–61.3GiB。十二个GPU进程均验证为当前自有catalog身份，四卡各一个actor、rollout、env角色，每卡16个环境槽位、2组，Wan真实batch16。瞬时利用率会随等待变化，约62GiB不是常数或持续测量峰值。

每轮8批共512轨迹槽位、20,480 chunk槽位；B2048/MB128/四rank形成每次4个microbatch累积、每轮10次optimizer调度。有效runner轮与optimizer次数分开。实际奖励Goal TaskEmbed ResNet经round输出0/1，再差分；mask比例不是成功率。准确机制、原配方来源、单视角公开依据和现场证据见[训练机制与四卡并行](WAN_GOAL_MECHANISM_AND_PARALLEL_20261001.md)。没有改学习/owner源码、方法或预算，CP40仍是下一验收。

## 14:32已超过上一轮中断位置

w137退出0，固定身份/host-key验证通过；owner活、RUNNING_WM，完成step0–4共5轮，第6轮采集5/8，monitor_diagnostics={}、recent_primary_error_lines=[]，checkpoint目录仍空。上一轮r5在完成4轮后监控中断，这次已超过该位置；这不是原断言根因复现或长期稳定证明。

有效GRPO梯度出现于step0/1/2；step3/4为全组过滤，mask=grad=loss=0，优势min/max为空样本nan。原过滤0.1–0.9及全部训练参数保持；空优势统计不等于权重nan，grad0也不能证明optimizer/权重完全没动。真实完整参数检查仍在原CP40保存结束后进行。

四卡显存63577/63981/63983/63501MiB，可用内存约1.69TiB。最近3轮耗时约995/994/995秒，继续原1000轮，不据此扩预算或提前保存。WM奖励模型success仍不代表真实LIBERO表现。本次里程碑轻量发布同时纳入w129原RLT归还完整性证据与当前资源表。

## 13:18首次有效更新

细日志：`local_logs/wan-goal-20261001/steps/w123-repair-first-update/`，命令与身份/host-key校验、完整stdout、退出0均保留。TensorBoard step0写入时间约13:15；读取时间13:18:13–14。

|项目|实际结果|
|---|---|
|runner已完成轮数|1，step0|
|有效更新step|[0]，正mask且非零有限梯度|
|grad norm|0.5559872985|
|有效样本mask|0.0137695316，即1.376953%|
|优势min/max|-1.6201815605 / 0.5400605202|
|total loss|0.0001563128|
|本轮time/step|1066.72秒|
|阶段|owner活，RUNNING_WM，下一轮采集1/8|
|监控与primary错误|monitor_diagnostics={}，近期primary error=[]|
|GPU4/5/6/7显存|63575 / 63975 / 63977 / 63495MiB|
|整机可用内存|1,877,215,551,488字节，约1.71TiB|
|正式checkpoint|尚无，仍按原save40|

本轮证明修复后真实训练已推进；尚未越过原4轮故障点。WM奖励模型的success_once=0.013671875不是实际LIBERO成功率。

## 后续唯一运行与只读检查

- 当前run：`/data/chenyiteng/projects/wan-goal-sz3/runs/wan-goal-sz3-20261001-r6/pi05-formal`；桥接v6、cycle repair-v1。不得重放prepare/launch。
- `status_wake.sh`沿active pointer核启动/退出与归还；`repair_runtime_evidence.sh`核原配置、真实placement、完整标量、错误与嵌套experiment/checkpoints目录。
- 首先检查完成轮数超过4，并持续有有效GRPO信号；全组过滤的空优势不直接当作权重NaN。
- 首次完整CP40出现后，仅执行一次`envs/pi05-wan/bin/python -B scripts/verify_formal_checkpoint40.py`。此前不要执行；若保存未完导致检查未通过，保持训练，等完整后再读。
- 该reader规范化`/data`的真实路径，以兼容深圳3到`/home/nvme/team-data`的路由；CPU只读检查自有DCP metadata、model/optimizer/scheduler/RNG结构、storage extent，以及full-model全部浮点权重有限。metadata、全部引用shard和full_weights在检查前后复核inode/size/mtime，并复读metadata结构；应在保存结束且下一轮已推进后执行。它不恢复optimizer、不启动CUDA、不证明真实LIBERO效果。
- 成功回执：r6根目录`checkpoint40-verification.json`，存在后不重复写。CP仍由原训练保存到数据盘；临时编译缓存仍沿既有独立路径。
- WM结束/失败后，同一个v6 owner完整核清释放再恢复原四RLT，无Dojo评测或竞争restorer。正常推进由既有统一检查维护，仅重要里程碑/故障通知。

## 13:40–13:42：第二轮与原RLT归还检查

w128：完成step0/1两轮有效GRPO更新；step1 grad=0.6200936、loss=0.0000444308、优势[-0.5400605,1.6201816]有限，第三轮采集4/8，monitor诊断及近期primary error仍为空。尚未越过原4轮故障点。

另一窗口同步深圳2最新CP全索引多于已保存payload的实际缺陷。本窗未推定深圳3同病：w129沿当前cycle/active pointer，使用同一个冻结helper对原四CP125完整CPU inspect，小文件SHA、resume_dir、contract全部一致，all_valid=true，CUDA未初始化。四份索引与对应payload数量为19036/19020/19710/19631，没有缺失。当前不需要供体或独立恢复副本，也未改CP/cycle/config/运行源码。

既有归还路径：v6调用冻结`rlt_cycle_sz3.py resume`；它在driver启动前再次核CP小SHA及完整DCP/replay结构，不允许丢索引后只加载固定部分。准备阶段可记录newer_rejected并选较旧完整CP，归还阶段使用已冻结CP；status进展读取不能替代完整payload检查。细日志w128/w129；本次新增证据并入下个重要里程碑的轻量发布，当前远端仍02589f43。
