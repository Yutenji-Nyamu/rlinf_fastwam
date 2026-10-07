# BC U / Norm τ扫描与旧实验收尾（2026-10-08）

按用户结论结束旧8项实验，不再延长DSRL、原BC U/Norm及三组RLT信号实验。下表是实际末次日志与保留断点；各目录JSON含轻量曲线和源码HEAD，YAML保存实际配置。历史日志与末次完整权重继续留在服务器。单次结果不作跨任务、跨预算的统计显著性断言。

| 机器/卡 | 结束的实验 | 末记录step | 保留CP | 末次真实评估成功率 |
|---|---|---:|---:|---:|
| 1/4 | BC U | 248 | 240 | 71.875% |
| 1/5 | RLT U-BC | 1221 | 1200 | 100% |
| 1/6 | DSRL Clean | 95 | 65 | 0% |
| 1/7 | DSRL U | 94 | 65 | 8.33% |
| 2/6 | RLT Q-U | 753 | 750 | 95% |
| 2/7 | RLT Q-Norm | 773 | 750 | 100% |
| 3/6 | BC Norm | 133 | 130 | 68.75% |
| 3/7 | DSRL Norm | 66 | 65 | 8.33% |

新实验使用相同π0.5 SFT、move_pillbottle_pad、N8/U5、300轮、200物理动作、global1024/micro32、每5轮固定32条真实评估。U/Norm生产器不变；复用历史DVCA两层log/minmax/exp-mean映射，local/chunk τ相同。两项历史控制为20% chunk恢复等权，以及α在第1至200轮从1降至0。

| 机器 | GPU4 | GPU5 | GPU6 | GPU7 |
|---|---|---|---|---|
| 1 | U τ1 | Norm τ1 | U τ2 | Norm τ2 |
| 2 | 原EXPO | 原EXPO | U τ3 | Norm τ3 |
| 3 | 原WM | 原WM | U τ4 | Norm τ4 |

模型源码为105170b871；5项服务器CPU检查通过，覆盖历史公式一致性、τ强度、mask、私有随机数与退火终点。正式运行首轮验收另记录，不以启动进程代替训练成功。

接续采用每机一个逐卡CPU程序：每卡的优先实验退出，且其namespace和该卡C/G进程均释放后，恢复该卡原RLT断点及3000轮总预算。未知占用就等待。2机EXPO沿原monitor逻辑转交，只释放4/5；6/7不再等待EXPO。3机WM原管理和4/5回接保持。逐卡候补程序停止只退出自身，不结束训练；不停止共享Ray或其他用户。BC新产物保留最新两份完整断点，旧代只清理明确的大权重/重复回放文件并记账。

1机本轮删除36个旧smoke/非最终大文件227.41GiB，并删除Dojo pip下载缓存11.73GiB，共约239.14GiB。data从95%降至89%，home从91%降至88%。保留正式末次断点、RLT候补依赖、配置和日志。

部署修正：第一版1机环境误用了3机的`/usr/share/vulkan/icd.d/nvidia_icd.json`，本机不存在，导致四路初始化`failed to find a rendering device`；当时逐卡RLT实际接替成功。后续1机BC v2恢复本机已跑通的环境，保留失败记录。2/3机正式配置未因此改动。

## 正式首轮与最终路由

2026-10-08约01:46，八组均完成首轮（TensorBoard step 0）真实更新，τ与非均匀权重生效；各机`formal/`保存实际YAML、首轮标量、源码HEAD及当前管理脚本SHA。五项CPU针对性检查已通过，未追加长smoke。1机v2采用本机原有图形环境后恢复正常，四卡候补开关已重新启用。

全部原RLT按卡候补：1机4–7由本轮单个逐卡程序管理；2机4/5等EXPO实际释放、6/7各等自己的BC结束；3机6/7各等自己的BC，4/5继续原WM归还流程。2机不再整组等待EXPO和Q实验。EXPO原driver3503441、WM原driver861883保持；本次三机0–3均无C/G进程，未对其他用户执行启停。新逐卡程序只有CPU调度、精确退出清理和保留最新两份BC断点；未知占用时等待，不抢卡、不循环重启失败RLT。

三机本轮控制目录均为`/data/chenyiteng/deployment-20261008/bc-signal-tau-v1`；只使用当前`plan.json`、`status.json`和`formal-accepted.json`，旧入口不重放。SZ1当前BC与候补均v2；SZ2/3 BC维持v1。运行模型源码105170b871不变，轻量记录与调度源码发布在`codex/bc-signal-tau-20261008`。

SZ2 Q-U停止后完整检查点实际为CP750，已用complete.json复核，修正早先停止前CP725快照。SZ1清理清单与删除回执保留在同一控制目录的`cleanup-approved-files.json`、`cleanup-done.json`和`pip-cleaned.json`。
