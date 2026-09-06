# π0.5 原地从100续到200：执行账本

## 1. 用户授权和合同

2026-09-05本窗口用户明确要求：“盯着pi0.5的实验，结束了，就改为200步，其他不动，原地放下去”。此授权覆盖等待当前100步自然结束、从完整step100恢复并执行101—200，以及必要的启动与健康跟进；不再重复请求相同批准。

保留原实验目录、GPU4/5、任务、模型、所有采样/优化/评估/保存参数。仅将 `runner.max_steps` 从100改200，`runner.resume_dir` 从null改原实验global_step_100。原启动配置、日志、checkpoint保留；接续命令与运行日志放原run下独立 `runtime-resume100-to200` 目录。恢复模型、Adam、scheduler与actor RNG；环境重新初始化，不声称轨迹逐比特连续。

新增预算：100轮、25,600训练轨迹、最多102,400 query records、5,120,000指令action槽、200次optimizer call、640固定评测episodes、10代checkpoint。GPU4/5两卡，约41—42小时、82—84 GPU小时；预计新增checkpoint约268.5GiB，启动前刷新磁盘空间。保留原345600秒超时上限，正常于global_step200停止；异常退出则报告，不自动改参或循环重启。

## 2. 逐操作记录

1. 已完整读取根上下文、HANDOFF与恢复源码审阅；当前用户授权覆盖前轮只读边界。检查现有automation，仅有独立每日调研，无重复π0.5跟进。
2. 已通过固定host-key、进程内密码的chenyiteng SSH执行既有只读状态脚本，结果保存 `pi05_resume200_preflight_status_20260905.json`；后续据现场结果填写。
3. 正在核对实际启动命令、配置、源码与同目录日志行为，准备逐叶仅两项变更的resolved和精确启动命令。尚未启动接续。

4. 12:43:51现场确认Sidney HEAD仍f50e235c、clean；已取回实际MetricLogger、RecordVideo、env_worker及原始命令/环境。TB固定config.yaml会更新、视频编号会归零；原robotwin_data无产物。确定原runtime完整保留，启动时先拷贝旧TB配置，原video/eval同盘改名video/eval_steps001_100，续训重建原视频目录。
5. 已执行 `local_scripts/remote_commands/sz_pi05_prepare_resume200_20260905.sh`，在原run新建 `runtime-resume100-to200`；CPU仅compose，退出码0。实际resolved逐叶比较通过，只有max_steps与resume_dir两叶改变，模型/算法源码未改。packet含独占启动标记，重复执行只报告已尝试；正常100未结束或GPU4/5未释放时只等待，不停止任何进程。
6. 完整配置：[RESUME100_TO200_RESOLVED_20260905.yaml](RESUME100_TO200_RESOLVED_20260905.yaml)；精确训练命令：[RESUME100_TO200_COMMAND_20260905.sh](RESUME100_TO200_COMMAND_20260905.sh)。实际准备结果及hash：`pi05_resume200_prepared_20260905.json`。
7. 已建立本任务heartbeat `0-5-100-200`，每5分钟检查；续训健康起步后改每小时，只在完成/失败/需处理时通知。
8. 用户追加授权100步结束时提供轻量ZIP：原始100步指标、配置、关键日志、资源和可视化；排除模型/checkpoint/视频大文件。原100步driver/runtime保留，TensorBoard导出限定raw step<100，避免与续训指标混入。正在准备打包脚本。
9. 独立agent审阅生成的launch.py通过：未发现恢复/覆盖/重复启动问题，无kill/shared Ray修改；最终只读closeout额外确认旧RLinf namespace清空。12:50:41原run仍完整99，step100 rollout3/4，错误计数0，未执行launch。
10. 12:54:45现场step100采样4/4完成，已进入最终fixed评估，尚未finished/checkpoint100，错误计数0。新打包脚本 `local_scripts/package_pi05_formal100_20260905.py` 已完成；真实99步preview PNG已视检，最终包只在100训练点、20个评测点、exit0、checkpoint100齐备后生成，不伪造最终结果。
11. 13:01:40最终只读快照确认原run于13:00:18正常exit0，完整100/100；train100=165/256=64.45%、MA10=66.91%、fixed100=19/32=59.38%；错误计数0。两rank各10,150,817,963 bytes、full8,526,574,644 bytes完整存在。旧RLinf namespace无named actors，GPU4/5显存0且无计算。`pi05_formal100_closeout_20260905.json`冻结，不再用续训状态覆盖。立即执行已审核原地接续；打包agent并行生成最终ZIP。
12. 执行 `sz_pi05_launch_prepared_resume200_20260905.sh` 返回LAUNCHED/exit0：wrapper602620、observer602621，原地目标global_step200、两叶变更。完成前置检查、原video/eval归档与TB配置备份；新进程已提交，真实checkpoint加载与rollout启动尚待日志确认。启动结果：`pi05_resume200_launch_20260905.json`。不重复launch。
13. 100步轻量产物包已完成：[formal100-summary-20260905.zip](formal100-summary-20260905.zip)，441001 bytes（约431KiB），20文件。完整100训练点、20eval、49项scalar、2432资源采样、原配置/命令/源码锁、关键日志、checkpoint清单、3PNG和离线HTML；无权重/视频/TB二进制/Ray完整日志。3PNG及手机HTML视检通过；9指标hover、11本地链接、390px无横溢出、无页面异常/外部请求，ZIP CRC通过。QA证据在同级formal100-summary-20260905-qa，未塞入ZIP。
14. 13:07:43只读确认接续正常：13:02:39启动，driver于13:05:59记录从global_step_100恢复，随后出现rollout0/4。按已核对runner源码，actor.load_checkpoint().wait()完成后才进入该采样，故已确认真实恢复并开始101轮（尚未完成101）。wrapper602620存活，所查fatal/OOM/RuntimeError/Traceback为0；GPU4/5约23.64/24.01GiB。证据：`pi05_resume200_startup_20260905.json`。完成健康启动检查，不继续密集轮询。
15. heartbeat `0-5-100-200` 已改名“π0.5原地续训到200步跟进”、降为每小时；正常安静，200完成/异常/需处理时通知，200完成后暂停。100步ZIP已交付，后续检查只用新runtime，不覆盖冻结100步快照或重复launch。
16. 14:13:39小时只读检查：新runtime原wrapper602620存活，完整103/200，104轮rollout0/4；train103=160/256=62.50%，跨接续最近10轮均值66.02%；最新fixed仍step100的19/32，下一次105，最新checkpoint100，下一次110。所查fatal/OOM/RuntimeError/Traceback均0，无结束标记；GPU4/5约23.07/23.61GiB，/data余583.09GiB。证据：`pi05_resume200_heartbeat_20260905_1412.json`。运行正常，无需用户处理；不通知例行进度、不改变任务或自动化。
17. 15:14:02小时只读检查：完整105/200，106轮rollout2/4；train105=179/256=69.92%，跨接续MA10=65.35%；续训首次fixed105=24/32=75.00%，较100的19/32回升，未超过历史best70的26/32，不以单点评估声称稳定增益。原wrapper602620存活，结束/退出标记为空，所查fatal/OOM/RuntimeError/Traceback均0。最新checkpoint100，尚未到110保存；GPU4/5约66.94/67.23GiB，/data余561.11GiB，仍高于本续训尚待保存10代checkpoint约268.5GiB预算。证据：`pi05_resume200_heartbeat_20260905_1513.json`。正常推进，保持安静；未改变服务器或自动化。
18. 16:15:34小时检查：完整108/200，109轮rollout1/4；train108=152/256=59.38%，MA10=65.55%，fixed仍105的24/32；原wrapper存活，无结束标记，所查fatal/OOM/RuntimeError/Traceback均0，checkpoint仍100。GPU4/5约66.94/67.35GiB。证据：`pi05_resume200_heartbeat_20260905_1615.json`。
19. 观察到/data空闲连续下降：13:02为626.15GiB、14:13为583.09、15:14为561.11、16:15为518.86；3.215h减少107.29GiB，期间本run无新增checkpoint。触发一次有界低优先级只读du核对本run，16:17:09整个run占268.49GiB，几乎全部为原30个checkpoint文件合计268.48GiB，其他目录约6.78MiB；未查看他人目录或调整任务。现场/data余518.43GiB，剩余110—200十代仍预计268.48GiB，当前及下次110保存足够。若最近约33.37GiB/h的共享盘下降持续，后续预算有风险；写入来源和未来速率未确定，不能声称必满或归因π0.5。原始构成：`pi05_disk_readonly_20260905_1616.json`。向用户一次性提示容量趋势，继续原配置运行与每小时检查；同一趋势不重复刷屏，余量低于剩余保存预算或出现写盘错误时再提示，不自行清理/改保存频率/停止训练。

20. 17:17:54小时只读检查：执行既有`sz_pi05_resume200_status_20260905.sh`，退出码0，证据`pi05_resume200_heartbeat_20260905_1717.json`。完整110/200，111轮rollout3/4；train110=147/256=57.42%，续训101—110均值64.77%；fixed110=17/32=53.13%，相对105的24/32回落，保留波动记录，不据单点调整参数或判断退化。原wrapper602620存活，无finished/exit标记，所查fatal/OOM/RuntimeError/Traceback均0。最新checkpoint110目录已在，本轮未逐文件核验或恢复测试；GPU4/5约66.94/53.62GiB，/data余454.73GiB，高于本run后续120—200九代预计241.64GiB保存预算。上轮已报告容量趋势，本轮仍正常且无需新增用户动作，保持安静；未改服务器、冻结100步产物或自动化。

21. 18:18:40小时只读检查：执行既有`sz_pi05_resume200_status_20260905.sh`，退出码0，证据`pi05_resume200_heartbeat_20260905_1818.json`。完整113/200，114轮rollout3/4；train113=176/256=68.75%，MA10=64.80%；fixed仍110的17/32，下一次115。原wrapper602620存活，无finished/exit标记，所查fatal/OOM/RuntimeError/Traceback均0；checkpoint目录最新110，本轮未逐文件核验或恢复测试。GPU4/5约41.77/40.40GiB，/data余439.43GiB，较17:17减少15.31GiB，仍高于本run后续九代预计241.64GiB保存预算。正常推进，已提示过的容量趋势不重复通知，继续每小时只读跟进；服务器、冻结100步产物与自动化均未改变。

22. 19:19:06小时只读检查：执行既有`sz_pi05_resume200_status_20260905.sh`，退出码0，证据`pi05_resume200_heartbeat_20260905_1918.json`。完整116/200，117轮rollout1/4；train116=182/256=71.09%，MA10=65.12%；fixed115=21/32=65.63%，由110的17/32回升，未据单点评估判断持续提升。原wrapper602620存活，无finished/exit标记，所查fatal/OOM/RuntimeError/Traceback均0；checkpoint目录最新110，本轮未逐文件核验或恢复测试。GPU4/5约45.73/46.59GiB，/data余422.04GiB，较18:18减少17.39GiB，仍高于本run后续九代预计241.64GiB保存预算。正常推进，保持安静并继续每小时跟进；未修改服务器、冻结100步产物或自动化。

23. 20:20:39小时只读检查：执行既有`sz_pi05_resume200_status_20260905.sh`，退出码0，证据`pi05_resume200_heartbeat_20260905_2020.json`。完整119/200，120轮rollout0/4；train119=179/256=69.92%，MA10=66.68%；fixed仍115的21/32，下一次120。原wrapper602620存活，无finished/exit标记，所查fatal/OOM/RuntimeError/Traceback均0；checkpoint目录最新110，本轮未逐文件核验或恢复测试。GPU4/5约58.38/58.54GiB，/data余370.03GiB，较19:19减少52.00GiB，仍高于本run后续九代预计241.64GiB保存预算。当前推进正常，容量下降已提示，尚未达到此前记录的再次告警条件；保持安静并继续每小时跟进。未修改服务器、冻结100步产物或自动化。

24. 21:22:26小时只读检查：执行既有`sz_pi05_resume200_status_20260905.sh`，退出码0，证据`pi05_resume200_heartbeat_20260905_2121.json`。完整121/200，122轮rollout3/4；train121=161/256=62.89%，MA10=67.73%；fixed120=19/32=59.38%。原wrapper602620存活，无finished/exit标记，所查fatal/OOM/RuntimeError/Traceback均0；checkpoint目录最新120，本轮未逐文件核验或恢复测试。GPU4/5约50.72/53.65GiB，/data现场余913.21GiB，高于本run后续八代预计214.79GiB保存预算。HANDOFF记录其他窗口20:49已执行获批101个大文件清理、21:12独立复查通过，且未删当前Sidney；具体依据在`docs/server-admin/CHENYITENG_CHECKPOINT_PRUNE_LEDGER_20260905.md`，本heartbeat未参与或重做清理。现场确认空闲容量回升，先前容量压力缓解，继续正常只读跟进；不重复其他窗口清理通知，不改服务器、冻结100步产物或自动化。

## 3. 后续执行入口

1. 固定host-key SSH执行 `local_scripts/remote_commands/sz_pi05_closeout100_readonly_20260905.sh`，将完整stdout直接保存为 `pi05_formal100_closeout_20260905.json`。不要根据历史快照宣称完成。
2. 确认原run exit0与step100 checkpoint后执行 `local_scripts/remote_commands/sz_pi05_launch_prepared_resume200_20260905.sh`。它会再次核对结束、checkpoint zip目录、源码/hash、GPU4/5无compute、磁盘预算和无重复启动，再归档视频/配置并launch；尚未就绪返回WAITING，无破坏性操作。
3. 使用 `local_scripts/remote_commands/sz_pi05_resume200_status_20260905.sh` 从新driver/TB确认local-shard恢复、global step100起点和101轮采样；不把仅PID存活当恢复成功。若失败，保留日志，不重复launch或自动改参。
4. 使用bundled Python执行 `local_scripts/package_pi05_formal100_20260905.py`（默认读取本专题 `pi05_formal100_closeout_20260905.json`），生成 `formal100-summary-20260905/` 与同名ZIP。实际view_image检查3张PNG，核验HTML能正常显示，再给用户ZIP链接；控制在5MB以内。
5. 恢复健康后将heartbeat降为每小时，200完成或失败时报告并暂停。100→200接续后原runtime/finished_at仍表示前100已结束，活动进程应查看新runtime，不能误判训练整体已停。
