# π0.5在线BC：独立实施与单卡smoke账本

## 授权与源锁

用户本轮明确授权“实现，简要检查，smoke，pi0.5 bc”。按前轮推荐继承原BC：32×1、micro32/global1024/U10、eval8×4、原Sidney SFT/pillbottle/H200/M10、expert-only、无增强/demo=0；单卡GPU6优先，GPU7预留。不授权formal、升级依赖、重启shared Ray、干预Sidney或其他用户。

唯一迁移计划为[02上下文](../02_PI05_ONLINE_BC_PLAN.md)。预定从原BC `2467d997`创建`codex/sz-pi05-online-bc`独立worktree，新增薄配置并迁入已验证Sidney数据adapter。实际源锁、完整resolved、命令及资源由下方执行前现场记录确认。

## 逐操作记录

1. 完整读取AGENTS/PROJECT_CONTEXT/HANDOFF/窗口交接及π0.5 BC唯一计划；只按本次迁移需要读原BC launch/observer/验收和测试，不遍历其他算法。读取命令一次本地PowerShell索引笔误在解析时退出，无远端操作，改正后读完。
2. 21:28普通账号固定host-key只读刷新GPU/RAM、进程、Sidney日志/checkpoint和磁盘；下一步核对精确基线和新分支/路径不存在，再开始创建独立worktree。开始本账本时尚无生产修改／测试／新训练。
3. 21:31prepare在新分支不存在检查处停止：git show-ref未加quiet，缺失ref返回128而非预期1；发生于worktree add/mkdir之前，未改生产。改为标准quiet存在性检查，仍先断言目标目录／packet／run不存在及基线clean再准备；不重放已执行的创建操作。
4. 21:33prepare成功：原BC HEAD2467d997、Sidney81be3193均clean，GPU6无compute。独立分支/worktree `codex/sz-pi05-online-bc`／`.../worktrees/pi05-online-bc`从原BC建立，未改原两树。仅取回待编辑两个adapter及必要配置/环境代码；本地不是完整仓库，缺失文件从已有服务器源码证据读取，不假定本地全量checkout。
5. 一个主体批次：两个adapter迁入Sidney同样的注册与可选mean/std覆盖；新21行薄YAML继承原BC，只指定pillbottle、Sidney模型及M10；新增两项集中CPU测试，另复用原11项。部署和验证工具均限定新树，资源观察器直接复用GPU6已有版本。未改变collector/replay/FM/actor/FSDP/保存代码。
6. 准备服务器验证：pytest原11项＋新增2项；实际Hydra/validate_cfg逐叶与v8比对；真实norm/tokenizer/动作往返／采集token在SFT准备中保留；固定评估8×4与Sidney两rank16对照。原生π0.5 SFT/同步/保存由接下来的完整两轮smoke覆盖，不再额外写组件权重探针。
7. 21:41 CPU测试12/13通过，原11全过；新配置组合失败于Hydra禁止子配置设置hydra.searchpath。未运行环境／模型GPU，也未改变学习链。窄修采用标准全局配置组`online_bc_model/pi05_sidney.yaml`，在原BC主入口追加`+online_bc_model=pi05_sidney`；删除本轮新建未提交的失败入口文件，不改原BC YAML／预算。两处adapter及测试其余不变，重新组合和测试。
8. 21:47测试13/13通过、真实配置确认GPU6。fixed32检查发现单rank原表前32与双rank各16只前半一致：现有partition先打乱再分进程，不是环境初始化相同。采用独立`eval_sidney_fixed32.json`：按旧双rank实际32ID及现有分区函数的逆排列生成，仅在π0.5配置组选用；不改训练seed表、原eval表或环境代码。此为保持评估初态的配置适配，不改变评估总量／并发／学习预算。
9. 21:54新增seed表后13/13再次通过，8×4与旧双rank固定32已完全同序通过。真实数据检查在对比动作时发现测试预期未处理原生FP32训练cast：JSON norm计算为FP64、现有prepare明确转FP32。这是检查脚本dtype预期错误，不是新的模型精度故障；只将预期按原生FP32转换比较，往返按同dtype比较，不改任何生产代码或容差预算。只重跑实际data/config检查，不重跑已通过pytest。
10. 21:57:29真实data/config检查通过：prepared actions[2,50,32]、tokens[2,200]；不同state改变token，采集token在SFT中精确保留，动作归一化往返最大误差6.10e-8；固定32与Sidney双rank完全同序。无模型forward/更新或GPU分配。保存完整resolved、逐叶delta、validation与种子表；接着只提交本次5文件，两个adapter与Sidney对应文件逐字节cmp，原BC预算配置不改。
11. 21:59代码提交／推送通过：`653fe0fbc05188eb0ec19077de5c78a00b8230ad`，5文件+188/-1；两个adapter与Sidney文件cmp完全相同。22行配置组、38行独立种子表、101行测试；原BC生产链未改。工作树clean，个人远端`codex/sz-pi05-online-bc`已创建；没有PR或force push。
12. 已回取服务器格式化后的5文件、完整resolved、delta/validation，与本地核对。聊天已展示[GPU6完整合同](GPU6_PI05_BC_SMOKE_CONTRACT_20260905.md)及resolved链接、两轮/32×1/B1024/U10/eval8×4、40—70分钟估计/90分钟上限；按用户本轮授权执行smoke，目标唯一v1 run，禁止formal。启动工具再次检查新树clean、GPU6无compute、run不存在，逐文件/配置字节一致才创建目录与wrapper。
13. wrapper实际22:01:28启动，PID/PGID1929038、observer1929039；driver1929043，Actor1929646/Rollout1929649/Env1929651均属于新GPU6单卡run。22:03初始化中无错误，Env已打印独立FD4096；原shared Ray仍在。没有启动第二任务／正式训练。
14. 22:09首轮32条采集完成（307.68秒），22:10GPU6约65.29GiB、主机可用1256.8GiB、Env FD762，所查错误空；进入后续更新阶段，尚无完整第一轮/评估/保存结论。只读watch每30秒记录到`PI05_BC_WATCH_20260905.jsonl`，最长与本smoke相同90分钟，无进程控制或重启逻辑。
15. 22:16只读核实Actor真实SFT持续前进、GPU6利用率100%／65.32GiB，非存活不动；22:17开始固定评估，说明首轮原生更新后权重导出／同步已跨过旧FSDP故障边界。22:18评估1/4、峰66.11GiB、FD882、无错误；尚待两轮完整保存验收。
16. 22:22固定评估4/4完成并开始Step1保存，22:23日志进入Global Step1/2及第二轮采集。22:25:40峰值仍66.11GiB、错误空，第二轮采样继续；未重复launch。此时只确认第一轮流程跨过，不提前认领两轮通过或完整恢复。
17. 22:28第二轮32条采集完成（303.95秒），第二次SFT期间峰值升至73.74GiB；22:33只读确认GPU6利用率100%、64℃、Env FD880／已测最大882、host available约1209GiB、PSI0、错误空。这是实际评估后再采样／更新，不是只测首轮。补充只读结束验收／精确worker状态及小证据发布清单，不改运行中的源码或配置。
18. 22:36第二轮更新／同步后开始固定评估；22:40:54评估4/4完成并开始Step2保存。22:41:20正常exit0，总墙钟39分52秒，无新fatal/OOM/断言；watch随后正常结束，无第二次launch或任何停止他人/共享服务操作。
19. 22:42运行只读`sz_pi05_bc_verify_20260905.sh`通过：两代完整预期文件、learner10/20Adam、累计8/26成功episode和28/89query；replay RNG读回一致。action_out_proj及time_mlp_in已变、抽查VLM q_proj未变；loss/grad有限。train8/18、fixed11/18 /32；native10.966GiB/full7.941GiB，含回放共37.892GiB两代。GPU峰73.743GiB、host available最低1206.59GiB、全机PSI短暂峰2.09、Env FD882；GPU6验收时11MiB。范围不是完整worker/optimizer恢复或100轮稳定。
20. 22:45仅发布本run小证据包12文件：合同、resolved/delta、CPU检查及失败预期修正说明、机器验收、4个运行/验证脚本；无密码/大权重/全量日志。证据commit`912bc6907d39a0eec1eb98a6d4c9358e69791924`已push到同一独立分支且clean，源码仍653fe0fb；更新02唯一结果、00路由和HANDOFF本专题，不覆盖另一窗口Sidney条目。未放正式。
21. 22:47:20最终只读刷新：wrapper和Actor/Env/Rollout三个精确PID均不存在，GPU6为11MiB/0%；HEAD912bc690、dirty为空、runtime源锁仍653fe0fb。主机available约1317.42GiB，/data available约875.12GiB、/home约1246.15GiB；证据`PI05_BC_FINAL_STATUS_20260905.json`。本轮训练已结束，不留继续launch/stop事项；正式待新合同确认。本地目标文档diff whitespace检查通过。
