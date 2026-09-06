# BC＋DVAC讨论与实验现场（2026-09-05）

## 1. 17:01北京时间现场

本轮先服务器只读刷新，再讨论、备份轻量证据；没有改生产代码/config、重启任务或shared Ray。原始数据见`BC_DVAC_SERVER_REFRESH_20260905_LATEST.json`，不是交接快照。

| 实验 | 最新完整进度 | 指标与状态 |
|---|---|---|
| Sidney π0.5 | 110/200 | train147/256=57.42%；最近10轮均值64.77%；fixed110=17/32，105=24/32，历史最好70=26/32；无所查fatal/OOM/Traceback，ckpt110双rank＋full文件均在，未做恢复测试 |
| π0成功BC U10，GPU6 | 2/100 | 两轮train25/32、24/32；累计49成功episode；FM loss .02162→.01510；每轮约12.5分钟，更新约6.6分钟；尚无正式固定评估/ckpt，不据两轮训练或loss下降宣称提升 |
| Fast-WAM noOIDN | 17，已结束 | train65/256，fixed15=13/32；09-05 04:47 exit255；checkpoint10在；没有重启 |

π0.5前10轮train均值41.68%，最近10轮64.77%；前6次fixed均值45.83%，最近6次66.15%。这是同一次训练的描述性分段均值，既非独立重复实验，也不是“已经稳定”；Step110单点评估低于Step10，不能因此抹去中段改善，也不能据最佳26/32称持续81%。

[π0.5分栏图和逐点数据](../../rlinf-shenzhen-multitask-pi05/evidence/pi05-bc-review-20260905/index.html)。

整机：GPU4/5约66.94/67.35GiB；GPU6采样瞬时43.24GiB（非峰值，阶段变化正常；smoke峰77.46GiB）；1/2/3/7各4MiB、无compute。CPU128核、idle95%；RAMavailable1333.8GiB；swap已有2.64GiB但无即时进出，memory/io PSI无压力。/data仅余460.4GiB、87%已用。GPU1历史correctable SRAM ECC=2，所查uncorrectable=0/remapfailure=No；服务无failed。未查管理员kernel/SMART；不声称整机硬件完全无异常。

磁盘比算力更需关注：按当前单代ckpt大小，Sidney110→200余9代约241.6GiB，BC100十代约172GiB，加起来约414GiB，尚未计回放/日志和其他用户增长。因此两任务按现保存策略结束已较紧；未来GPU7再开同预算DVAC，先明确存储/保留策略，不擅自删旧产物。

## 2. BC＋DVAC推荐的干净增量

唯一设计稿：[01_DVAC_DESIGN.md §6—9](../01_DVAC_DESIGN.md#6-当前推荐配方让模糊点变成明确选择)。本轮只讨论，没有实现、没有创建GPU7任务。

主线：同次M4 ODE已有endpoint → L3尾方差V[50] → 成功筛选/入池时映射w → 累计成功池 → 原生masked FM误差乘w后平均。原SFT/空池起跑，继承32×1、micro32/global1024/U10、M4、评估16×2，GPU6不变。

建议高V相对多学、α0.25、过去5个采集轮次统计、首轮等权；入池固定w，不随反复抽样改写。V来自所有新有效query（失败只交小统计）；监督只学成功命令。w=1+0.25×中心化clip-z，每个query均值1且[0,2]；例[.75,1,1.25,1]。均值1守住系数总量，不保证实际梯度大小不变。固定位置趋势可能参与收益，不把分歧称作value/已证明credit。

代码五个既有接点＋小权重模块/薄配置/集中测试，不改runner/FSDP/checkpoint算法或仿真依赖。必须穿过collector白名单；BC采集model mode是eval，需显式record开关；保持RNG调用顺序。权重标定状态随replay恢复。详见设计稿接口与公式，α/方向/固定权重是待用户确认的新方法选择。

## 3. Git核查与补推边界

16:43全量本人范围：23工作树、22codex分支；21已与personal远端一致，缺诊断branch；已有1463个tracked轻量证据文件，约41.11MiB。94个候选run/diagnostic目录中67命中直接来源标记，其余27需按layout解释，不能直接当漏备份。

16:55已补：PPO历史one-update/reload/fixed、Fast scene-fence/OIDN开关诊断、RLT checkpoint诊断日志与未合并patch快照，合计92文件约526KiB。三份证据提交分别c41b7ff0/0d5daf6f/d3acd650并push/远端校验；缺失诊断branch已push。原诊断dirty文件及OIDN未跟踪文件未修改、未套用到训练；其内容现在有快照备份。详见[补推输出](../../server-admin/SHENZHEN_LIGHT_BACKFILL_FOLLOWUP_20260905.txt)。

本轮再发布本设计/现场摘要及π0.5既有正式100轮轻量包、当前110轮图。执行结果记在[实施账本](BC_DVAC_REVIEW_AND_GIT_LEDGER_20260905.md)。committed分支覆盖不等于全部历史日志100%完整，更不包含权重、回放图像和视频；活动BC/π0.5的最终日志尚未生成。
