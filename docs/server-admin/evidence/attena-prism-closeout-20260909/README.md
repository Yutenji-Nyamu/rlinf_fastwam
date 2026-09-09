# AttenA 与 Prism 最终轻量归档 · 2026-09-09

对照快照时间：2026-09-09T21:10:15.371576+08:00；AttenA与Prism终点以已冻结raw日志为准，SARM仅引用该快照。按本轮授权结束两组；精确进程、退出状态与其他实验保护以 [最终停止回执](stop-final-receipt.json) 为准，不把主动停止写成正常完成100/200轮。停止时尚未完整记录的轮次不补计。Prism wrapper记录exit0，但由SIGTERM handler结束，并非自然完成200轮；AttenA记录exit134。初始stop-receipt.json如实保留当时异步GPU释放尚未完成的状态，最终是否释放以stop-final-receipt.json为准。

[八张全程宽图](index.html) · [数值汇总](experiment-summary.json) · [绘图数据](plot-data.json) · [最后完整checkpoint保留清单](RETENTION_MANIFEST.json)

| 实验 | 状态 / 完成轮次 | 采集 MA5 / MA10 | 最新固定评估 |
|---|---|---:|---:|
| 干净 BC | 历史已结束 · R100 | 75.00% / 65.00% | R100：14/32 |
| 旧 DVAC [0,5] | 历史已结束 · R94 | 60.00% / 52.50% | R90：15/32 |
| DVAC new | 历史已结束 · R100 | 75.00% / 72.50% | R100：17/32 |
| AttenA FK | 本次授权停止 · R84 | 40.00% / 55.00% | R80：8/32 |
| SARM κ=2 | 运行中，仅引用快照 · R28 | 65.00% / 55.00% | R25：17/32 |
| 干净 π0.5 GRPO | 历史已结束 · R168 | 67.27% / 67.66% | R165：24/32 |
| Prism GRPO | 本次授权停止 · R87 | 67.66% / 67.30% | R85：23/32 |

## 对比现象

AttenA 共同固定R5–80平均35.55%，clean为43.36%（-7.81pp），旧DVAC为49.41%（-13.87pp），new为48.05%（-12.50pp）。与同样长度过滤≤3的new比较，胜/平/负为0/0/16。

Prism 共同固定R5–85平均61.58%，Control为56.43%（+5.15pp）；最近5个共同点R65–85差值-3.12pp。最新共同点分别23/32、24/32。Control完整背景仍保留到采集R168、固定R165。

BC全部曲线放在完整R100视野，Prism与Control放在完整R168视野，每组raw、MA5、MA10、fixed各一图；一行一张，颜色、图形、线型同时区分。BC上轴为新采集次数（每轮4次，不含固定评估）。曲线只画已记录数据，MA使用满5/10轮的尾随窗口，不补点、不外推。

clean、旧DVAC和SARM成功长度过滤关闭；new、AttenA为≤3，因此对clean的结果包含长度过滤差异。AttenA与new过滤一致。SARM使用额外冻结8B评分器，本次只作为4/U5的已有对照，未在本次停止。Prism与Control每轮256条、U2，Prism差异包含质量排序、RLOO与关闭二值组过滤。各组为单seed、重复固定32初态，检查点均值是描述性结果，不作独立样本显著性判断。

## 包含内容与恢复边界

`raw/` 保存两组已冻结runtime日志、命令、实际配置、TensorBoard事件、全量scalars JSON/CSV、关键源码与source-lock，以及包内文件逐项SHA256清单。`plot.py`是可复现绘图入口，八张PNG、网页与本说明可离线查看。停止/归档/发布回执按实际完成状态附入。

按本轮指定 **每组只保留最后完整checkpoint**，不另保留best。原位绝对路径及逐文件大小/身份以RETENTION_MANIFEST.json为准；日志最终轮次可能晚于最后周期保存轮次。checkpoint大张量、完整成功池RGB及视频不放入ZIP，ZIP是轻量分析和源码记录，不能单独恢复训练。任何服务器删除以主代理单独的授权执行回执为准，本地图册不执行服务器动作。


## 本轮清理结果

按latest-only授权分两批清理，共71个历史模型/优化器大文件、650.94 GiB，范围包含AttenA与Prism的中间代，并非这两组单独释放650.94 GiB。两组各保留R80最后完整checkpoint，成功数据与小状态保留。本轮数量不含此前52文件清理。21:16磁盘剩余：/home 1057.94 GiB、/data 1072.77 GiB。详见 [清理汇总](CLEANUP_SUMMARY.json)；raw中的全量checkpoint库存是清理前记录，当前保留以RETENTION_MANIFEST.json为准。
