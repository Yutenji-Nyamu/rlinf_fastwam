# 两组 BC DVAC new 最终归档 · 2026-09-09

两组均已关闭，11:39 CST GPU6/7无计算进程；共享Ray其他actor和GPU4/5 Prism保护进程保持。

| 实验 | 最终完整记录 | 最近fixed32 | 采集MA5 / MA10 | 状态 |
|---|---:|---:|---:|---|
| DVAC new 4/U5＋长度≤3 | R100 / 400次训练采集 | R100：17/32 | 75% / 72.5% | 08:53:41自然结束，exit0 |
| DVAC new 8/U10＋长度≤3 | R72 / 576次训练采集 | R70：18/32 | 60% / 65% | 11:39:14按授权停止，wrapper记录exit134 |

停止操作只向新8的已核身份driver发送SIGTERM，driver自身清理独立namespace；两组目标driver/actor均已退出。exit134为停止时最终记录，不能写成正常完成；不计停止时未完整落盘的轮次。详见 [唯一停止回执](stop-receipt.json)。

## 对比结论与图册

[4/U5三组8张宽图：raw、MA5、MA10、fixed32，轮次/采样双横轴](comparison4/index.html) · [详细统计](comparison4/README.md)

共同fixed R5–90：干净BC **42.88%**、旧DVAC **49.48%**、new **48.78%**。new相对旧版−0.69个百分点（7胜2平9负），尾5点两版均50%；本轮不能得出新版优于旧版。new采集成功更多，但固定评估并未超过旧版。

[8/U10最终4张对照图](comparison8/index.html) · [共同fixed统计](comparison8/README.md)

8/U10共同fixed R5–70：干净BC62.95%、new49.78%，差值−13.17个百分点（1胜2平11负）；最后5个共同点差值−15.63个百分点。这组组合方法没有显示收益。

新方法同时启用长度≤3过滤，历史clean/旧DVAC未过滤；差异是组合效果。全部为单seed和重复固定32任务，不把检查点当成独立实验重复。

## 包里有什么

- `raw/runs/{4u5,8u10}/raw/`：完整driver/wrapper日志、运行命令、实际/基线配置、逐叶diff、TensorBoard events。
- `raw/runs/.../scalars.json/csv`：所有训练/采集/评估/池/权重标量；round=原始TB step+1。
- `raw/runs/.../source/`、`source-lock.json`：关键方法源码、配置和哈希；训练版本均为e185219a9856521fcc705a96c9fd4d5eb5252b37。
- `comparison4/`、`comparison8/`：独立PNG/SVG、交互页面、CSV/JSON和绘图源码。
- `RETENTION_MANIFEST.json`：最新/最好已保存checkpoint的最低保留清单，全部原始checkpoint及成功池文件仍在服务器，未删除。

4/U5保存至R100，8/U10保存至R70；R71–72的日志存在，但没有新的周期checkpoint。ZIP不含大模型/优化器张量、RGB成功池或视频，因此是轻量分析归档，不是完整恢复备份。原文件位置和大小见每组checkpoint-inventory.json。

## 复核

原始服务器ZIP已校验SHA256、全部文件清单哈希及ZIP CRC。首次归档因一个非必需runner路径不存在而停止；第二次在独立新目录重新完成，无训练文件修改、无重复停止。所有图从已冻结日志生成，无平滑补点、无追加评估。
