# DVAC new BC 4/U5续训收尾

用户要求停止，最终完成R174，最后完整权重R170留在服务器。原模型、成功池均保留。

[pi05-bc-new-4u5-R100to174-closeout-20260912.zip](pi05-bc-new-4u5-R100to174-closeout-20260912.zip) 包含完整续训日志、events、原始指标、配置与停止回执，以及合并R1–174指标、逐轮/MA5/MA10/固定评估宽图；大权重不在ZIP内。

ZIP SHA256：`271d9009a62fdfe79f2a94f2d587569cd81f90389f32ff06d67eda6630e746f5`。

最近固定评估R170：18/32。用户中止后wrapper记录exit134，不能标记为自然完成或exit0。模型源码未改。

比较时保留原差异：clean长度过滤off、new≤3；R100恢复未保存rollout随机状态及环境游标。两条图中的R100线标记续训边界。

[四张图](plots/index.html) · [完整指标](metrics-merged.csv) · [checkpoint目录清单](checkpoint-inventory.json)。原始日志在ZIP的raw/runtime/driver.log。
