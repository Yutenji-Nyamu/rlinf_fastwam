# π0 RLT · DVAC τ2 · 成功倍率1 收尾

按用户要求停止。采集末轮R695，固定评估末次R675：19/20；末检查点R675。退出码15，结束时间2026-09-17T11:30:13+00:00。

从既有Stage1开始，Stage2 fresh，每轮4采样。

[五张图](plots/index.html)：逐轮采集、MA10、MA20、MA50和固定评估。完整指标在metrics.csv/metrics.json；每段原始TensorBoard events、日志、实际配置及命令均在segments/。

源码提交703408b1f525645b58451e90404973a75cf0e41a；收尾前分支HEAD 606fbc3e7fce9d4cf28e44075d5ab93029452880，分支codex/sz-rlt-t2-resume-t15-half4-800-20260916。source-file-ledger.txt是归档HEAD的Git blob源码清单；生产时源码清单保留在runtime/contract.json。末检查点仅收文件大小和完成标记，权重、optimizer、经验池、视频均不打包，服务器文件保留。本包用于分析，不用于独立恢复训练。

run：`/data/chenyiteng/results/rlinf-rlt/current-single-gpu-dvac-exp-t2-half4env-fresh800-phys7-20260916-v1`
