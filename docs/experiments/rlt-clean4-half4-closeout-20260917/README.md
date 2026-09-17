# π0 RLT · Clean4 · 完整800轮 收尾

累计800轮正常完成。采集末轮R800，固定评估末次R800：20/20；末检查点R800。退出码0，结束时间2026-09-17T11:00:33+00:00。

主线使用旧R1–525与续训R526–800；旧段后续原始事件保留但不重复计入。

[五张图](plots/index.html)：逐轮采集、MA10、MA20、MA50和固定评估。完整指标在metrics.csv/metrics.json；每段原始TensorBoard events、日志、实际配置及命令均在segments/。

源码提交8b198b113a596186b0d22bb525c4bb6319c63cdb；收尾前分支HEAD 8b198b113a596186b0d22bb525c4bb6319c63cdb，分支codex/sz-rlt-clean4-resume800-20260917。source-file-ledger.txt是归档HEAD的Git blob源码清单；生产时源码清单保留在runtime/contract.json。末检查点仅收文件大小和完成标记，权重、optimizer、经验池、视频均不打包，服务器文件保留。本包用于分析，不用于独立恢复训练。

run：`/data/chenyiteng/results/rlinf-rlt/current-single-gpu-clean-half4env-resume525to800-phys6-20260917-v1`
