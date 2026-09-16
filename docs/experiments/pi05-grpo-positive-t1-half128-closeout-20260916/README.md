# π0.5 GRPO128 · 指数仅成功 τ1 收尾

用户主动中止并切换linear仅正优势内alpha0.8外alpha0.5。最终已记录采集R115，原计划200轮；最新固定评估R115：15/32。最近checkpoint目录R110；目录清单并非模型加载/完整恢复验证。runtime退出码为134；终止原因记录为`user_requested_stop_for_linear_positive_alpha08_alpha05_replacement`，原始结束时间2026-09-16T05:21:24+00:00。

正式200轮、64并行×2串行=128条/轮，G8、U2、B512/micro32、LR5e-6；原模型fresh、rollout seed42+rank及评估RNG隔离，每5轮fixed32、每10轮保存。DVAC two_level_group / exp_mean / scope=positive，两层tau1、alpha1，selected_l=3；失败保持原GRPO权重1，原chunk概率比与PPO裁剪保持。

采集与fixed32分开；只含本组fresh R1起历史，不混入smoke或其他方法。 最后一轮采集与尾部优化标量可能不同，逐tag截止已写入SUMMARY.json，不将部分尾部更新冒充完整训练轮。

[五张单列宽图](plots/index.html) · [完整标量CSV](metrics.csv) · [原始日志](raw/runtime/driver.log) · [实际配置](raw/tensorboard/config.yaml)

ZIP包含完整driver日志、原始TensorBoard events、全部标量CSV/JSON、运行命令和实配、源码版本与文件SHA、checkpoint文件目录及本说明。模型、optimizer、回放及环境图像/视频不入包，仍保留在服务器。本次未读取模型或池payload、未删除checkpoint，所有末代文件保留；此包不用于独立恢复训练。

服务器run：`/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/pi05-grpo-dvac-positive-t1-half128-seed42-formal200-phys45-20260915-v1`

生产源码：`1732f1ef3e4a993657ac54dc052e6f0d90d2afc3`；归档前HEAD：`2eb6d34574121a0bd7bc3c844c24d2f6b38fe436`；分支：`codex/sz-pi05-grpo-dvac-positive-t1-20260915`。
