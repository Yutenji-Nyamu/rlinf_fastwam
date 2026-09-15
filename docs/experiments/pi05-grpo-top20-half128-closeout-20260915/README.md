# π0.5 GRPO128 · 80/20 top20 收尾

用户主动中止并切换success-only τ1。最终已记录采集R96，原计划200轮；最新固定评估R95：12/32。最近checkpoint目录R90；目录清单并非模型加载/完整恢复验证。runtime退出码为134；终止原因记录为`user_requested_stop_for_success_only_tau1_replacement`，原始结束时间2026-09-15T05:52:08+00:00。

正式200轮、64并行×2串行=128条/轮，G8、U2、B512/micro32、LR5e-6；原模型fresh、rollout seed42+rank及评估RNG隔离，每5轮fixed32、每10轮保存。top20 selection_domain=batch，按原始V选最高20%位置，selected_l=3；普通更新系数0/5，p=0.1整次Adam全量旁路，method_seed=20260911。原chunk概率比与PPO裁剪保持，旧DVAC weighting关闭。

采集与fixed32分开；只含本组fresh R1起历史，不混入smoke或其他方法。 最后一轮采集与尾部优化标量可能不同，逐tag截止已写入SUMMARY.json，不将部分尾部更新冒充完整训练轮。

[五张单列宽图](plots/index.html) · [完整标量CSV](metrics.csv) · [原始日志](raw/runtime/driver.log) · [实际配置](raw/tensorboard/config.yaml)

ZIP包含完整driver日志、原始TensorBoard events、全部标量CSV/JSON、运行命令和实配、源码版本与文件SHA、checkpoint文件目录及本说明。模型、optimizer、回放及环境图像/视频不入包，仍保留在服务器。本次未读取模型或池payload、未删除checkpoint，所有末代文件保留；此包不用于独立恢复训练。

服务器run：`/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/pi05-grpo-top20-half128-seed42-formal200-phys45-20260914-v1`

生产源码：`06d4e1c9bd8e2d89f9b6dcefb9d8c2bca16d8082`；归档前HEAD：`a2bf6c7667faa1e27ad3fc161eb6cfad37c377a4`；分支：`codex/sz-pi05-grpo-top20-half-seed42-20260914`。
