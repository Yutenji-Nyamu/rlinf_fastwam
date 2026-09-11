# pi05-grpo-dvac-new-256 收尾分析包

2026-09-11收尾。按用户授权停止，最后完整训练R109，原计划200轮；不是自然跑满。 本包对应运行已结束，后续新训练不混入曲线。

每轮256条/G8/U2、1024/micro32、LR5e-6；旧指令环境；DVAC new两级group、both、alpha1/1，保留原chunk PPO裁剪。

采集MA5 / MA10：60.47% / 60.82%；最新fixed R105：68.75%。

[10张独立单列对照图](comparison/index.html)：每轮、MA5/10、fixed、累计采集；颜色、线型和marker共同区分。BC历史预算参照有环境/过滤差异；旧GRPO/clean/Prism使用同一旧环境。

raw/runtime含日志、配置、命令、合同和退出记录；raw/tensorboard为原始events；scalars.json/csv为全量指标；source与source-lock固定已校验源码。checkpoint-inventory来自停止前冻结盘点，仍列出随后按授权清理的中间文件，不是当前文件清单。

服务器保留最后完整checkpoint R100及相关成功池/小状态；大模型、optimizer、数据池、视频不装入ZIP。2026-09-11 11:56已按授权清理这三组非最终大checkpoint，最后R100组件通过保留检查；清理回执位于项目`docs/server-admin/evidence/storage-structure-20260911/execute.json`。本ZIP是轻量分析包，不能单独恢复训练。

输出：`/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/pi05-grpo-dvac-adv-new-formal200-phys45-20260909-v1`；封存时源码HEAD：`59c05e68a98178880e249d057125a8b82d75d515`；分支：`codex/sz-pi05-grpo-dvac-adv-new-20260908`。

BC全部17条运行复盘另见项目入口：`docs/rlinf-robotwin-pi0-online-bc/BC_COMPREHENSIVE_DECISION_20260911.md`。本包不包含其他用户进程或私有文件。
