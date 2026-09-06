# 深圳Git补推结果：BC U10启动后

审计时间16:43、补推16:55/17:09；本人23工作树/22codex分支。没有操作其他账号、shared Ray或训练进程；没有删除大产物。

| 分支 | 新证据提交 | 内容 |
|---|---|---|
| codex/sz-ppo-pi0-robotwin | c41b7ff0 | 3个旧PPO/重载/fixed评估目录的19个轻量文件 |
| codex/sz-fastwam-current-rlinf-grpo | 0d5daf6f | scene-fence/OIDN诊断36文件，含原未跟踪脚本快照 |
| codex/sz-rlt-pi0-robotwin-ar | d3acd650 | checkpoint诊断37文件，含原dirty patch快照，不应用到源码 |
| codex/sz-rlt-checkpoint-diagnosis | f3ea5f69（已有本地HEAD，补远端ref） | 原dirty前后完全相同；推的是已提交基线，不宣称dirty已验证 |
| codex/sz-pi0-online-bc | 385d4e75 | 本轮DVAC讨论、现场、正式启动快照，5文件 |
| codex/sz-sidney-pi05-current-rlinf | 81be3193 | 已有完成100轮轻量ZIP＋续训110轮图/CSV/HTML，7文件 |

新增5个evidence-only commit均push后远端匹配/树clean；共104个新文件（含README/manifest），不是104个新增实验。RLinf仓19分支最终再次ls-remote全部匹配；RoboTwin3分支16:43匹配、本轮未改。22个committed tip现在均有远端。

生产实现和resolved不变，活动任务启动时source-head仍有效；仅Git证据HEAD前进。诊断树dirty及OIDN未跟踪脚本仍留原位，已归档快照但不混入生产。

覆盖边界：初始已track1463个轻量evidence，约41.11MiB。候选94目录里67直接marker匹配，27未直接匹配含另类证据布局/预备目录/早期探针，不能把它们全部当漏备份，也不能据分支已推宣布全部历史产物100%完备。当前活动BC/π0.5最终日志尚未产生；模型、checkpoint、成功池图像、视频/完整Ray日志仍在服务器，不进Git。

原始证据：[清单](SHENZHEN_GIT_COVERAGE_AFTER_BC_U10_20260905.json)、[历史补推](SHENZHEN_LIGHT_BACKFILL_FOLLOWUP_20260905.txt)、[本轮发布和最终refs](SHENZHEN_BC_DVAC_REVIEW_PUBLISH_20260905.txt)。
