# BC＋DVAC讨论、服务器刷新与轻量Git核查账本

本轮用户授权：刷新本人各实验及整机；深入讨论BC＋DVAC，不实施DVAC或启动GPU7；核对并推送应备份的小文件。保持GPU6正式、Sidney、shared Ray、其他用户任务原样；大模型/checkpoint/数据/视频留服务器，未审阅dirty不混入算法提交。

1. 完整恢复根规则、交接、窗口路由及BC唯一SSOT；定向读DVAC设计与09-03/09-05 Git覆盖记录。旧快照只用于定位，不作当前结论。
2. 新建本轮固定host-key/password-only读取工具，复用既有服务器只读审计与Git清单脚本，仅更新BC正式目录、Sidney resume运行目录及tracked docs/evidence路径。先身份探针，再读GPU/RAM/CPU/磁盘/服务/ECC、日志/TB/最新checkpoint与全部本人工作树/分支远端；不加载模型、不做GPU推理、不改运行配置。
3. 16:43 inventory完成：23工作树/22codex分支，21远端匹配；当前运行快照与Git覆盖分别保存JSON。16:47定向核对BC4/GRPO2/RLT1源码SHA256均与本地一致；RLT诊断dirty两文件35行、OIDN未跟踪脚本只读。
4. 重新查看DVAC原论文、OpenPI原生FM和LeRobot RA-BC官方接口；完善01_DVAC_DESIGN §6—9，明确alpha0.25/高V/过去5轮/入池固定w为建议，未实施或启动DVAC。区分GRPO/RLT信号复用与训练目标，保留现BC随机调用顺序。
5. 16:55按预先记录的精确目录和HEAD执行一次轻量补推：PPO c41b7ff0、Fast 0d5daf6f、RLT d3acd650，三者远端HEAD匹配/clean；92文件538720bytes。诊断branch f3ea5f69推送成功，dirty patch前后完全一致；原OIDN脚本未动。详细命令和输出保存在local_scripts/sz_light_backfill_followup_20260905.py及server-admin/SHENZHEN_LIGHT_BACKFILL_FOLLOWUP_20260905.txt。没有修改算法或重放补推脚本。
6. 17:01只读refresh：Sidney完整110，fixed17/32，ckpt110双rank/full在；BC完整2轮，49成功episode、尚无正式评估/ckpt；Fast未重启。GPU/RAM无当前容量报错；/data460.4GiB，现有两run余下ckpt约414GiB，GPU7开跑前需存储计划。
7. 生成π0.5全程分栏图：最初artifact脚本发现本机无matplotlib，未安装，改用已有Pillow；已运行成功并view_image确认无遮挡，PNG/HTML/CSV/summary在π0.5专题pi05-bc-review-20260905。原始TB step+1，仅已有点，不补造Step0。此为本机文档图表生成，不是项目测试。
8. 17:08—17:09发布完成：BC讨论/现场/启动快照提交385d4e75（5文件，payload20844bytes），Sidney既有完成100轮ZIP＋当前110轮图/CSV/HTML提交81be3193（7文件，payload559497bytes）。仅新建evidence/bc_dvac_review_20260905，先校验源HEAD/clean、敏感串/ZIP内容及大小，commit/push后远端HEAD匹配、clean、全部diff限evidence；生产源码/config零改动。原运行source-head继续有效，不能把新的证据HEAD理解为热更新训练。
9. 最后再次ls-remote核对RLinf仓19个codex分支全部与local一致，含原缺失诊断branch；RoboTwin仓3个分支在16:43已匹配且本轮未改，总22个committed tip均有远端。输出在server-admin/SHENZHEN_BC_DVAC_REVIEW_PUBLISH_20260905.txt。原有dirty未合并但已快照备份；活动运行终稿和部分旧目录布局仍不等于完全覆盖，未声称所有产物100%推送。
10. 已更新HANDOFF/BC唯一SSOT的最新状态、磁盘风险、DVAC待确认语义及Git结果。未新增长期监控、未跑额外训练测试、未动shared Ray/其他用户进程。
