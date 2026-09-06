# chenyiteng checkpoint 最小清理流水账（2026-09-01）

## 授权与边界

- 用户授权：GRPO 与 PPO 的有效正式实验只保留各自最新完整 checkpoint，删除所有更早 checkpoint 的大文件；π0.5 与 Fast-WAM 的 smoke checkpoint 清理大文件。
- 最小删除口径：每个目标 checkpoint 仅删除按实际字节数排序最大的 1--2 个训练文件，保留目录、日志、manifest、metadata 与小型 sidecar。
- 明确保留：当前活跃 π0.5 formal 两个运行目录；每条有效 GRPO/PPO lineage 的最新完整恢复点；RLT、DSRL、其他用户及所有非 checkpoint 数据。
- 本轮不递归删除目录，不使用通配符删除，不删除空故障目录；每个目标使用复核后的绝对文件路径。

## 执行前验证

- 2026-09-01 14:56--14:58 CST 只读盘点确认 checkpoint 是 `results` 的主要占用。
- π0.5 当前两条 formal 均在运行，且执行前盘点时尚无 `global_step_*`；其目录与 smoke 目标零匹配。
- 待删除文件清单必须满足：位于 `/data/chenyiteng/results/rlinf-shenzhen/` 内、所属 checkpoint 不是保留点、文件真实存在、实际大小与预览一致。

## 精确执行清单

### π0.5 / Fast-WAM smoke

- 9 个 smoke checkpoint，每个只删 Top-2 大文件，共 18 files。
- 逻辑字节：`225,755,207,998` bytes。
- π0.5 四个 local-shard smoke 删除两份 rank checkpoint，保留约 8.53 GB 的 `full_weights.pt`、目录和小文件。
- Fast-WAM 五个 smoke 删除两份 local-shard 或 DCP rank shard，保留目录、`.metadata` 与 DVAC sidecar。
- 当前两个 π0.5 formal 根在目标清单中匹配数为 0。

### GRPO / PPO

- 14 条有效 formal lineage 的最新点完整保留；其 82 个关键文件在删除前后逐一验证存在。
- 64 个淘汰 checkpoint 根各删 Top-2 大文件，共 128 files：
  - GRPO：98 files / `657,622,603,387` bytes；
  - PPO：30 files / `245,808,384,698` bytes。
- 合计逻辑字节：`903,430,988,085` bytes。
- 误配置 `32x8/B512` 的 Step10 是该 run 唯一点；遵循“可以少清”的谨慎口径，改列
  `KEEP_CONSERVATIVE_ONLY_POINT`，本轮未删。
- 两个 12 KiB / 0-file 的失败目录无大文件，本轮未删。

## 执行与复核终态

- 总计精确删除 `146` 个大文件，逻辑大小 `1,129,186,196,083` bytes（约 1.129 TB）。
- `/data` 使用率从 `75%` 降至 `44%`；最终现场约 `2,031,780,605,952` bytes 可用。
- 删除后第二条独立 SSH 连接确认：
  - π0.5/Fast-WAM 18/18 目标不存在；
  - GRPO/PPO 128/128 目标不存在；
  - 14 个最新 checkpoint 的 82/82 个关键文件仍存在。
- 当前 π0.5 Control / DVAC formal wrapper、driver、Ray actor 与 GPU 4--7 均仍存在，近期日志/视频继续更新，fatal scan为空。
- 所有被处理的旧 checkpoint 目录仍保留，但因删除训练大分片而有意变成不可恢复的历史骨架；日志、TensorBoard、CSV、manifest、metadata和小型sidecar未删。
- GRPO/PPO脚本首次经保留CRLF的传输方式执行时，在任何preflight或删除前语法退出，删除数为0；随后以既有helper只在传输时规范化换行，执行同一SHA256脚本成功，没有改变目标或删除命令。

## 审计材料

- GRPO/PPO执行脚本SHA256：`37F536BAC79F89B413000170A9A6E007E09E0E7450135EB0F69709A8BB79F15B`。
- GRPO/PPO最终manifest SHA256：`CB7E3842F0D1DFCDDBA4AC1F79990996F0779450F0890B0ABD797CD6BF95F812`。
- π0.5/Fast-WAM执行脚本SHA256：`33BB8BD41936912E6F881E074659FB351BBAFC97F3054B2CAFFD6059F9261BEC`。
- 执行stdout与独立复核stdout保存在本地工作区，未复制checkpoint正文。
