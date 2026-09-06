# 2026-08-23 深圳整机管理员巡检、GRPO刷新与DVAC教学取证流水

范围：只读服务器巡检与本地文档整理。普通运行状态使用`chenyiteng`，系统/全用户元数据使用`toom`；
密码只注入当前进程，未写入命令文件、输出、文档或Git。没有sudo写入、安装、下载、进程控制、删除、
覆盖或训练配置修改。

## AUD-001 — 全量管理员只读巡检

- 时间：2026-08-23 10:46 CST。
- command file：`local_scripts/remote_commands/shenzhen_admin_whole_server_audit_20260823.sh`；
  SHA-256 `ef02852e9907b458daea6c6a71703eec20163f7a81393d7b6fa71f0be743e317`。
- 调用：`remote_exec_autodl.py --host 120.241.223.9 --port 22 --user toom --host-key-sha256
  <fixed-fingerprint> run --command-file <above>`。
- 结果：exit0，marker `SZ_ADMIN_WHOLE_SERVER_AUDIT_OK`。确认Ubuntu 22.04.5、128 CPU、2 TiB RAM；
  `/ /home /data`容量与inode宽裕；当前用户会话、进程、GPU、系统服务、auth/journal均可读。
- 首轮长输出中的中段资源表过长，因此下一条只做同一字段的定点复核；没有据截断输出下结论。

## AUD-002 — host-key参数格式纠正

- 第一次GRPO状态调用把helper参数误写成`SHA256:<fingerprint>`；helper自身会添加`SHA256:`，因此在认证前
  本地拒绝并显示`expected SHA256:SHA256:...`。远端命令没有执行。
- 实际返回fingerprint与已锁值逐字符一致；只删除参数中的冗余前缀后重试，没有更换主机key、账号或认证路线。

## AUD-003 — 日常账号刷新GRPO v2

- 时间：2026-08-23 10:50 CST。
- command file：`local_scripts/remote_commands/shenzhen_grpo_v2_final_status_20260823.sh`；
  SHA-256 `aa461b4d20d4aa1053afb78438e7816077164096f049be65e230313b71fbcfe5`。
- 结果：exit0，marker `SZ_GRPO_V2_FINAL_STATUS_OK`。
- driver/observer alive；4 actor + 4 rollout + 4 env + 1 GCS + 1 raylet。完整到Step29并进入下一步rollout。
- Step29：512 trajectories，success=`0.921875`、KL=`0.016`、clip=`0.066`、grad=`12.970`、
  step time=`1390.599s`；fatal0、nonfinite0。
- GPU4--7约65--69 GiB/卡；cgroup memory=`1667795156992` bytes=`1553.3 GiB`，swap=`5.95 GiB`，
  host available=`466.1 GiB`，memory events的`high/max/oom/oom_kill`均0；checkpoint count2、eval MP4 count8。

## AUD-004 — 管理员定点资源/用户/系统错误复核

- 时间：2026-08-23 10:51 CST。
- command file：`local_scripts/remote_commands/shenzhen_admin_targeted_refresh_20260823.sh`；
  SHA-256 `e7d9345aec90c1e8ede8106f26d2c91c008a7f8cb995fffd023bad07992a1598`。
- 结果：exit0，marker `SZ_ADMIN_TARGETED_REFRESH_OK`。
- CPU约96% idle；3个`vmstat`样本`si=so=0`。`/`49/296G、`/home`94G used、`/data`287G used；
  inode最高2%。GPU0--3无compute，GPU4--7所有compute PID均属于`chenyiteng` GRPO。
- 四个EnvWorker RSS约316/373/377/396 GiB，是主存高占用主体。其他用户只有低负载tmux/Codex/watchdog；
  没有其他GPU作业。
- SSH、Mihomo、Docker、containerd均active，failed unit0；端口22、7890、Ray dashboard8265符合当前服务。
- 自8月20日起：OOM=0，I/O/EXT4/NVMe错误=0；Xid6条只在8月21日，segfault1条只在8月22日04:29 UTC。

## AUD-005 — SSH来源与segfault归因复核

- 时间：2026-08-23 10:53 CST。
- command file：`local_scripts/remote_commands/shenzhen_admin_auth_and_segfault_context_20260823.sh`；
  SHA-256 `58b9d2546c2388114a98c53f6edff3bc89436834d0dc08c8713dcf40c3740a99`。
- 结果：exit0，marker `SZ_ADMIN_AUTH_SEGFAULT_CONTEXT_OK`。
- 保留auth日志中的Accepted只涉及既有五账号；公网失败/invalid扫描量大，top失败来源6,685条，未见未知用户名
  成功。没有读取shell history、私有项目正文或凭据。
- 04:29:45 UTC segfault与`chenyiteng` session 699结束同秒；本地既有Fast-WAM ledger的
  `fw-sz-400-20260822_042858`记录精确对应MPLib `Box(...)`在NumPy2组合下的segfault。随后NumPy窄降与
  official evaluator闭环已记录，因此不是当前新故障。

## AUD-006 — 本地DVAC结果与图复核

- 完整读取13/15/16号计划/结果、π0/Fast-WAM实现账与最终`analysis_summary.json`、
  `outcome_summary.csv`、`L_sensitivity.csv`。
- 直接对最终CSV复计：759 query rows、717 baseline-eligible/pre-success rows、70,592 horizon rows；
  policy/task分组为π0 256，Fast-WAM 80/162/133/128。
- 目视核对跨任务总览、π0 raw/residual三联图、move-stapler phase timeline及turn-switch成功/失败storyboard。
- 形成上级教学文档`18_SERVER_HEALTH_GRPO_AND_DVAC_TEACHING_20260823.md`。没有重算或覆盖source payload。
