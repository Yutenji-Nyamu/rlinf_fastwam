# RLT 单卡 matched-width 双实验：实施流水账

日期：2026-08-26

目标：精确停止并保留旧单卡 control / success-episode DVAC-BC 正式训练；用相同方法定义 fresh 启动两条新 480-step 训练。相对旧单卡版只调整 `micro_batch_size=256`、`warmup_min_size=20000`、replay `cache/sample_window=80000` 以及新运行路径。

## 流水

1. 本机只读复核 `PROJECT_CONTEXT.md`、`HANDOFF.md`、41/42 号专题说明、旧 v2 resolved YAML 与启动脚本。结果：旧两条配置为 `micro=128 / warmup=10000 / replay=50000`；方法版为 `success_episode_bc / strength=0.25`，本轮不改方法。
2. 12:11 CST 服务器只读探针成功：身份为目标 AutoDL root；旧 control/method wrapper=`586602/586603`、PGID 与 PID 一致且均存活，训练 Python 位于各自进程组；shared Ray=`585779`，monitor/cleanup=`586608/586611`；source=`64f2779f...` clean；GPU0/1=`25264/18516 MiB`，cgroup memory=`169339535360` bytes，OOM/OOM-kill=`0/0`。
3. 12:12--12:13 CST 对两个已核验 PGID 发送 INT，必要时只结束同组进程；随后停止该 pair 独占的 monitor/cleanup/shared-Ray。旧训练进程均退出，GPU0/1归零，OOM/OOM-kill仍为`0/0`；旧产物原样保留，control/method约`1.6/1.8 GiB`，两边最新完整checkpoint均含Step125。
4. 本机新增两个薄 overlay YAML；分别继承旧control/method叶，只共同覆盖`micro=256 / warmup=20000 / replay=80000`及新实验名。旧v2 YAML与方法字段不改。
5. 服务器compose两份resolved成功：共同必需字段错误数均为0；control/method共23个差异，全部属于placement、run路径及15个方法字段。对历史成功双卡resolved共14个差异，正好是已批准的4个训练字段、placement、fresh入口及路径；seed文件虽换worktree路径，SHA256逐项一致。首次commit前`git diff --check`只报两个新YAML末尾多一个空行，已做纯格式窄修复后重传。
6. 服务器commit成功为`848b6127...`；首次向`origin` push返回GitHub 403（该remote指向上游`RLinf/RLinf`，当前账号无写权限）。提交本身完整且worktree待核；下一步使用该专题既有fork remote，不改代码。
7. 改向既有`personal` fork push成功：`64f2779f..848b6127`，remote tracking恢复，worktree clean。
8. 12:22 CST fresh启动新pair：shared Ray=`172.17.0.9:52001`，control wrapper=`243563`/GPU0，method wrapper=`243564`/GPU1，沿用120秒错峰、只读5秒资源记录与结束清理。
9. 12:27 CST健康检查：两边compose=`0`，resolved现场均为`micro=256 / max_steps=480 / warmup=20000 / replay=80000`；method另为`apply / success_episode_bc / strength=.25`。placement分别`[[0]]/[[1]]`，80k replay已建立，两边均进入首个`Generating Rollout Epochs: 0/1`。wrapper存活，cgroup约75.0 GB，GPU0/1约21.2/10.8 GiB，OOM/OOM-kill=`0/0`。Curobo可选导入告警与旧成功运行相同，本配置实际使用MPLib，不是启动失败。
