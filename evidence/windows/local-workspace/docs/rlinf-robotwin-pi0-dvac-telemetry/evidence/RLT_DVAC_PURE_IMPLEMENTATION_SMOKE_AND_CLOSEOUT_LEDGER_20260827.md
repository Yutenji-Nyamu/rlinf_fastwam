# RLT-DVAC-Pure 实现、smoke 与旧实验收尾流水

日期：2026-08-27

## 约定

- 方法：保留 frozen π0 reference BC target；仅成功 episode 的 student C10 使用 detached DVAC 权重。
- 映射：`z -> r=z-mean_h(z) -> raw=max(0,1+0.5r) -> w=raw/mean_h(raw)`，逐 query mean-one。
- 不改 actor-Q、critic TD、replay、调度、环境并发或单卡正式配置。
- 本账本从第一次服务器操作开始逐项记录命令、结果和窄修复。

## 操作流水

### L001 本机上下文与连接工具复核

- 指令：读取 `PROJECT_CONTEXT.md`、`HANDOFF.md`、密码提示包装器与 Paramiko helper；检索 matched-width 正式运行路径及 Pure 计划。
- 结果：确认固定 host-key 的低层 Paramiko 路线仍可用；两条旧正式运行路径、现有 DVAC worktree 与最小实现切口均已定位。
- 问题：46号计划仍保留上一轮临时讨论的“非 mean-one 直接 [0,2]”文字。
- 处理：按用户最终决定，在实现批次中统一改回 mean-one、`strength=.5`。

### S001 AutoDL 只读身份与旧正式训练现场

- 指令：通过固定 host-key 的 Paramiko command-file 路线执行 `tmp/autodl_rlt_matched_width_live_refresh_20260827.sh`。
- 身份：`autodl-container-nekaqbwt43-6ce5babb`，`/root`，UID 0，时间 `2026-08-27T22:22:41+08:00`。
- 结果：control 与旧 success-executed DVAC-BC 两条 fresh-480 均仍在运行；两边均已保存到 `global_step_450`，driver/wrapper存活。
- 资源：GPU0约18.0 GiB、GPU1约24.6 GiB；cgroup约233.5 GiB/240 GiB，`oom=0`、`oom_kill=0`；五类fatal关键词均为0。
- 决策：不抢占GPU、不停止旧训练；先在本机完成窄实现，待两条自然结束和资源释放后再做服务器检查与smoke。

### S002 服务器代码基线确认

- 指令：只读查询 `/root/autodl-tmp/RLinf_rlt_dvac_success_bc` 的分支、HEAD、dirty状态与四个目标文件SHA256。
- 结果：worktree clean，分支=`codex/rlt-dvac-success-episode-bc`，HEAD=`848b61278687702ea717c56b3734f1486cea3b95`，远端同分支指向一致；matched-width control YAML与本地副本SHA256一致。
- 决策：代码在该既有专题分支做增量；配置从matched-width单卡control继承，不另建重复的RLT实现分支。

### L002 Pure mean-one 窄实现

- 修改 `rlinf/algorithms/rlt/dvac_weighting.py`：centered权重改为非负截断后逐query归一到均值1；BC builder新增兼容默认`success_target=executed`，并支持`reference`。
- 修改 `rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py`：读取/校验/传递target选择；移除旧`.25`上限，Q路径不变；增加reference-target模式标量。
- 修改目标单测：增加`.5`截断+mean-one、reference target、非法target三项覆盖，旧executed测试保留。
- 新增Pure正式YAML：从matched-width单卡control继承，仅增加`success_episode_bc + reference + L3 + strength=.5`及必要输出名。
- 文档：46号短计划与索引/交接改回用户最终确认的mean-one语义。

### S003 隔离worktree与代码同步

- 指令：从clean `848b6127`创建 `/root/autodl-tmp/RLinf_rlt_dvac_pure`，分支
  `codex/rlt-dvac-pure-reference-bc`；将四个精确目标文件组成119,808-byte tar，经SFTP传到`/tmp`后只解入该新worktree。
- 结果：新worktree HEAD仍为`848b6127`；dirty仅为3个预期修改文件和1个新Pure YAML；`git diff --check`通过，tracked stat=`+84/-11`。
- 说明：正在运行的旧正式实验仍读取原 `/root/autodl-tmp/RLinf_rlt_dvac_success_bc`，没有被改动。

### L003 静态复核后的两处窄修正

- 修正`.5`截断测试样例：使用`[-2,2,2,2]`，真实覆盖负raw权重被截到0并重新mean-one。
- 收紧新映射的理论诊断上界为clamp前正权最大值；避免使用无信息的C10上界10。
- shared actor-loss也服务于DVAC关闭的async RLT；target参数改为缺省`executed`的`getattr`，避免async旧路径因没有新属性而改变行为。
- Pure测试补充反向断言，直接确认reference-BC的每个$h$梯度按对应权重缩放。

### S004 用户授权后精确停止旧 matched-width 实验

- 23:05现场：control/method均已保存`global_step_475`；wrapper/driver仍存活，fatal与OOM关键词均为0；GPU约25.2/15.4 GiB，cgroup约239.5/240 GiB，`oom=0`、`oom_kill=0`。
- 用户明确认为证据已足够并授权停止、整理。指令：读取并核验两个wrapper PID/PGID及其精确启动命令，只向PGID `243563`、`243564`发送`SIGINT`；随后停止本pair的monitor、cleanup与独立Ray head `242745`。未触碰其他进程，未删除运行目录或checkpoint。
- 23:08复核：两个wrapper与独立Ray head均退出；GPU0/1均为0 MiB，cgroup降至约56.6 GiB；`oom=0`、`oom_kill=0`。运行目录完整保留，等待轻量封存。
- 问题：wrapper在收到process-group INT时随driver一起退出，没有机会写自然完成用的`exit_code.txt/finished_at.txt`。
- 处理：收尾包显式记作`user-stopped`，以metrics中的最后完整step和`global_step_475`完整checkpoint为终点，不伪写`480/480`或`exit0`。

### S005 前置检查、提交与推送

- 指令：对3个Python目标文件执行`ruff format/check`、`py_compile`，运行定向单测，compose control/Pure并逐叶比较resolved config。
- 结果：格式未产生额外修改，lint/compile通过，`12 passed`；Pure相对control只差`algorithm.rlt_dvac.*`、rollout telemetry mode和实验名，其他正式参数一致。
- Git：提交`cb88e9c5d817a248fef6d6ee02127b874ab851a8`，4 files `+115/-11`。
- 问题：第一次普通HTTPS push遇到`GnuTLS recv error (-110)`。
- 处理：按AutoDL既有网络路线，仅在单一子shell临时加载`/etc/network_turbo`，有界`ls-remote/push/ls-remote`；非force push成功，远端新分支`personal/codex/rlt-dvac-pure-reference-bc`精确指向`cb88e9c5...`，未持久化代理。

### S006 首次真实smoke的配置错误与窄修

- v1使用正式单卡配置，smoke-only覆盖为1 cycle、warmup后8次update、最多20次update、save1，并试图以`val_check_interval=1000`跳过评估。
- 结果：模型/环境初始化、GPU0 placement和1个rollout均完成；随后`check_progress`拒绝`save_interval=1`与`val_check_interval=1000`不可整除，driver以断言退出。该错误发生在runner调度配置，不是Pure公式、模型、显存或RoboTwin执行错误。
- 源码复核：`val_check_interval<0`是项目原生的关闭评估方式，同时仍允许末步save；actor training位于checkpoint判断之前。
- 窄修：v2只将smoke覆盖改为`val_check_interval=-1`并使用全新v2输出目录；正式YAML与Python实现均不改。

### S007 v2 真实单卡 smoke 完成

- 指令：复用正式配置，只覆盖为1 cycle、关闭eval、warmup后8次critic/4次actor update并保存Step1。
- 结果：自然`exit0`，8条trajectory、2条成功；Step1完整checkpoint产生。Pure路径确认
  `success_target_reference=1`、`executed_target_ratio=0`。
- 方法：p05/mean/p95=`.448/1.000/1.499`，weight ESS=`.912`，top20 mass=`.277`；4份trace落盘。
- 资源：RAM峰值82.151 GiB，GPU0峰值20.711 GiB，GPU1空闲；CUDA OOM、worker crash、NCCL fatal均为0。

### S008 旧 matched-width pair 收尾

- 两条运行按用户授权停止于共同完整Step476，最新完整checkpoint均为Step475；运行目录和checkpoint保留。
- 截至共同Step476，control/旧DVAC-BC累计train success=`55.49/55.75%`，末5步=`87.5/92.5%`，
  末10步=`90.0/88.75%`；Step475 fixed20=`17/20 vs 15/20`。曲线反复交叉，未形成稳定领先。
- 轻量包：`exports/rlt_success_bc_matched_width_stopped_step476_high_info_20260827.zip`，
  1,016,570 bytes，SHA256=`C3D09D3B03949142D0D934C951F90D2081F4F73BEFC084EC6F490281053A56CC`；
  含日志、metrics、resolved config、资源、方法trace与图，不含约1.5 GB/条的checkpoint正文。
