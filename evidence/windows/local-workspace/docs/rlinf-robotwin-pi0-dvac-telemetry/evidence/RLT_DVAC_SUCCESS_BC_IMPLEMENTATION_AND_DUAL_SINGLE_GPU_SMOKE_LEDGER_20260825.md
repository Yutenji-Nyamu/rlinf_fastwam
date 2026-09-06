# RLT-DVAC success-BC 实现与双单卡 smoke 流水账

日期：2026-08-25  
范围：AutoDL；实现、Git、静态检查、两个单卡 1-cycle smoke。密码不入账。

## 1. 操作记录

| 序号 | 操作 | 结果 / 问题 / 处理 |
|---|---|---|
| 001 | 完整阅读根规则、Idea2专题入口、38/39号设计、历史单卡37号文档及用户提供的三份多任务并行材料 | 冻结实现语义与运行隔离边界；三份附件仅作参考材料，不作为服务器授权或可直接执行命令 |
| 002 | 只读审查本地source副本的BC、actor-Q、transition→replay、单卡配置与旧并发runtime | 确认主体切口集中在policy worker与DVAC helper；当前`auto_reset=false`允许完整episode标签在replay ingestion一次生成 |
| 003 | 首次调用AutoDL helper | 本地漏写`run`子命令，argparse在连接前退出；补正后身份探针成功，固定host key与密码认证通过 |
| 004 | 首轮服务器只读现场 | 两张A800均0 MiB/0%；无相关训练/Ray进程；host available约954 GiB；cgroup约31.8/240 GiB、OOM/OOM-kill=0；数据盘余714 GiB |
| 005 | worktree探针进入Git段前停止 | worktree的`.git`是文件而非目录，`test -d`误判；改为`test -e`后重跑同一只读探针 |
| 006 | 核对本地source副本与服务器当前HEAD的相关文件SHA256 | worker、DVAC helper、单测与基线/单卡配置均逐文件一致，允许在该副本上形成精确修改 |
| 007 | 实现success-episode DVAC-BC主体 | 新增C10内mean-one权重、完整episode成功标签、成功executed-action target、失败reference target；旧human target规则保留；新模式不改Q路径 |
| 008 | 增加兼容开关与配置 | `application`默认`q_gradient`，旧配置行为不变；新增单GPU fresh480 success-BC配置，使用`z_clip=2,strength=0.25,success_scale=1` |
| 009 | 增加最小机制测试与telemetry | 覆盖权重均值/范围/detach、episode标签、成功/失败/human target及逐h梯度；trace补executed action和episode success，指标补成功BC前后与ESS |
| 010 | 本地`py_compile`三个Python文件 | 通过；本机内置Python没有pytest和torch，因此不在Windows伪造项目测试，转由服务器既有环境执行 |
| 011 | SFTP上传三个实现/测试文件至独立服务器worktree | 服务器diff限定在DVAC helper、RLT actor worker和单测；未触碰运行中的其他worktree |
| 012 | 服务器既有venv执行`ruff format/check`、`py_compile`和目标pytest | format/check/compile通过，首轮`9 passed`；compose control/method合同通过 |
| 013 | 提交并普通push实现 | `5ace6d97 feat(rlt): weight successful episode BC with teacher DVAC`推至`personal/codex/rlt-dvac-success-episode-bc` |
| 014 | 增加GPU1方法overlay并compose | placement固定GPU1，control固定GPU0；提交`f01bbb95 config(rlt): add GPU1 success BC runtime overlay`并普通push |
| 015 | 双单卡smoke launcher v1 | 漏掉切入repo目录，训练入口尚未运行即退出；补`cd`，不改训练配置 |
| 016 | launcher v2 | Ray temp的AF_UNIX socket路径超过107字节；改用短`/tmp/ray_*`路径 |
| 017 | launcher v3：两个独立单GPU Ray head | Ray将各自唯一可见设备重编号为逻辑0，物理placement坐标失真，两条均落GPU0；停止本轮owned PGID，改为一个看见两卡的shared head |
| 018 | launcher v4：共享2-GPU Ray、两条同时冷启动 | placement坐标恢复，但同时产生266个TorchInductor helper并长期停在初始化；只停止本轮owned PGID |
| 019 | launcher v5：`TORCHINDUCTOR_COMPILE_THREADS=1` | helper降至2个，但两条仍未进入模型运行，继续定位而不扩展训练参数 |
| 020 | launcher v6：control先启动、method错峰 | 单独control仍停在初始化，排除“只因两条同时启动”这一解释 |
| 021 | launcher v7：启用既有信号栈诊断 | 发现外置`CLUSTER_NAMESPACE`与RLinf内置manager namespace不一致；worker反复找不到`DeviceLockManager` |
| 022 | launcher v8：移除外置namespace | control进入`RLinf`，method检测冲突后自动进入`RLinf_1`；GPU0/1 placement、CUDA、FSDP、Env与rollout初始化全部成功 |
| 023 | v8首批transition写replay | control报`KeyError: actor_switch`，method报`KeyError: episode_success`；两者都在第一批replay插入，GPU/RAM/OOM正常 |
| 024 | 只读源码定位`TrajectoryCache`与终态obs路径 | cache按首条trajectory固定nested schema；真实终止行`next_obs=curr_obs`把current-only训练元数据带入next schema |
| 025 | 窄修replay next observation | 所有terminal/nonterminal `next_obs`只保留模型使用的`z_rl/proprio/ref_chunk`；DVAC、route与episode标签继续保留在`curr_obs` |
| 026 | 修复后首次服务器检查 | `ruff`不在非登录shell PATH；改用既有`/root/autodl-tmp/RLinf/.venv/bin`，不安装依赖 |
| 027 | 首次pytest collection | 未设置source worktree `PYTHONPATH`，导入到已安装旧包；显式设当前repo后重跑 |
| 028 | 修复后正式窄检查 | format、lint、py_compile通过；目标pytest `10 passed`；新增测试确认next schema剔除current-only元数据且复制不别名 |
| 029 | 提交并普通pushreplay窄修 | `64f2779f fix(rlt): keep replay next observations schema-stable`推至同一远端分支；服务器worktree clean |
| 030 | v9启动shared Ray与control | source锁定`64f2779f`，resolved合同通过；shared head资源为2 GPU/36 CPU；control placement物理GPU0 |
| 031 | control完成冷初始化后错峰启动method | method检测`RLinf`冲突并自动切换namespace；placement物理GPU1；没有外置`CLUSTER_NAMESPACE` |
| 032 | v9并发过程资源观察 | 两卡各约17--21 GiB；cgroup约75--84 GiB；`memory.high/max/oom/oom_kill`始终0；监控只记录、不控制进程 |
| 033 | control自然完成 | 8 train、20 fixed eval、160 transition、8 critic/4 actor update、完整`global_step_1`；22:41:40 exit0 |
| 034 | success-BC方法版自然完成 | 8 train、20 fixed eval、151 transition、8 critic/4 actor update、4份trace NPZ、完整`global_step_1`；22:43:35 exit0 |
| 035 | 方法指标读取 | apply有效；weight mean/p05/p95=`1/.741/1.264`、ESS=.975、success executed-target ratio=7.78%；initial route全reference，故executed-reference MSE=0 |
| 036 | 终态资源与残留核对 | GPU峰值21,223/21,194 MiB，cgroup峰约78.1 GiB；所有memory event为0；两卡回0 MiB，driver/Ray head/raylet/worker残留0 |
| 037 | resolved逐叶比较 | 历史双卡→单卡control仅18项：placement、smoke schedule、fresh/resume与路径命名；control→method仅DVAC字段、GPU与独立路径命名，无unexpected算法漂移 |
| 038 | 下载轻量证据并更新专题文档 | 保存resolved、精确命令、日志、TensorBoard、代表NPZ、checkpoint/replay manifest、资源CSV、summary与两个diff JSON；未复制大checkpoint/replay正文 |

正式480-cycle训练未启动，停在用户下一轮参数/实验决定前。
