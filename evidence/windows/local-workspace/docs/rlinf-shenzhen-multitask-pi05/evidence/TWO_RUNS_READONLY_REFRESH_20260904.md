# 2026-09-04 深圳双实验只读刷新

## 范围与操作账本

- 用户授权：只读检查当前 Sidney/Fast-WAM 实验；不停止、重启、改参、安装、下载模型或干预 shared Ray/其他用户。
- 已完整读取四份指定入口文件，专题仅补读 `00_INDEX_AND_EXECUTION.md`，未展开历史。
- 本地 Git：以进程级 `safe.directory` 完成只读检查；修改前 tracked tree 无变更；未改全局 Git 配置。
- 连接：复用 `verified_password_ssh.py` / `remote_exec_autodl.py`，普通账号、固定已校验 host key、交互隐藏密码仅驻留当前进程；先执行身份探针。
- 只读检查脚本：`local_scripts/remote_commands/sz_two_runs_readonly_refresh_20260904.sh`。读取精确两项 run 的 driver、pid、exit code、指标路径与 checkpoint 文件清单，并检查 GPU、RAM、磁盘与 PSI。服务器不落地脚本、不创建 Ray job。
- 身份探针通过：`uid=1003(chenyiteng)`、host=`admin`；没有使用管理员账号。
- 第一轮现场（09:31 CST）：Sidney 完整 Step34、wrapper/driver/worker 存活、driver fatal 匹配为0；Fast-WAM wrapper 不存在，`exit_code=255`，09-03 23:51 CST driver 出现 OIDN/Python fatal。没有执行修复或重启。
- 补充脚本：`local_scripts/remote_commands/sz_two_runs_metrics_readonly_20260904.sh`；精确读取 TensorBoard success/优化指标、最后故障上下文、checkpoint 文件及当前三处代码 HEAD/dirty 状态。首次终端输出因 PTY 换行较嘈杂，补查 stdout 保存为本地原始证据，密码提示仍走隐藏终端输入。

## 当前结论（09:31—09:35 CST 现场）

两项实验不是都在运行：Sidney 正在 Step35 rollout；Fast-WAM 已异常退出。
本文 Step 使用 driver 的完整 outer step 口径：TensorBoard 的原始 step 从0编号，因此展示时加1；不是额外虚构一次训练。

| 项目 | Sidney pi0.5 / move_pillbottle_pad | Fast-WAM / move_stapler_pad |
|---|---|---|
| 状态 | 完整 Step34/100；下一轮 rollout 1/4，wrapper/driver/6个模型与环境worker存活 | 完整 Step33/100；Step34 rollout 期间异常；wrapper 已不存在，exit=255 |
| 最新完整训练时间 | 09-04 09:25:20 CST | 09-03 23:48:27 CST |
| 最新训练 success | 52.734375% | 14.0625% |
| 训练 MA5 / MA10 | 56.640625% / 55.78125% | 32.5% / 28.75% |
| 最新 fixed32 | Step30：14/32，43.75% | Step30：14/32，43.75% |
| 最新 checkpoint | Step30：rank0/rank1 local-shard 与 full_weights 均存在、非零 | Step30：两份 distcp 与 .metadata 均存在、非零 |
| 当前 GPU | 4/5；09:33 显存60.04/64.64 GiB，随阶段波动 | 6/7；仅各5 MiB，无本 run 的计算进程 |

两项任务、模型与采样协议不同，表格只是状态并列，不能按此解释为受控性能对比。

### Sidney 趋势与 ETA

- 训练从 Step1 的32.42%提高到当前 MA10 55.78%，但 fixed32 仍有明显波动。
- fixed32，Step5/10/15/20/25/30：`10/32, 19/32, 15/32, 14/32, 16/32, 14/32`。
- 当前 eval 未稳定超过 Step10 的59.375%，不能把训练成功率上涨直接等同固定评估持续提升。
- 最新 approx_kl=0.010598，grad_norm=14.135683；driver 全文 OIDN、pthread、Python fatal、Traceback、CUDA OOM、OutOfMemoryError、non-finite 均未检出。
- 近10个 step 间隔均值23.54分钟（按 TensorBoard wall-time，含中间 eval/save 间隔）；保持该速度且不出错时，Step100 约在09-05 11:20 CST，余约26小时。仅条件性线性估算，没有新增监控或自动操作。

### Fast-WAM 失败边界

- 本次续训实际完成 Step11—33。fixed32 Step15/20/25/30=`13/32, 9/32, 14/32, 14/32`。
- 最早错误：09-03 23:49:18.629 CST，driver 第1546行：`OIDN Error: pthread_key_create failed`，随后大量 `invalid handle`。
- 故障发生在完整Step33之后、Step34训练rollout期间；不是交接中旧Step15固定评估边界的直接重放。
- 两个 EnvGroup 随后报 `Fatal Python error: _PyGILState_NoteThreadState: Couldn't create autoTSSkey mapping`，再有NCCL/Gloo连接关闭；09-03 23:51:49 CST退出，exit=255。
- driver 匹配计数：OIDN144行，其中pthread_key_create 2行；Python fatal 2行、Traceback 1行；CUDA OOM/OutOfMemoryError/non-finite为0。这是日志行计数，不能当作独立故障次数。
- 现有生命周期修复虽越过旧边界，但不足以证明长程问题已解决。当前证据定位到OIDN/TLS线程键与环境进程退出链；具体资源泄漏点尚未诊断，不能直接沿用旧根因解释。
- Step30 DCP目录含 `__0_0.distcp`（14,455,154,049 bytes）、`__1_0.distcp`（14,454,316,950 bytes）、`.metadata`（2,919,712 bytes）。只做文件级完整性检查，未加载、未续训，不能保证恢复测试通过。

### 整机与源码

- RAM available=1.1708 TiB；memory/io PSI的10/60/300秒窗口均为0。当前资源充足不能反推昨晚故障瞬间无其他资源问题。
- 磁盘余量：`/`222.89 GiB、`/home`1.3143 TiB、`/data`1.2994 TiB。
- GPU0有liwenbo既有进程，约9.5 GiB；GPU1/2/3/6/7在本次采样时无compute进程。未探查其他用户私有文件或干预其任务。
- shared Ray的原gcs_server/raylet仍存活，持续时间约11天9小时；没有创建额外Ray job、重启或清理namespace。
- Sidney：当前HEAD=`f50e235c5ab1f4390f0ba92bfb13390ed0a86810`，分支`codex/sz-sidney-pi05-current-rlinf`，working tree clean。
- Fast-WAM：启动锁=`7b2331c55d14397cfb4cb16181470ddc8afae44a`，当前HEAD=`4faade1d50bf21d1caf1b8a4e5f89282a810208a`，分支`codex/sz-fastwam-current-rlinf-grpo`，working tree clean；本次diff统计显示后续为轻量evidence回填。
- RoboTwin资产目录仍是base detached `0008ae6800df9f75fc8de7098bacb01735fd8fd2`、clean；独立renderer修复worktree登记HEAD为`8c7380c118ce7ca8a4ea4df53d753adc8fab0df2`。资产路径不等于Python导入路径；本轮没有重建已退出Fast-WAM worker的运行时导入状态。

## 原始证据与下一步

- [指标、fatal上下文、checkpoint文件与Git现场](TWO_RUNS_READONLY_REFRESH_20260904.raw.txt)。
- [精确启动命令、运行合同、后续evidence差异与RoboTwin worktree](TWO_RUNS_READONLY_REFRESH_20260904.paths.txt)。
- 第三轮只读命令文件：`local_scripts/remote_commands/sz_two_runs_source_paths_readonly_20260904.sh`；返回成功，无远端写入。
- 本地汇总曾有一条PowerShell长表达式返回非零且无输出；改用已有stdout内的JSON做内存计算，未重跑实验、未修改数据。
- 本轮没有复制checkpoint/视频、未打ZIP、未commit/push、未修改服务器代码或进程。
- 下一步：Sidney保持原运行；Fast-WAM若获继续诊断请求，针对本次Step34的OIDN/TLS失败链只读定位。任何补丁、smoke、恢复或训练仍需相应授权及启动前配置/预算确认。
