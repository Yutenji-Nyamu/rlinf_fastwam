# 深圳服务器简要只读审计：2026-09-05 11:05 CST

## 1. 范围与证据

本窗口用于审计与讨论。以 `chenyiteng` 密码认证成功，固定 SSH host-key 与既有记录一致；服务器 `admin`，UID 1003，工作目录 `/home/chenyiteng`。本轮未使用管理员账号，未修改服务器文件、训练配置、进程或服务，未启动测试/训练。其他用户仅记录普通账号可见的 GPU/进程资源摘要，没有读取私有项目或日志。

- 采样时间：2026-09-05 11:05:09—11:05:15 CST。
- 原始小型证据：[SZ_BRIEF_AUDIT_20260905.json](SZ_BRIEF_AUDIT_20260905.json)。
- 精确采集命令：[sz_brief_audit_20260905.sh](../../local_scripts/remote_commands/sz_brief_audit_20260905.sh)。脚本经固定指纹 Paramiko SSH 传入执行，无远端脚本落盘。
- 操作记录：完整读取根上下文/交接与当前路由；11:02:47执行 `hostname; pwd; id; date -Is` 身份探针；11:05执行只读快照（TensorBoard标量、日志、checkpoint文件元数据、nvidia-smi、/proc、df、vmstat、systemctl）；本地保存47 KB JSON并整理本简报。
- 本地初始 `git status` 遇到工作区属主检查，后用本条命令级 `-c safe.directory=...` 只读查看；没有更改全局Git配置，没有混入现有未跟踪文件。本轮不报告未刷新Git HEAD/dirty或SMART/内核日志为当前事实。

## 2. 实验现状

| 实验 | 当前运行状态 | 最近训练与固定评估 | 保存证据 |
|---|---|---|---|
| Sidney π0.5 GRPO，move_pillbottle_pad，GPU4/5 | 完整95/100步，driver 3176215在，进入第96步采样 | 最新188/256=73.44%；最近10步均值67.42%；Step95 fixed22/32=68.75%，最近10次fixed最高Step70=26/32 | Step90双rank `.pt` 和 `full_weights.pt` 均在 |
| Fast-WAM GRPO，scene-fence-v3 | 完整17/100步；09-05 04:47:11退出255；原driver不在 | 最新65/256=25.39%；MA10=26.33%；Step5/10/15 fixed11/11/13 /32 | Step10双 `.distcp` 和 `.metadata` 均在 |
| π0在线成功BC，GPU6 smoke v2 | 09-05 10:52:47启动，11:01:04退出255；原wrapper/driver不在，GPU6释放 | 未产生完整训练轮标量；SFT训练预处理中 `grid_sample` 报 `expected scalar type BFloat16 but found Float` | 所查run下未发现 `global_step_*` checkpoint，不能称smoke通过 |

TensorBoard原始step从0开始，本表outer Step=TB step+1，并用driver `Global Step`交叉核对。train与fixed评估分别报告；不把训练73.44%当成固定评估结果。checkpoint仅核对文件与大小，没有恢复测试。

Sidney日志所查OIDN/pthread/fatal/OOM/RuntimeError/退出异常关键字均未检出；当前fixed仍波动，但近期已高于早期多次结果。Fast首个fatal为 `PyGILState_Release: auto-releasing thread-state, but no thread-state for this thread`，随后是通信peer关闭；本轮不进一步推断其与旧OIDN或挂帧故障的因果。BC未检出OOM，其错误位于OpenPI训练图像增强/采样的dtype边界；本轮只记录，不实施修复或重启。

三个run的完整路径、原日志尾、关键字计数与checkpoint文件清单见JSON的 `runs`。本次只盘点近期三项和现场GPU任务，没有重审所有历史实验。

## 3. 整机资源与健康

- 硬件：8×NVIDIA H100 80GB，128逻辑CPU，RAM约2015.5 GiB；开机约16.95天。
- GPU0：liwenbo进程916861，StarVLA环境Python，占约9.6 GiB；当前利用率0%，仅凭此瞬时值不能判断实验完成或停滞。
- GPU4/5：Sidney，显存67.08/66.23 GiB，瞬时利用率66%/1%。其余GPU1/2/3/6/7均无compute进程，仅4—8 MiB常驻；GPU7既有预留仍保留。
- CPU：load1/5/15=8.16/9.08/8.93（128逻辑CPU）；两次1秒采样idle94—95%，iowait0。
- RAM available1104.77 GiB；Sidney两EnvWorker RSS约394.84/420.87 GiB，仍是主要内存占用者。瞬时内存PSI各窗口0；IO PSI avg10/60为0.02%/0.03%，没有明显即时压力。
- Swap已用5.81 GiB，接近容量，但本次两次瞬时采样si/so均0；不能仅由累计swap占用判定当前内存紧张。
- 磁盘：`/`余220.98 GiB（22%已用）、`/home`余1262.99 GiB（46%已用）、`/data`余731.04 GiB（79%已用）；inode均充足。后续大量checkpoint保存应关注`/data`余量，本轮不清理。
- GPU1可纠正SRAM计数2；8卡不可纠正ECC均0，row-remap pending/failure均No。该项没有复核NVMe SMART或管理员内核日志。
- ssh/mihomo均active，systemd failed units为0；shared Ray原gcs_server321933/raylet322685存活约12天10小时，没有重启。

## 4. 当前停点

本轮审计完成。Sidney继续自然运行；Fast与BC当前退出事实已回写根交接。下一次讨论可分别针对BC dtype边界、Fast线程状态fatal、Sidney最终fixed结果展开；本窗口没有获得或执行新修复、训练、清理、安装或进程干预。
