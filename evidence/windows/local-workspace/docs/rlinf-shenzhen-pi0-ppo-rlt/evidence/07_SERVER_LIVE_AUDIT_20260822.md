# SZ-H100 双账号、训练、资源、网络与 Fast-WAM 只读现场审计

现场窗口：2026-08-22 10:28:31–10:33:10 CST  
机器：`SZ-H100` / `admin` / `120.241.223.9:22`  
边界：固定 host-key 的 Paramiko 密码认证；`chenyiteng` 普通视角与经认证的只读 sudo 探针，`toom`
管理员只读视角。没有停止、安装、下载、写服务器文件、修改服务或训练状态；密码只进入当前进程，未写入
命令、脚本或文档。

## 1. 可复现命令与退出

通用入口：

```powershell
C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  local_scripts/verified_password_ssh.py `
  --host 120.241.223.9 --port 22 --user <ACCOUNT> `
  --host-key-sha256 qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY `
  run --command-file <COMMAND_FILE>

C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  local_scripts/verified_password_ssh_sudo.py `
  --host 120.241.223.9 --port 22 --user <ACCOUNT> `
  --host-key-sha256 qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY `
  --command-file <COMMAND_FILE>
```

| command file | SHA256 | 账号 / exit | 作用 |
|---|---|---|---|
| `shenzhen_full_readonly_audit_chenyiteng_20260822.sh` | `b3a9429552de3186a7875d8f1b4780409a1475d25fc1c91d0959eda1fb19bdc2` | `chenyiteng / 0` | GPU、PPO、内存、磁盘、网络与 Fast-WAM 断点 |
| `shenzhen_full_readonly_audit_toom_20260822.sh` | `e81e33688f923c77bf441bd319743b8dbf06322949495541c9b5c24f02256bd1` | `toom / 0` | 账号、用户资源、服务、内核、认证与磁盘健康 |
| `shenzhen_mihomo_quota_sanitized_20260821.sh` | `9a0053c17275b32b1d1ce570cc883328a9c3bdb7a1e665214145d34bfadf71c2` | `toom / 1`，随后 `chenyiteng / 0` | 只输出 numeric quota/status |
| `shenzhen_rlinf_formal100_step_table_csv_20260822.sh` | `f48604c15743c5f1416cde9f1ade58ece89eb84e26517f9199219edba51c3b66` | `chenyiteng / 0` | TensorBoard Step 1–22 标量 |
| `shenzhen_rlinf_formal100_memory_trend_snapshot_20260822.sh` | `72d3b9ce0ad14457cc10dfe70db328d729bad9e48e99cd54b3324be5ede94c48` | `chenyiteng / 0` | 10:31 内存与 rollout 进度 |
| `shenzhen_readonly_audit_followup_chenyiteng_20260822.sh` | `0af8898879f940fec311c1d3dd642dbe1df2fcc3be6dffcc1ea23278be11863e` | `chenyiteng / 0` | 当前 sudo 合同与 ModelScope 首页探针 |
| `shenzhen_readonly_audit_followup_toom_20260822.sh` | `bc5e7233dc48f92df6dd8f8f9bb453b7491fe9b576c2966526c87c80948a8a71` | `toom / 0` | 今日 kernel/SSH 异常窄复核 |

`toom` 运行 quota command 的 exit 1 发生在 sudo 前：该账号无权读取位于 `chenyiteng` 私有运行目录中的
已锁 probe，SHA256 预检因此拒绝继续；没有远端副作用。保持同一 probe/hash，改由文件属主
`chenyiteng` 走 process-only sudo 后 exit 0。没有复制脚本、放宽权限或读取订阅 URL/token。

## 2. 账号、权限与其他用户

- 两账号 SSH 与固定 host key 均通过。
- `chenyiteng`：UID 1003，组为 `chenyiteng sudo labdata`；现场 `sudo -l` 为 `(ALL:ALL) ALL`，需要本人密码。
- `toom`：UID 1000，组含 `adm sudo lxd labdata`；现场 `sudo -l` 为 `(ALL:ALL) ALL`。
- 当前 `sudo` 组成员为 `toom, liwenbo, chenyiteng`；这是 live 事实，不能继续沿用“只有 toom 有 sudo”的旧描述。
- 其他用户没有 GPU compute app。约 10:29 的进程 RSS：`liwenbo≈1.47 GiB`、`zhangwei≈0.22 GiB`；
  `liwenbo` 有研究 watchdog/Codex/nvitop，`zhangwei` 有 Codex/tmux。它们不是本轮 PPO 的显存或主存主因。
- 今日 SSH journal 中 `accepted=39`，只涉及 `chenyiteng/toom` 且只有一个 accepted source；同时有
  `failed_password=[REDACTED]`、`invalid_user=144`、16 个失败来源。后者符合公网 22 端口的自动扫描形态，属于
  应关注的安全噪声，但本次没有看到未知账号成功登录证据；本轮未修改 SSH、防火墙或封禁策略。

## 3. GPU、PPO 与产物

- 物理 GPU 0–3 没有 compute process；瞬时显存约 `69/4/4/4 MiB`。GPU 3 从 GPU 隔离角度可供
  Fast-WAM 单卡使用。
- 物理 GPU 4–7 的全部 compute PID 均属于 `chenyiteng` 的 PPO Ray worker；10:28–10:29 瞬时显存约
  `62.3–64.6 GiB/卡`，剩余 `14.6–18.2 GiB/卡`。低瞬时 utilization 位于 simulator/rollout 相位，
  不等于 driver 退出。
- driver PID `1375834` alive；10:31 完整到 Step 22/100，Step 23 rollout 已达 `3/4`；fatal 扫描为 0。
- Step 22：train success `0.927734375`、KL `0.01363`、clip fraction `0.05286`、grad norm `28.01`、
  ratio `0.99926`、critic value loss `0.01963`、explained variance `0.41906`，全部 finite。
- checkpoint 仍为 `global_step_10`、`global_step_20`；run 约 `35G`。Step 30 尚未完成，因此没有
  `global_step_30` 并非保存失败。

## 4. 主存分解：真实匿名私有内存，不是 page cache

10:28 的训练 session cgroup：

```text
memory.current = 1,579,751,776,256 B = 1.437 TiB
memory.high/max = max
memory.swap.current = 0
memory.events high/max/oom/oom_kill = 0/0/0/0
host MemAvailable = 582,639,096 KiB = 555.6 GiB
```

相比同一 run 的 10:08 快照，20 分钟内 cgroup 增加约 `41.2 GiB`，host available 同量下降约
`41 GiB`。该区间处于 Step 23 rollout 内，不能单凭两点证明永久泄漏，但继续增长信号没有消失。

四个 EnvWorker 的 RSS 合计 `1,431,581,644 KiB = 1,365.3 GiB = 1.333 TiB`；单进程约
`300–372 GiB`。`smaps_rollup` 显示每个 worker 几乎全部为：

- `Pss ≈ RSS`；不是四进程重复计算的 shared mapping；
- `Pss_Anon / Anonymous / Private_Dirty` 占绝大部分；不是 Linux page cache 或 checkpoint 文件缓存；
- file-backed PSS 仅约 `0.32 GiB/worker`，shared clean 不到 `0.75 GiB/worker`；
- `VmSwap=0`，每个 worker 191 threads、约 784–819 FDs。

因此可以确定“增长主体位于 EnvWorker 进程的真实私有匿名内存”；现有只读证据还不能继续断言是
RoboTwin/SAPIEN/CuRobo 哪个对象未释放。CPU 整机仍约 94–95% idle、无 swap、无 OOM，所以当前是
高优先级黄灯而非已经崩溃；但它也不是普通文件 cache 可以自然忽略的占用。

## 5. 存储、网络、代理与系统健康

| mount | used / available | inode use |
|---|---|---:|
| `/` | 48 GiB / 236 GiB | 2% |
| `/home` | 83 GiB / 2.2 TiB | 1% |
| `/data` | 178 GiB / 3.1 TiB | 1% |

- Mihomo `active + enabled`，监听 `127.0.0.1:7890`；profile 仍向 login shell 注入本地 HTTP(S)/SOCKS 代理。
- 显式 profile 后 GitHub、PyPI、Hugging Face、ModelScope 首页均 HTTP 200；GitHub direct 与 proxy 都为
  HTTP 200。最初使用的某个 ModelScope API 测试路径返回 404，但首页复核 200，说明是 endpoint 选择而非网络失败。
- 订阅 live：total `100 GiB`、used `65.511 GiB`、remaining `34.489 GiB`，expiry
  `2026-08-23 14:01:48 CST`。相比上次 `52.962 GiB used` 又消耗约 `12.55 GiB`；total 与 expiry
  均未变化，现场没有看到“充值后扩容/续期”已经生效。
- 无 failed systemd unit；SSH、Mihomo、Docker、containerd 均 active。自 2026-08-22 00:00 起没有
  kernel NVRM/Xid、OOM、I/O/filesystem/NVMe severe match。
- journal 中 8 月 21 日的 GPU 0 Xid 13/43 对应已知 CuRobo illegal-instruction 调试；8 月 20 日还有旧
  NVRM IOVAS 清理 warning。它们不是当前 PPO 启动后的新错误。

## 6. Fast-WAM 精确恢复点与并行边界

```text
source  /data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
HEAD    7faa71108368fbb3b6885649f112af607427a2d4
status  detached clean
env     /home/chenyiteng/venvs/fastwam-7faa-py310-cu128
Python  3.10.20
cache   4.0 MiB
```

- source 约 11 MiB；env 约 228 MiB，只含 Python 与 packaging/pip/setuptools/wheel。
- torch、torchvision、Fast-WAM editable install 尚未完成；model/result 目录尚不存在；没有遗留 install PID。
- 因而可以从既定 `FW-SZ-101` continuation 续装，不应重跑 `conda create` 或 clone。
- GPU 3 与 PPO 4–7 不重叠；但当前 PPO 只剩约 556 GiB host available 且 EnvWorker 仍在增长。安装、
  小型探针在 GPU 隔离上无冲突，真正 Fast-WAM simulator/model inference 会再占主存并干扰 PPO 内存归因；
  是否并发进入真实推理应由主线资源决策明确，而不能只凭 GPU 3 空闲自动启动。
- official Fast-WAM 所需模型净体积约 24.84 GB（十进制），当前 34.489 GiB proxy quota 理论上可覆盖，
  但还需 Torch/依赖并且约 27.5 小时后到期，下载前必须继续使用 revision/hash 与 before/after quota 门。

## 7. 当前判断

训练的数值、checkpoint、GPU 与系统服务仍正常；唯一直接运行风险仍是 EnvWorker 私有匿名主存增长，而且
10:08→10:28 没有转为平台。GPU 0–3 的空闲是真实的，但“有空卡”不等于“与当前 PPO 完全无资源冲突”。
Fast-WAM 可安全从已存在的 base env 继续准备，真实单卡 simulator/inference 最好在 PPO 内存是否继续增长
有了决定后再启动。
